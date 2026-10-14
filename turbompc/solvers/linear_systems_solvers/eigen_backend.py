"""JAX FFI wrapper for the Eigen block-tridiagonal solver."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

try:
    from turbompc.solvers.linear_systems_solvers import _eigen_block_tridi
except (ImportError, OSError) as exc:
    _eigen_block_tridi = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

if _eigen_block_tridi is not None:
    for _name, _target in _eigen_block_tridi.registrations().items():
        jax.ffi.register_ffi_target(_name, _target, platform="cpu")


def _extension():
    if _eigen_block_tridi is None:
        raise ImportError(
            "Eigen backend extension is not built. Install Eigen headers, then "
            "reinstall: `TURBOMPC_EIGEN=ON pip install .` (set EIGEN_INCLUDE_DIR "
            "if the headers are in a non-standard location)."
        ) from _IMPORT_ERROR
    return _eigen_block_tridi


@jax.tree_util.register_pytree_node_class
class EigenBlockTridiWorkspace:
    """Own one lazily sized native Eigen workspace with a stable handle."""

    def __init__(self) -> None:
        extension = _extension()
        handle = int(extension.create_workspace())
        if handle <= 0:
            raise RuntimeError("native Eigen workspace creation failed")

        self._extension = extension
        self._handle = handle

    def tree_flatten(self):
        return (), self

    @classmethod
    def tree_unflatten(cls, owner, children):
        del children
        return owner

    @property
    def closed(self) -> bool:
        """Whether the native workspace has been released."""
        return self._handle == 0

    @property
    def handle(self) -> int:
        """Opaque native handle passed to the JAX FFI handler."""
        if self.closed:
            raise RuntimeError("Eigen workspace is closed")
        return self._handle

    def solve(self, blocks, rhs, *, factorize):
        """Solve with the factorization storage owned by this workspace."""
        return eigen_solve(blocks, rhs, factorize=factorize, workspace=self)

    def ensure_capacity(self, batch_size: int, T: int, n: int, dtype) -> None:
        """Eagerly size the workspace; solve calls do this automatically."""
        dtype = jnp.dtype(dtype)
        if batch_size < 1 or T < 1 or n < 1:
            raise ValueError(
                "Eigen workspace batch size and dimensions must be positive"
            )
        if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise TypeError("Eigen workspace dtype must be float32 or float64")
        self._extension.ensure_workspace_capacity(
            self.handle,
            int(batch_size),
            int(T),
            int(n),
            8 if dtype == jnp.dtype(jnp.float64) else 4,
        )

    def close(self) -> None:
        """Release the native factorization and workspace storage."""
        if self.closed:
            return
        handle = self._handle
        self._handle = 0
        self._extension.destroy_workspace(handle)

    def __enter__(self) -> EigenBlockTridiWorkspace:
        """Return this live workspace from a context manager."""
        if self.closed:
            raise RuntimeError("Eigen workspace is closed")
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release native storage when leaving a context manager."""
        self.close()

    def __del__(self) -> None:
        """Release native storage when Python destroys the owner."""
        try:
            self.close()
        except Exception:
            pass


def _solve_host(workspace: EigenBlockTridiWorkspace, blocks, rhs, factorize):
    extension = _extension()
    if blocks.dtype == np.float32:
        return extension.solve_float(workspace.handle, blocks, rhs, bool(factorize))
    return extension.solve_double(workspace.handle, blocks, rhs, bool(factorize))


def _eigen_solve_impl(blocks, rhs, *, factorize, workspace: EigenBlockTridiWorkspace):
    """Solve ``blocks @ x = rhs`` with an explicitly owned Eigen workspace."""
    _extension()
    if not isinstance(workspace, EigenBlockTridiWorkspace):
        raise TypeError("workspace must be an EigenBlockTridiWorkspace")
    blocks = jnp.asarray(blocks)
    rhs = jnp.asarray(rhs)
    factorize = jnp.asarray(factorize, dtype=jnp.bool_)
    if blocks.ndim not in (3, 4) or rhs.ndim != blocks.ndim - 1:
        raise ValueError("expected blocks (..., T, n, 3n) and rhs (..., T, n)")
    T, n, width = blocks.shape[-3:]
    if width != 3 * n or rhs.shape != blocks.shape[:-1]:
        raise ValueError("expected blocks (..., T, n, 3n) and rhs (..., T, n)")
    if factorize.shape != ():
        raise ValueError("factorize must be a scalar boolean")
    if blocks.dtype != rhs.dtype or blocks.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("blocks and rhs must have the same float32 or float64 dtype")

    output = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)

    def solve_finite(_):
        # CPU uses FFI directly. Other backends transfer through a host callback.
        if jax.default_backend() != "cpu":
            return jax.pure_callback(
                partial(_solve_host, workspace),
                output,
                blocks,
                rhs,
                factorize,
                vmap_method="sequential",
            )

        target = (
            "eigen_block_tridi_cpu_f64"
            if blocks.dtype == jnp.float64
            else "eigen_block_tridi_cpu"
        )
        return jax.ffi.ffi_call(target, output, vmap_method="sequential")(
            blocks, rhs, factorize, workspace_handle=workspace.handle
        )

    # only solve via eigen if inputs are finite
    finite = jnp.all(jnp.isfinite(blocks)) & jnp.all(jnp.isfinite(rhs))
    return jax.lax.cond(
        finite, solve_finite, lambda _: jnp.full_like(rhs, jnp.nan), None
    )


def eigen_solve(blocks, rhs, *, factorize, workspace: EigenBlockTridiWorkspace):
    """Solve one system, combining surrounding ``vmap`` axes natively."""

    def solve_one(blocks, rhs, factorize):
        return _eigen_solve_impl(blocks, rhs, factorize=factorize, workspace=workspace)

    solve_one = jax.custom_batching.custom_vmap(solve_one)

    @solve_one.def_vmap
    def solve_one_vmap(axis_size, in_batched, blocks, rhs, factorize):
        blocks_batched = (
            blocks
            if in_batched[0]
            else jnp.broadcast_to(blocks, (axis_size,) + blocks.shape)
        )
        rhs_batched = (
            rhs if in_batched[1] else jnp.broadcast_to(rhs, (axis_size,) + rhs.shape)
        )
        factorize_batched = (
            factorize if in_batched[2] else jnp.broadcast_to(factorize, (axis_size,))
        )
        batch_shape = blocks_batched.shape[:-3]
        blocks_flat = blocks_batched.reshape((-1,) + blocks_batched.shape[-3:])
        rhs_flat = rhs_batched.reshape((-1,) + rhs_batched.shape[-2:])
        output_flat = solve_one(blocks_flat, rhs_flat, jnp.any(factorize_batched))
        output = output_flat.reshape(batch_shape + output_flat.shape[-2:])
        return output, True

    return solve_one(blocks, rhs, factorize)
