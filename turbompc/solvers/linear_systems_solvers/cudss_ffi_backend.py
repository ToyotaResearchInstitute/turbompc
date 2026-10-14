"""cuDSS solver for block-tridiagonal systems via CUDA FFI."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from turbompc.solvers.ffi_loader import load_ffi_lib

_LIB_NAME = "libcudss_blktridi_ffi.so"
_LIB = load_ffi_lib(_LIB_NAME, env_var="CUDSS_BLKTRIDI_FFI_LIB")
jax.ffi.register_ffi_target(
    "cudss_blktridi_workspace_create",
    jax.ffi.pycapsule(_LIB.TurboMpcCreateCudssBlockTridiWorkspace),
    platform="CUDA",
)
jax.ffi.register_ffi_target(
    "cudss_blktridi_workspace_ensure_capacity",
    jax.ffi.pycapsule(_LIB.TurboMpcEnsureCudssBlockTridiWorkspaceCapacity),
    platform="CUDA",
)
jax.ffi.register_ffi_target(
    "cudss_blktridi_workspace_destroy",
    jax.ffi.pycapsule(_LIB.TurboMpcDestroyCudssBlockTridiWorkspace),
    platform="CUDA",
)

jax.ffi.register_ffi_target(
    "cudss_blktridi_cuda",
    jax.ffi.pycapsule(_LIB.CudssBlkTridiCuda),
    platform="CUDA",
)
jax.ffi.register_ffi_target(
    "cudss_blktridi_cuda_solve_only",
    jax.ffi.pycapsule(_LIB.CudssBlkTridiCudaSolveOnly),
    platform="CUDA",
)


@jax.tree_util.register_pytree_node_class
class CudssBlockTridiWorkspace:
    """Own one lazily sized native cuDSS workspace with a stable handle."""

    def __init__(self) -> None:
        self._handle = 0
        self._device = jax.config.jax_default_device
        if getattr(self._device, "platform", None) != "gpu":
            self._device = jax.local_devices(backend="gpu")[0]
        with jax.ensure_compile_time_eval(), jax.default_device(self._device):
            result = jax.ffi.ffi_call(
                "cudss_blktridi_workspace_create",
                jax.ShapeDtypeStruct((2,), np.uint32),
                has_side_effect=True,
            )()
            result.block_until_ready()
            self._handle = int(np.asarray(result).view(np.uint64).item())

    def tree_flatten(self):
        return (), self

    @classmethod
    def tree_unflatten(cls, owner, children):
        del children
        return owner

    @property
    def closed(self) -> bool:
        return self._handle == 0

    @property
    def handle(self) -> int:
        if self.closed:
            raise RuntimeError("cuDSS workspace is closed")
        return self._handle

    def solve(self, S, rhs, *, factorize=True):
        return cudss_ffi_solve(S, rhs, workspace=self, factorize=factorize)

    def ensure_capacity(self, batch_size: int, T: int, n: int, dtype) -> None:
        """Eagerly size the workspace; solve calls do this automatically."""
        dtype = jnp.dtype(dtype)
        if batch_size < 1 or T < 1 or n < 1:
            raise ValueError("cuDSS workspace dimensions must be positive")
        if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise TypeError("cuDSS workspace dtype must be float32 or float64")
        with jax.ensure_compile_time_eval(), jax.default_device(self._device):
            result = jax.ffi.ffi_call(
                "cudss_blktridi_workspace_ensure_capacity",
                jax.ShapeDtypeStruct((), np.uint32),
                has_side_effect=True,
            )(
                workspace_handle=np.int64(self.handle),
                batch_size=np.int64(batch_size),
                T=np.int64(T),
                n=np.int64(n),
                dtype_bytes=np.int64(dtype.itemsize),
            )
            result.block_until_ready()

    def close(self) -> None:
        if self.closed:
            return
        with jax.ensure_compile_time_eval(), jax.default_device(self._device):
            result = jax.ffi.ffi_call(
                "cudss_blktridi_workspace_destroy",
                jax.ShapeDtypeStruct((), np.uint32),
                has_side_effect=True,
            )(workspace_handle=np.int64(self.handle))
            result.block_until_ready()
        self._handle = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass


jax.ffi.register_ffi_target(
    "cudss_blktridi_cuda_f64",
    jax.ffi.pycapsule(_LIB.CudssBlkTridiCudaF64),
    platform="CUDA",
)
jax.ffi.register_ffi_target(
    "cudss_blktridi_cuda_solve_only_f64",
    jax.ffi.pycapsule(_LIB.CudssBlkTridiCudaSolveOnlyF64),
    platform="CUDA",
)


def _cudss_ffi_solve_impl(
    S: jax.Array,
    rhs: jax.Array,
    *,
    workspace: CudssBlockTridiWorkspace,
    factorize=True,
) -> jax.Array:
    """Solve block-tridiagonal system S @ x = rhs via cuDSS FFI.

    Supports both unbatched and batched inputs:
        Unbatched: S (T,n,3n), rhs (T,n) -> x (T,n)
        Batched:   S (Nb,T,n,3n), rhs (Nb,T,n) -> x (Nb,T,n)

    Set ``factorize=False`` to reuse the workspace's existing numerical
    factorization for a new right-hand side. A workspace must be factorized
    before it can be reused.
    """
    dtype = S.dtype
    factorize = jnp.asarray(factorize, dtype=jnp.bool_)
    if factorize.shape != ():
        raise ValueError("factorize must be a scalar boolean")
    batched = S.ndim == 4

    if batched:
        Nb, T, n, _ = S.shape
    else:
        T, n, _ = S.shape
        Nb = 1

    # Determine kernel dtype: support f32 and f64 natively
    if dtype == jnp.float64:
        kernel_dtype = jnp.float64
        factorize_target = "cudss_blktridi_cuda_f64"
        solve_target = "cudss_blktridi_cuda_solve_only_f64"
    else:
        kernel_dtype = jnp.float32
        factorize_target = "cudss_blktridi_cuda"
        solve_target = "cudss_blktridi_cuda_solve_only"

    S_k = jnp.asarray(S, dtype=kernel_dtype)
    rhs_k = jnp.asarray(rhs, dtype=kernel_dtype)

    if batched:
        out_x = jax.ShapeDtypeStruct((Nb, T, n), kernel_dtype)
    else:
        out_x = jax.ShapeDtypeStruct((T, n), kernel_dtype)

    def call(target):
        return jax.ffi.ffi_call(
            target,
            out_x,
            vmap_method="broadcast_all",
        )(S_k, rhs_k, workspace_handle=workspace.handle)

    x_out = jax.lax.cond(
        factorize,
        lambda: call(factorize_target),
        lambda: call(solve_target),
    )
    return jnp.asarray(x_out, dtype=dtype)


def cudss_ffi_solve(
    S: jax.Array,
    rhs: jax.Array,
    *,
    workspace: CudssBlockTridiWorkspace,
    factorize=True,
) -> jax.Array:
    """Solve one system, combining surrounding ``vmap`` axes natively."""

    def solve_one(S, rhs, factorize):
        return _cudss_ffi_solve_impl(S, rhs, workspace=workspace, factorize=factorize)

    solve_one = jax.custom_batching.custom_vmap(solve_one)

    @solve_one.def_vmap
    def solve_one_vmap(axis_size, in_batched, S, rhs, factorize):
        S_batched = S if in_batched[0] else jnp.broadcast_to(S, (axis_size,) + S.shape)
        rhs_batched = (
            rhs if in_batched[1] else jnp.broadcast_to(rhs, (axis_size,) + rhs.shape)
        )
        factorize_batched = (
            factorize if in_batched[2] else jnp.broadcast_to(factorize, (axis_size,))
        )
        batch_shape = S_batched.shape[:-3]
        S_flat = S_batched.reshape((-1,) + S_batched.shape[-3:])
        rhs_flat = rhs_batched.reshape((-1,) + rhs_batched.shape[-2:])
        # cuDSS factorizes the combined block-diagonal matrix as a unit. If any
        # mapped solve needs a fresh factorization, refreshing every block is
        # correct and preserves one native call.
        output_flat = solve_one(S_flat, rhs_flat, jnp.any(factorize_batched))
        output = output_flat.reshape(batch_shape + output_flat.shape[-2:])
        return output, True

    return solve_one(S, rhs, factorize)
