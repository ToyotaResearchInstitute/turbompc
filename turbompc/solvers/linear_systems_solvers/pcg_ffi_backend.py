"""PCG solver for block-tridiagonal systems via CUDA FFI."""
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from turbompc.solvers.ffi_loader import load_ffi_lib

_LIB_NAME = "libpcg_blktridi_ffi.so"
_LIB = load_ffi_lib(_LIB_NAME)

jax.ffi.register_ffi_target(
    "pcg_blktridi_cuda",
    jax.ffi.pycapsule(_LIB.PcgBlkTridiCuda),
    platform="CUDA",
)
jax.ffi.register_ffi_target(
    "pcg_blktridi_cuda_f64",
    jax.ffi.pycapsule(_LIB.PcgBlkTridiCudaF64),
    platform="CUDA",
)


def _pcg_ffi_solve_impl(
    S: jax.Array,
    Phiinv: jax.Array,
    rhs: jax.Array,
    x0: Optional[jax.Array] = None,
    *,
    eps: float = 1e-8,
    max_iters: int = 200,
) -> Tuple[jax.Array, jax.Array]:
    """Solve block-tridiagonal system S @ x = rhs via CUDA PCG FFI.

    Supports both unbatched and batched inputs:
        Unbatched: S (T,n,3n), Phiinv (T,n,3n), rhs (T,n) -> x (T,n), iters ()
        Batched:   S (Nb,T,n,3n), Phiinv (Nb,T,n,3n), rhs (Nb,T,n) -> x (Nb,T,n), iters (Nb,)
    """
    dtype = S.dtype
    batched = S.ndim == 4

    if batched:
        Nb, T, n, _ = S.shape
    else:
        T, n, _ = S.shape

    # Determine kernel dtype: support f32 and f64 natively
    if dtype == jnp.float64:
        kernel_dtype = jnp.float64
        ffi_target = "pcg_blktridi_cuda_f64"
    else:
        kernel_dtype = jnp.float32
        ffi_target = "pcg_blktridi_cuda"

    if x0 is None:
        if batched:
            x0 = jnp.zeros((Nb, T, n), dtype=kernel_dtype)
        else:
            x0 = jnp.zeros((T, n), dtype=kernel_dtype)

    S_k = jnp.asarray(S, dtype=kernel_dtype)
    Phiinv_k = jnp.asarray(Phiinv, dtype=kernel_dtype)
    rhs_k = jnp.asarray(rhs, dtype=kernel_dtype)
    x0_k = jnp.asarray(x0, dtype=kernel_dtype)

    if batched:
        out_x = jax.ShapeDtypeStruct((Nb, T, n), kernel_dtype)
        out_it = jax.ShapeDtypeStruct((Nb,), jnp.uint32)
    else:
        out_x = jax.ShapeDtypeStruct((T, n), kernel_dtype)
        out_it = jax.ShapeDtypeStruct((1,), jnp.uint32)

    call = jax.ffi.ffi_call(
        ffi_target,
        [out_x, out_it],
        vmap_method="broadcast_all",
    )

    x_out, iters = call(
        S_k, Phiinv_k, rhs_k, x0_k, max_iters=int(max_iters), eps=float(eps)
    )

    x_out = jnp.asarray(x_out, dtype=dtype)

    if batched:
        return x_out, iters  # (Nb, T, n), (Nb,)
    else:
        return x_out, iters[0]  # (T, n), ()


def pcg_ffi_solve(
    S: jax.Array,
    Phiinv: jax.Array,
    rhs: jax.Array,
    x0: Optional[jax.Array] = None,
    *,
    eps: float = 1e-8,
    max_iters: int = 200,
) -> Tuple[jax.Array, jax.Array]:
    """Solve one system, combining surrounding ``vmap`` axes natively."""
    if x0 is None:
        x0 = jnp.zeros_like(rhs)

    def solve_one(S, Phiinv, rhs, x0):
        return _pcg_ffi_solve_impl(S, Phiinv, rhs, x0, eps=eps, max_iters=max_iters)

    solve_one = jax.custom_batching.custom_vmap(solve_one)

    @solve_one.def_vmap
    def solve_one_vmap(axis_size, in_batched, S, Phiinv, rhs, x0):
        args = (S, Phiinv, rhs, x0)
        materialized = tuple(
            value if is_batched else jnp.broadcast_to(value, (axis_size,) + value.shape)
            for value, is_batched in zip(args, in_batched)
        )
        S_batched, Phiinv_batched, rhs_batched, x0_batched = materialized
        batch_shape = S_batched.shape[:-3]
        S_flat = S_batched.reshape((-1,) + S_batched.shape[-3:])
        Phiinv_flat = Phiinv_batched.reshape((-1,) + Phiinv_batched.shape[-3:])
        rhs_flat = rhs_batched.reshape((-1,) + rhs_batched.shape[-2:])
        x0_flat = x0_batched.reshape((-1,) + x0_batched.shape[-2:])
        solution_flat, iterations_flat = solve_one(
            S_flat, Phiinv_flat, rhs_flat, x0_flat
        )
        solution = solution_flat.reshape(batch_shape + solution_flat.shape[-2:])
        iterations = iterations_flat.reshape(batch_shape)
        return (solution, iterations), (True, True)

    return solve_one(S, Phiinv, rhs, x0)
