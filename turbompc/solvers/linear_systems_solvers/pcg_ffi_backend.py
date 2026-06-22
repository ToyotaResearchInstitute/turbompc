"""PCG solver for block-tridiagonal systems via CUDA FFI."""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def _find_lib() -> Path:
    """Locate libpcg_blktridi_ffi.so relative to this file or in build/."""
    turbompc_dir = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        turbompc_dir / "solvers" / "csrc" / "build" / "libpcg_blktridi_ffi.so",
        repo_root / "build" / "ffi" / "libpcg_blktridi_ffi.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find libpcg_blktridi_ffi.so. Build first:\n"
        "  cmake -S turbompc/solvers/csrc -B build/ffi -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build build/ffi -j\n"
    )


_LIB = ctypes.cdll.LoadLibrary(str(_find_lib()))

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


def pcg_ffi_solve(
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
