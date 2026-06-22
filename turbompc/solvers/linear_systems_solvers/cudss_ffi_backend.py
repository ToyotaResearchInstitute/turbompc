"""cuDSS solver for block-tridiagonal systems via CUDA FFI."""
from __future__ import annotations

import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp


def _find_lib() -> Path:
    turbompc_dir = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        turbompc_dir / "solvers" / "csrc" / "build" / "libcudss_blktridi_ffi.so",
        repo_root / "build" / "ffi" / "libcudss_blktridi_ffi.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find libcudss_blktridi_ffi.so. Build first:\n"
        "  cmake -S turbompc/solvers/csrc -B build/ffi -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build build/ffi -j\n"
    )


_LIB = ctypes.cdll.LoadLibrary(str(_find_lib()))

jax.ffi.register_ffi_target(
    "cudss_blktridi_cuda",
    jax.ffi.pycapsule(_LIB.CudssBlkTridiCuda),
    platform="CUDA",
)
jax.ffi.register_ffi_target(
    "cudss_blktridi_cuda_f64",
    jax.ffi.pycapsule(_LIB.CudssBlkTridiCudaF64),
    platform="CUDA",
)


def cudss_ffi_solve(
    S: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
    """Solve block-tridiagonal system S @ x = rhs via cuDSS FFI.

    Supports both unbatched and batched inputs:
        Unbatched: S (T,n,3n), rhs (T,n) -> x (T,n)
        Batched:   S (Nb,T,n,3n), rhs (Nb,T,n) -> x (Nb,T,n)
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
        ffi_target = "cudss_blktridi_cuda_f64"
    else:
        kernel_dtype = jnp.float32
        ffi_target = "cudss_blktridi_cuda"

    S_k = jnp.asarray(S, dtype=kernel_dtype)
    rhs_k = jnp.asarray(rhs, dtype=kernel_dtype)

    if batched:
        out_x = jax.ShapeDtypeStruct((Nb, T, n), kernel_dtype)
    else:
        out_x = jax.ShapeDtypeStruct((T, n), kernel_dtype)

    call = jax.ffi.ffi_call(
        ffi_target,
        out_x,
        vmap_method="broadcast_all",
    )

    x_out = call(S_k, rhs_k)
    return jnp.asarray(x_out, dtype=dtype)
