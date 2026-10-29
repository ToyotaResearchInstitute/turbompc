"""Allocation accounting and lifetime regressions for XLA-owned native buffers."""

import gc
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="CUDA device required",
)


@pytest.fixture(scope="module", autouse=True)
def _require_libraries():
    from turbompc.solvers.ffi_loader import find_ffi_lib

    for name in (
        "pcg_blktridi",
        "cudss_blktridi",
        "admm_cudss",
        "cudss_sparse_kkt",
    ):
        try:
            find_ffi_lib(f"lib{name}_ffi.so")
        except FileNotFoundError:
            pytest.skip(f"{name} FFI not built")


def _bytes_in_use():
    stats = jax.local_devices(backend="gpu")[0].memory_stats()
    if not stats or "bytes_in_use" not in stats:
        pytest.skip("JAX allocator accounting unavailable")
    return stats["bytes_in_use"]


def _wait_for_bytes_in_use_at_most(limit, timeout=1.0):
    """Wait for deferred PJRT buffer destruction to reach ``limit``."""
    gc.collect()
    deadline = time.monotonic() + timeout
    while True:
        current = _bytes_in_use()
        if current <= limit:
            return
        if time.monotonic() >= deadline:
            pytest.fail(
                f"device memory was not released: {current} bytes in use, "
                f"expected at most {limit}"
            )
        time.sleep(0.001)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("backend", ["block", "admm", "kkt"])
def test_workspace_allocations_are_accounted_and_released(backend, dtype):
    from turbompc.solvers.admm.admm_cudss_ffi_backend import AdmmCudssWorkspace
    from turbompc.solvers.backward.backward_kkt_cudss_ffi import CudssSparseKktWorkspace
    from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
        CudssBlockTridiWorkspace,
    )

    def create():
        if backend == "block":
            workspace = CudssBlockTridiWorkspace()
            workspace.ensure_capacity(1, 512, 8, dtype)
        elif backend == "admm":
            workspace = AdmmCudssWorkspace()
            # Include zero-sized constraint buffers.
            workspace.ensure_capacity(1, 512, 8, 4, 0, 0, dtype)
        else:
            pattern = SimpleNamespace(
                N_kkt=4096,
                nnz=4096,
                rowPtr=np.arange(4097, dtype=np.int32),
                colIdx=np.arange(4096, dtype=np.int32),
            )
            workspace = CudssSparseKktWorkspace(pattern, dtype)
        return workspace

    # Warm the lifecycle executable before measuring native allocation deltas.
    warm = create()
    warm.close()
    before = _bytes_in_use()
    workspace = create()
    allocated = _bytes_in_use()
    try:
        assert allocated - before >= 4096 * np.dtype(dtype).itemsize
    finally:
        workspace.close()
    workspace.close()  # idempotent
    _wait_for_bytes_in_use_at_most(before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_large_pcg_scratch_is_released(dtype):
    from turbompc.solvers.linear_systems_solvers.pcg_ffi_backend import pcg_ffi_solve

    # total > 1024 selects the multi-kernel path that allocates five buffers.
    T, n = 1025, 2
    blocks = np.zeros((T, n, 3 * n), dtype=dtype)
    blocks[:, :, n : 2 * n] = 2 * np.eye(n, dtype=dtype)
    S = jnp.asarray(blocks)
    preconditioner = S / 4
    rhs = jnp.ones((T, n), dtype=dtype)
    x0 = jnp.zeros_like(rhs)
    solve = jax.jit(
        lambda rhs: pcg_ffi_solve(S, preconditioner, rhs, x0, eps=1e-6, max_iters=5)
    )
    jax.block_until_ready((S, preconditioner, rhs, x0))
    before = _bytes_in_use()

    # Check numerical correctness once, then wait for the device-to-host copy's
    # deferred result-buffer reference to be released.
    out = solve(rhs)
    jax.block_until_ready(out)
    np.testing.assert_allclose(out[0], 0.5)
    del out
    _wait_for_bytes_in_use_at_most(before)

    # Exercise scratch lifetime without creating more device-to-host copies.
    for _ in range(5):
        out = solve(rhs)
        jax.block_until_ready(out)
        del out
    _wait_for_bytes_in_use_at_most(before)


def test_cudss_factorization_storage_is_released():
    from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
        CudssBlockTridiWorkspace,
    )

    T, n = 64, 4
    blocks = np.zeros((T, n, 3 * n), dtype=np.float64)
    blocks[:, :, n : 2 * n] = 2 * np.eye(n)
    S, rhs = jnp.asarray(blocks), jnp.ones((T, n), dtype=np.float64)
    jax.block_until_ready((S, rhs))
    before = _bytes_in_use()
    workspace = CudssBlockTridiWorkspace()
    workspace.ensure_capacity(1, T, n, np.float64)
    buffers_only = _bytes_in_use()
    try:
        out = workspace.solve(S, rhs)
        jax.block_until_ready(out)
        # cuDSS's internal analysis/factorization allocations also use XLA.
        assert _bytes_in_use() > buffers_only + out.nbytes
        np.testing.assert_allclose(out, 0.5)
        del out
    finally:
        workspace.close()
    _wait_for_bytes_in_use_at_most(before)
