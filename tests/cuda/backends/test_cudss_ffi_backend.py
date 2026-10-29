"""Tests for the block-tridiagonal cuDSS FFI backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.backend_utils import backend_available
from tests.helpers.schur_fixtures import make_spacecraft_schur_system
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.linear_systems_solvers.linear_solve import (
    solve_block_tridi_system,
)


@pytest.mark.skipif(
    not backend_available(SchurSolverBackend.CUDSS_FFI), reason="cuDSS FFI not built"
)
class TestCudssFfiBackend:
    @pytest.mark.parametrize("batch_shape", [(1,), (2,), (2, 1)])
    def test_custom_vmap_reuses_one_lazy_workspace(self, batch_shape):
        from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
            CudssBlockTridiWorkspace,
        )

        schur, rhs = make_spacecraft_schur_system(horizon=5)
        workspace = CudssBlockTridiWorkspace()
        expected = workspace.solve(schur.S, rhs)
        jax.block_until_ready(expected)
        workspace_handle = workspace.handle

        S_batched = jnp.broadcast_to(schur.S, batch_shape + schur.S.shape)
        rhs_batched = jnp.broadcast_to(rhs, batch_shape + rhs.shape)
        solve_batched = workspace.solve
        for _ in batch_shape:
            solve_batched = jax.vmap(solve_batched)
        actual = jax.jit(solve_batched)(S_batched, rhs_batched)
        jax.block_until_ready(actual)

        np.testing.assert_allclose(
            actual,
            jnp.broadcast_to(expected, batch_shape + expected.shape),
            rtol=1e-5,
        )
        scalar_again = workspace.solve(schur.S, rhs)
        jax.block_until_ready(scalar_again)
        assert workspace.handle == workspace_handle
        np.testing.assert_allclose(scalar_again, expected, rtol=1e-5)

    @pytest.mark.parametrize("horizon", [5, 25])
    def test_cudss_ffi_vs_scipy(self, horizon):
        """cuDSS FFI solution matches a direct `jnp.linalg.solve` reference."""
        from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
            CudssBlockTridiWorkspace,
        )

        schur, gamma = make_spacecraft_schur_system(horizon)
        S = schur.S

        S_np = np.array(S)
        gamma_np = np.array(gamma)
        ref = solve_block_tridi_system(
            S_np, gamma_np, backend=SchurSolverBackend.JAX_DENSE
        )
        ref = jnp.array(ref)

        workspace = CudssBlockTridiWorkspace()
        sol = workspace.solve(S, gamma)

        diff = float(jnp.max(jnp.abs(sol - ref)))
        assert (
            diff < 5e-6
        ), f"horizon={horizon}: cuDSS FFI vs JAX dense mismatch: {diff}"

    @pytest.mark.parametrize("horizon", [5, 10])
    def test_cudss_ffi_is_deterministic(self, horizon):
        """Repeated solves with same inputs give the same result."""
        from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
            CudssBlockTridiWorkspace,
        )

        schur, gamma = make_spacecraft_schur_system(horizon)
        S = schur.S

        workspace = CudssBlockTridiWorkspace()
        sol1 = workspace.solve(S, gamma)
        sol2 = workspace.solve(S, gamma, factorize=False)

        assert jnp.allclose(sol1, sol2)

    def test_cudss_ffi_jit(self):
        """cuDSS FFI is JIT-compilable."""
        from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
            CudssBlockTridiWorkspace,
        )

        schur, gamma = make_spacecraft_schur_system(horizon=10)
        S = schur.S

        workspace = CudssBlockTridiWorkspace()
        solve_jit = jax.jit(
            lambda blocks, rhs, factorize: workspace.solve(
                blocks, rhs, factorize=factorize
            )
        )
        sol1 = solve_jit(S, gamma, True)
        jax.block_until_ready(sol1)

        sol2 = solve_jit(S, gamma, False)
        jax.block_until_ready(sol2)
        assert jnp.allclose(sol1, sol2)

    def test_cudss_ffi_factorization_reuse(self):
        """Solve-only calls reuse the existing numerical factorization."""
        from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
            CudssBlockTridiWorkspace,
        )

        schur, gamma = make_spacecraft_schur_system(horizon=10)
        S = schur.S

        workspace = CudssBlockTridiWorkspace()
        sol1 = workspace.solve(S, gamma)
        sol2 = workspace.solve(S, gamma * 2.0, factorize=False)

        ref1 = solve_block_tridi_system(
            np.array(S), np.array(gamma), SchurSolverBackend.JAX_DENSE
        )
        ref2 = solve_block_tridi_system(
            np.array(S), np.array(gamma * 2.0), SchurSolverBackend.JAX_DENSE
        )

        np.testing.assert_allclose(np.array(sol1), ref1, rtol=1e-5)
        np.testing.assert_allclose(np.array(sol2), ref2, rtol=1e-5)

    def test_cudss_ffi_reuse_requires_factorization(self):
        """A fresh workspace rejects a solve-only request."""
        from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
            CudssBlockTridiWorkspace,
        )

        schur, gamma = make_spacecraft_schur_system(horizon=5)
        S = schur.S
        workspace = CudssBlockTridiWorkspace()

        with pytest.raises(Exception, match="has not been factorized"):
            jax.block_until_ready(workspace.solve(S, gamma, factorize=False))
