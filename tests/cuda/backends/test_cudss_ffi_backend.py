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
    @pytest.mark.parametrize("horizon", [5, 25])
    def test_cudss_ffi_vs_scipy(self, horizon):
        """cuDSS FFI solution matches a direct `jnp.linalg.solve` reference."""
        from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
            cudss_ffi_solve,
        )

        schur, gamma = make_spacecraft_schur_system(horizon)
        S = schur.S

        S_np = np.array(S)
        gamma_np = np.array(gamma)
        ref = solve_block_tridi_system(
            S_np, gamma_np, backend=SchurSolverBackend.JAX_DENSE
        )
        ref = jnp.array(ref)

        sol = cudss_ffi_solve(S, gamma)

        diff = float(jnp.max(jnp.abs(sol - ref)))
        assert (
            diff < 5e-6
        ), f"horizon={horizon}: cuDSS FFI vs JAX dense mismatch: {diff}"

    @pytest.mark.parametrize("horizon", [5, 10])
    def test_cudss_ffi_is_deterministic(self, horizon):
        """Repeated solves with same inputs give the same result."""
        from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
            cudss_ffi_solve,
        )

        schur, gamma = make_spacecraft_schur_system(horizon)
        S = schur.S

        sol1 = cudss_ffi_solve(S, gamma)
        sol2 = cudss_ffi_solve(S, gamma)

        assert jnp.allclose(sol1, sol2)

    def test_cudss_ffi_jit(self):
        """cuDSS FFI is JIT-compilable."""
        from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
            cudss_ffi_solve,
        )

        schur, gamma = make_spacecraft_schur_system(horizon=10)
        S = schur.S

        solve_jit = jax.jit(lambda s, g: cudss_ffi_solve(s, g))
        sol1 = solve_jit(S, gamma)
        jax.block_until_ready(sol1)

        sol2 = solve_jit(S, gamma)
        jax.block_until_ready(sol2)
        assert jnp.allclose(sol1, sol2)

    def test_cudss_ffi_plan_reuse(self):
        """Repeated calls with same shape reuse the cuDSS plan."""
        from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
            cudss_ffi_solve,
        )

        schur, gamma = make_spacecraft_schur_system(horizon=10)
        S = schur.S

        sol1 = cudss_ffi_solve(S, gamma)
        sol2 = cudss_ffi_solve(S, gamma * 2.0)

        ref1 = solve_block_tridi_system(
            np.array(S), np.array(gamma), SchurSolverBackend.JAX_DENSE
        )
        ref2 = solve_block_tridi_system(
            np.array(S), np.array(gamma * 2.0), SchurSolverBackend.JAX_DENSE
        )

        np.testing.assert_allclose(np.array(sol1), ref1, rtol=1e-5)
        np.testing.assert_allclose(np.array(sol2), ref2, rtol=1e-5)
