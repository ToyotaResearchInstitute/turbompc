"""Tests for the block-tridiagonal PCG FFI backend."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.helpers.backend_utils import backend_available
from tests.helpers.problem_fixtures import make_spacecraft_params
from tests.helpers.schur_fixtures import make_spacecraft_schur_system
from tests.helpers.solver_fixtures import turbompc_solver_params
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.linear_systems_solvers.linear_solve import (
    solve_block_tridi_system,
)
from turbompc.solvers.linear_systems_solvers.pcg_primal import PCGPrimalOptimalControl
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)


@pytest.mark.skipif(
    not backend_available(SchurSolverBackend.PCG_FFI), reason="PCG FFI not built"
)
class TestPcgFfiBackend:
    @pytest.mark.parametrize("horizon", [5, 20])
    def test_pcg_ffi_vs_scipy(self, horizon):
        """PCG FFI solution matches a direct `jnp.linalg.solve` reference."""
        from turbompc.solvers.linear_systems_solvers.pcg_ffi_backend import (
            pcg_ffi_solve,
        )

        schur, gamma = make_spacecraft_schur_system(horizon)
        S = schur.S
        Phiinv = schur.preconditioner_Phiinv

        S_np = np.array(S)
        gamma_np = np.array(gamma)
        ref = solve_block_tridi_system(
            S_np, gamma_np, backend=SchurSolverBackend.JAX_DENSE
        )
        ref = jnp.array(ref)

        T, n, _ = S.shape
        x0 = jnp.zeros((T, n), dtype=S.dtype)
        sol, _ = pcg_ffi_solve(S, Phiinv, gamma, x0, eps=1e-12, max_iters=500)

        diff = float(jnp.max(jnp.abs(sol - ref)))
        assert diff < 5e-6, f"horizon={horizon}: PCG FFI vs SciPy mismatch: {diff}"

    @pytest.mark.parametrize("horizon", [5, 10])
    def test_pcg_ffi_vs_jax_pcg(self, horizon):
        """PCG FFI matches the pure JAX PCG implementation."""
        from turbompc.solvers.linear_systems_solvers.pcg_ffi_backend import (
            pcg_ffi_solve,
        )

        schur, gamma = make_spacecraft_schur_system(horizon)
        S = schur.S
        Phiinv = schur.preconditioner_Phiinv
        T, n, _ = S.shape

        pcg = PCGPrimalOptimalControl(
            problem_horizon=horizon,
            problem_num_states=3,
            problem_num_controls=3,
            solver_params={"max_iter": 500, "tol_epsilon": 1e-12},
        )
        zs_guess = jnp.zeros((T, n))
        jax_sol, _ = pcg.solve_linear_system(schur, gamma, zs_guess)

        x0 = jnp.zeros((T, n), dtype=S.dtype)
        ffi_sol, _ = pcg_ffi_solve(S, Phiinv, gamma, x0, eps=1e-12, max_iters=500)

        diff = float(jnp.max(jnp.abs(ffi_sol - jax_sol)))
        assert diff < 1e-6, f"horizon={horizon}: PCG FFI vs JAX PCG mismatch: {diff}"

    def test_pcg_ffi_jit(self):
        """PCG FFI is JIT-compilable."""
        from turbompc.solvers.linear_systems_solvers.pcg_ffi_backend import (
            pcg_ffi_solve,
        )

        schur, gamma = make_spacecraft_schur_system(horizon=10)
        S = schur.S
        Phiinv = schur.preconditioner_Phiinv
        T, n, _ = S.shape
        x0 = jnp.zeros((T, n), dtype=S.dtype)

        solve_jit = jax.jit(lambda s, p, g, x: pcg_ffi_solve(s, p, g, x))
        sol1, _ = solve_jit(S, Phiinv, gamma, x0)
        jax.block_until_ready(sol1)

        sol2, _ = solve_jit(S, Phiinv, gamma, x0)
        jax.block_until_ready(sol2)
        assert jnp.allclose(sol1, sol2)

    def test_pcg_ffi_numerical_stability_ill_conditioned(self):
        """PCG FFI handles moderately ill-conditioned systems."""
        from turbompc.solvers.linear_systems_solvers.pcg_ffi_backend import (
            pcg_ffi_solve,
        )

        schur, gamma = make_spacecraft_schur_system(horizon=10)
        S = schur.S
        Phiinv = schur.preconditioner_Phiinv

        S_scaled = S * 1e4
        Phiinv_scaled = Phiinv * 1e-4
        gamma_scaled = gamma * 1e4

        T, n, _ = S_scaled.shape
        x0 = jnp.zeros((T, n), dtype=S.dtype)

        ref = solve_block_tridi_system(
            np.array(S_scaled),
            np.array(gamma_scaled),
            backend=SchurSolverBackend.JAX_DENSE,
        )
        ref = jnp.array(ref)

        sol, _ = pcg_ffi_solve(
            S_scaled, Phiinv_scaled, gamma_scaled, x0, eps=1e-12, max_iters=1000
        )

        rel_err = float(jnp.linalg.norm(sol - ref) / jnp.linalg.norm(ref))
        assert rel_err < 1e-4, f"Ill-conditioned relative error too large: {rel_err}"


@pytest.mark.skipif(
    not backend_available(SchurSolverBackend.PCG_FFI), reason="PCG FFI not built"
)
def test_pcg_ffi_in_admm_forward():
    """PCG FFI works end-to-end in ADMM forward solve."""
    horizon = 10
    params = make_spacecraft_params(horizon=horizon)
    ocp = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    sp = turbompc_solver_params(tol=1e-8)

    solver_pcg = TurboMPCSolver(
        ocp,
        params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
    )
    solver_ffi = TurboMPCSolver(
        ocp,
        params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG_FFI,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
    )

    sol_pcg = solver_pcg.solve(solver_pcg.initial_guess(params), problem_params=params)
    sol_ffi = solver_ffi.solve(solver_ffi.initial_guess(params), problem_params=params)

    diff_states = float(jnp.max(jnp.abs(sol_pcg.states - sol_ffi.states)))
    diff_controls = float(jnp.max(jnp.abs(sol_pcg.controls - sol_ffi.controls)))
    assert diff_states < 1e-5, f"States mismatch: {diff_states}"
    assert diff_controls < 1e-5, f"Controls mismatch: {diff_controls}"
