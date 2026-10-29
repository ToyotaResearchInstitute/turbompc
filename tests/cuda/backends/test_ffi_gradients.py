"""Tests for FFI backend gradient correctness."""
import jax
import jax.numpy as jnp
import pytest

from tests.helpers.backend_utils import backend_available
from tests.helpers.constraint_handling_fixtures import HANDLINGS, with_handling
from tests.helpers.problem_fixtures import make_spacecraft_params
from tests.helpers.solver_fixtures import turbompc_solver_params
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)


def _make_gradient_setup(log_barrier_kappa):
    horizon = 10
    params = make_spacecraft_params(horizon=horizon)
    dynamics = SpacecraftDynamics()
    ocp = OptimalControlProblem(dynamics=dynamics, params=params)
    sp = with_handling(turbompc_solver_params(tol=1e-8), log_barrier_kappa)
    weights = {
        "weights_penalization_reference_state_trajectory": params[
            "weights_penalization_reference_state_trajectory"
        ],
        "weights_penalization_control_squared": params[
            "weights_penalization_control_squared"
        ],
    }
    return ocp, sp, params, weights


def _make_objective(solver, params):
    def objective(w):
        sol = solver.solve(
            solver.initial_guess(params), problem_params=params, weights=w
        )
        return jnp.linalg.norm(sol.states) + jnp.linalg.norm(sol.controls)

    return objective


@pytest.mark.skipif(
    not backend_available(SchurSolverBackend.PCG_FFI), reason="PCG FFI not built"
)
@pytest.mark.parametrize(("constraint_handling", "log_barrier_kappa"), HANDLINGS)
def test_pcg_ffi_backward_gradient(constraint_handling, log_barrier_kappa):
    """PCG FFI gradients match pure JAX PCG gradients."""
    ocp, sp, params, weights = _make_gradient_setup(log_barrier_kappa)

    solver_ref = TurboMPCSolver(
        ocp,
        params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
    )
    solver_ffi = TurboMPCSolver(
        ocp,
        params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG_FFI,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG_FFI,
    )

    grad_ref = jax.grad(_make_objective(solver_ref, params))(weights)
    grad_ffi = jax.grad(_make_objective(solver_ffi, params))(weights)

    for key in weights:
        diff = float(jnp.max(jnp.abs(grad_ref[key] - grad_ffi[key])))
        assert diff < 1e-5, f"Gradient mismatch for {key}: {diff}"


@pytest.mark.skipif(
    not backend_available(SchurSolverBackend.CUDSS_FFI), reason="cuDSS FFI not built"
)
@pytest.mark.parametrize(("constraint_handling", "log_barrier_kappa"), HANDLINGS)
def test_cudss_ffi_backward_gradient(constraint_handling, log_barrier_kappa):
    """cuDSS FFI gradients match pure JAX PCG gradients."""
    ocp, sp, params, weights = _make_gradient_setup(log_barrier_kappa)

    solver_ref = TurboMPCSolver(
        ocp,
        params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
    )
    solver_ffi = TurboMPCSolver(
        ocp,
        params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
    )

    grad_ref = jax.grad(_make_objective(solver_ref, params))(weights)
    grad_ffi = jax.grad(_make_objective(solver_ffi, params))(weights)

    for key in weights:
        diff = float(jnp.max(jnp.abs(grad_ref[key] - grad_ffi[key])))
        assert diff < 1e-5, f"Gradient mismatch for {key}: {diff}"
