"""Tests for the SQP solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit

from tests.helpers.backend_utils import backend_available
from tests.helpers.constraint_handling_fixtures import HANDLINGS, with_handling
from tests.helpers.problem_fixtures import tile_spacecraft_inertia
from tests.helpers.solver_fixtures import sqp_params, turbompc_solver_params
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.sqp_diffmpc import SQPDiffMPCSolver
from turbompc.solvers.status import SQPStatus
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)
from turbompc.utils.gradient_finitediff import gradient_finite_diff
from turbompc.utils.load_params import load_problem_params


def test_sqp_solve_differentiable():
    np.random.seed(0)
    horizon = 10

    problem_params = load_problem_params("spacecraft.yaml")
    turbompc_horizon = horizon - 1  # turbompc horizon is N+1
    min_state = -0.1
    max_state = 0.1
    problem_params["initial_state"] = min_state + np.random.rand(3) * (
        max_state - min_state
    )
    problem_params["horizon"] = turbompc_horizon
    tile_spacecraft_inertia(problem_params, horizon=turbompc_horizon)
    problem_params["reference_state_trajectory"] = jnp.zeros((turbompc_horizon + 1, 3))
    problem_params["reference_control_trajectory"] = jnp.zeros(
        (turbompc_horizon + 1, 3)
    )
    problem_params["weights_penalization_final_state"] = jnp.zeros(3)

    solver_params = sqp_params(
        tol=1.0e-12,
        sqp_iters=30,
        pcg_eps=1.0e-24,
        linesearch=False,
        warm_start_backward=True,
    )

    dynamics = SpacecraftDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    solver = SQPDiffMPCSolver(program=problem, params=solver_params)

    weights = {
        k: problem_params[k]
        for k in [
            "weights_penalization_reference_state_trajectory",
            "weights_penalization_control_squared",
        ]
    }

    def objective(weights):
        solution = solver.solve(
            solver.initial_guess(problem_params),
            problem_params=problem_params,
            weights=weights,
        )
        return jnp.linalg.norm(solution.states) + jnp.linalg.norm(solution.controls)

    def auto_grad(weights):
        grad = jax.grad(objective)(weights)
        return grad

    grad = auto_grad(weights)
    grad_fd = gradient_finite_diff(objective, weights=weights, eps=1e-12)

    for key in weights.keys():
        diff = jnp.array(grad_fd[key]) - grad[key]
        max_diff = float(jnp.max(jnp.abs(diff)))
        assert jnp.allclose(
            diff, 0.0, atol=1e-3, rtol=1e-3
        ), f"{key}: max finite-difference gradient mismatch={max_diff:.3e}"


def _run_turbompc_solver_diff_test(
    problem_params,
    solver_params,
    eps,
    use_slack_variables,
    forward_backend: ForwardBackend,
    backward_backend: BackwardBackend,
):
    dynamics = SpacecraftDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    solver = TurboMPCSolver(
        program=problem,
        params=solver_params,
        forward_backend=forward_backend,
        backward_backend=backward_backend,
    )
    guess = solver.initial_guess(problem_params)
    # Gradients of an unconverged solve are not the implicit-function gradients.
    assert int(solver.solve(guess, problem_params).status) == int(SQPStatus.SOLVED)

    weights = {
        k: problem_params[k]
        for k in [
            "weights_penalization_reference_state_trajectory",
            "weights_penalization_control_squared",
        ]
    }
    if use_slack_variables:
        weights["slack_penalization_weight"] = jnp.asarray(
            problem_params["slack_penalization_weight"]
        )

    # Jitted once and reused for every finite-difference evaluation; eager
    # solves cost ~20x more. This also covers differentiating under jit.
    @jit
    def objective(weights):
        solution = solver.solve(guess, problem_params=problem_params, weights=weights)
        total = jnp.linalg.norm(solution.states) + jnp.linalg.norm(solution.controls)
        if use_slack_variables:
            total = total + jnp.sum(solution.slack**2)
        return total

    grad = jit(jax.grad(objective))(weights)
    grad_fd = gradient_finite_diff(objective, weights=weights, eps=eps)

    for key in weights.keys():
        diff = jnp.array(grad_fd[key]) - grad[key]
        max_diff = float(jnp.max(jnp.abs(diff)))
        assert jnp.allclose(
            diff, 0.0, atol=1e-3, rtol=1e-3
        ), f"{key}: max finite-difference gradient mismatch={max_diff:.3e}"
        assert not jnp.isnan(grad[key]).any()


@pytest.mark.parametrize(
    "forward_backend,backward_backend",
    [
        pytest.param(
            ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
            BackwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
            id="admm_jax_loop_jax_dense",
        ),
        pytest.param(
            ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
            BackwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
            marks=pytest.mark.skipif(
                not backend_available(SchurSolverBackend.CUDSS_FFI),
                reason="cuDSS FFI not built",
            ),
            id="admm_jax_loop_cudss_ffi",
        ),
    ],
)
@pytest.mark.parametrize(("constraint_handling", "log_barrier_kappa"), HANDLINGS)
@pytest.mark.parametrize(
    "with_box_constraints, use_slack_variables, horizon",
    [(False, False, 10), (True, False, 8), (True, True, 6)],
)
def test_turbompc_solver_solve_differentiable(
    forward_backend,
    backward_backend,
    constraint_handling,
    log_barrier_kappa,
    with_box_constraints,
    use_slack_variables,
    horizon,
):
    """Test SQP-ADMM gradients with and without box constraints."""
    np.random.seed(0)

    problem_params = load_problem_params("spacecraft.yaml")
    turbompc_horizon = horizon - 1
    min_state = -0.1
    max_state = 0.1
    problem_params["initial_state"] = min_state + np.random.rand(3) * (
        max_state - min_state
    )
    problem_params["horizon"] = turbompc_horizon
    tile_spacecraft_inertia(problem_params, horizon=turbompc_horizon)
    problem_params["reference_state_trajectory"] = jnp.zeros((turbompc_horizon + 1, 3))
    problem_params["reference_control_trajectory"] = jnp.zeros(
        (turbompc_horizon + 1, 3)
    )
    problem_params["weights_penalization_final_state"] = jnp.zeros(3)
    problem_params["use_slack_variables"] = use_slack_variables
    problem_params["slack_penalization_weight"] = 1.0 if use_slack_variables else 0.0

    if with_box_constraints:
        problem_params["state_min_bounds"] = jnp.array([-0.2, -0.2, -0.2])
        problem_params["state_max_bounds"] = jnp.array([0.2, 0.2, 0.2])
        problem_params["control_min_bounds"] = jnp.array([-0.3, -0.3, -0.3])
        problem_params["control_max_bounds"] = jnp.array([0.3, 0.3, 0.3])
    solver_params = with_handling(
        turbompc_solver_params(tol=1e-10, sqp_iters=10, admm_max=500), log_barrier_kappa
    )
    solver_params["linesearch"]["method"] = "none"
    solver_params["warm_start_backward"] = True

    _run_turbompc_solver_diff_test(
        problem_params,
        solver_params,
        1e-5,
        use_slack_variables,
        forward_backend=forward_backend,
        backward_backend=backward_backend,
    )
