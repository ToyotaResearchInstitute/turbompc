"""Tests for the SQP solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit
from tests.helpers.backend_utils import backend_available
from tests.helpers.problem_fixtures import tile_spacecraft_inertia
from tests.helpers.solver_fixtures import sqp_params, turbompc_solver_params
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.optimal_control_problem import (
    OptimalControlProblem,
    OptimalControlProblemSlack,
)
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.sqp_diffmpc import SQPDiffMPCSolver
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
    test_jittable: bool = False,
):
    dynamics = SpacecraftDynamics()
    if use_slack_variables:
        problem = OptimalControlProblemSlack(dynamics=dynamics, params=problem_params)
    else:
        problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    solver = TurboMPCSolver(
        program=problem,
        params=solver_params,
        forward_backend=forward_backend,
        backward_backend=backward_backend,
    )

    weights = {
        k: problem_params[k]
        for k in [
            "weights_penalization_reference_state_trajectory",
            "weights_penalization_control_squared",
        ]
    }
    if use_slack_variables:
        weights["slack_penalization_weight"] = problem_params[
            "slack_penalization_weight"
        ]

    def objective(weights):
        solution = solver.solve(
            solver.initial_guess(problem_params),
            problem_params=problem_params,
            weights=weights,
        )
        total = jnp.linalg.norm(solution.states) + jnp.linalg.norm(solution.controls)
        if use_slack_variables:
            total = total + jnp.sum(solution.slack**2)
        return total

    grad_obj_fn = jax.grad(objective)
    if test_jittable:
        grad_obj_fn = jit(grad_obj_fn)
    grad = grad_obj_fn(weights)
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
            ForwardBackend.ADMM_JAX_LOOP_PCG,
            BackwardBackend.ADMM_JAX_LOOP_PCG,
            id="admm_jax_loop_pcg",
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
@pytest.mark.parametrize(
    "with_box_constraints, use_slack_variables, horizon, tol, test_jittable",
    [
        (False, False, 10, 1e-12, False),
        (True, False, 8, 1e-10, False),
        (True, True, 6, 1e-10, False),
        (False, False, 10, 1e-12, True),
        (True, False, 8, 1e-10, True),
    ],
)
def test_turbompc_solver_solve_differentiable(
    forward_backend,
    backward_backend,
    with_box_constraints,
    use_slack_variables,
    horizon,
    tol,
    test_jittable,
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

    if with_box_constraints:
        problem_params["state_min_bounds"] = jnp.array([-0.2, -0.2, -0.2])
        problem_params["state_max_bounds"] = jnp.array([0.2, 0.2, 0.2])
        problem_params["control_min_bounds"] = jnp.array([-0.3, -0.3, -0.3])
        problem_params["control_max_bounds"] = jnp.array([0.3, 0.3, 0.3])
    if use_slack_variables:
        problem_params["use_slack_variables"] = True
        problem_params["slack_penalization_weight"] = jnp.array(1.0)

    solver_params = turbompc_solver_params(tol=tol, sqp_iters=10, admm_max=500)
    solver_params["linesearch"] = False
    solver_params["warm_start_backward"] = True

    eps = (
        1e-8
        if with_box_constraints
        else (
            1e-10
            if forward_backend == ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI
            else 1e-12
        )
    )
    _run_turbompc_solver_diff_test(
        problem_params,
        solver_params,
        eps,
        use_slack_variables,
        forward_backend=forward_backend,
        backward_backend=backward_backend,
        test_jittable=test_jittable,
    )
