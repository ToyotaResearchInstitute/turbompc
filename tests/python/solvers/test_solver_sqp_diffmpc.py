"""Tests for the SQP solver."""
import jax.numpy as jnp
import numpy as np
import pytest
from tests.helpers.problem_fixtures import tile_spacecraft_inertia
from tests.helpers.solver_fixtures import sqp_params
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.sqp_diffmpc import SolverReturnStatus, SQPDiffMPCSolver
from turbompc.utils.load_params import load_problem_params


def generate_problem_data(num_batch, seed):
    np.random.seed(seed)
    min_inertia = 1.0
    max_inertia = 10.0
    inertia_vector = min_inertia + np.random.rand(3) * (max_inertia - min_inertia)
    min_state = -0.1
    max_state = 0.1
    initial_states = min_state + np.random.rand(num_batch, 3) * (max_state - min_state)
    return jnp.array(inertia_vector), jnp.array(initial_states)


def sqp_solve_spacecraft(num_batch, horizon, use_linesearch, warm_start_backward):
    problem_params = load_problem_params("spacecraft.yaml")
    inertia_vector, initial_states = generate_problem_data(num_batch, seed=0)
    turbompc_horizon = horizon - 1  # turbompc horizon is N+1
    problem_params["horizon"] = turbompc_horizon
    problem_params["dynamics_state_dot_params"]["inertia_vector"] = inertia_vector
    tile_spacecraft_inertia(problem_params, horizon=turbompc_horizon)
    problem_params["reference_state_trajectory"] = jnp.zeros((turbompc_horizon + 1, 3))
    problem_params["reference_control_trajectory"] = jnp.zeros(
        (turbompc_horizon + 1, 3)
    )
    problem_params["weights_penalization_final_state"] = jnp.zeros(3)

    solver_params = sqp_params(
        tol=1.0e-6,
        sqp_iters=20,
        pcg_eps=1.0e-12,
        linesearch=use_linesearch,
        warm_start_backward=warm_start_backward,
        linesearch_alphas=[1.0],
    )

    dynamics = SpacecraftDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    solver = SQPDiffMPCSolver(program=problem, params=solver_params)

    sqp_guess = solver.initial_guess()

    def solver_initial_guess(initial_state):
        params = {**problem_params, "initial_state": initial_state}
        return solver.initial_guess(params)

    weights = {
        k: problem_params[k]
        for k in [
            "weights_penalization_reference_state_trajectory",
            "weights_penalization_control_squared",
        ]
    }
    params = dict(problem_params)

    solution = solver.solve(solver_initial_guess(initial_states[0]), params, weights)
    return solver, sqp_guess, solution, dynamics


@pytest.mark.parametrize("num_batch", [1, 2])
@pytest.mark.parametrize("horizon", [10, 20])
@pytest.mark.parametrize("use_linesearch", [True, False])
@pytest.mark.parametrize("warm_start_backward", [True, False])
def test_sqp_solve(num_batch, horizon, use_linesearch, warm_start_backward):
    solver, sqp_guess, solution, dynamics = sqp_solve_spacecraft(
        num_batch, horizon, use_linesearch, warm_start_backward
    )

    assert solver.name == "SQPDiffMPCSolver"
    assert solver.pcg is not None
    assert sqp_guess.states.shape == (horizon, dynamics.num_states)
    assert sqp_guess.controls.shape == (horizon, dynamics.num_controls)
    assert sqp_guess.dual.shape == (horizon, dynamics.num_states)
    assert solution.status == SolverReturnStatus.SUCCESS
    assert jnp.isfinite(solution.convergence_error)


def test_sqp_solve_warmstart():
    num_batch, horizon, use_linesearch = 1, 20, True
    num_iters = {}
    for warm_start_backward in [True, False]:
        _, _, solution, _ = sqp_solve_spacecraft(
            num_batch, horizon, use_linesearch, warm_start_backward
        )
        assert solution.status == SolverReturnStatus.SUCCESS
        assert jnp.isfinite(solution.convergence_error)
        if warm_start_backward:
            num_iters["warm"] = solution.num_iter
        else:
            num_iters["cold"] = solution.num_iter
    assert num_iters["warm"] <= num_iters["cold"]
