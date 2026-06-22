"""Tests of the optimal control features."""

import jax.numpy as jnp
import pytest
from tests.helpers.problem_fixtures import tile_spacecraft_inertia
from tests.helpers.solver_fixtures import sqp_params
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.sqp_diffmpc import SolverReturnStatus, SQPDiffMPCSolver
from turbompc.utils.load_params import load_problem_params


@pytest.fixture
def solver_parameters():
    """Returns the solver parameters."""
    return sqp_params(pcg_eps=1e-15)


@pytest.fixture
def spacecraft_problem_params():
    """Returns optimal control problem parameters for the spacecraft problem."""
    problem_params = load_problem_params("spacecraft.yaml")
    tile_spacecraft_inertia(problem_params)
    return problem_params


def test_spacecraft_with_sqp_solver(spacecraft_problem_params, solver_parameters):
    """
    Test solving optimal control problem for
    the spacecraft system using the SQP solver.
    """
    dynamics = SpacecraftDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=spacecraft_problem_params)
    solver = SQPDiffMPCSolver(program=problem, params=solver_parameters)
    sol = solver.solve(solver.initial_guess(), problem.params)
    convergence_error, status = sol.convergence_error, sol.status
    assert convergence_error < 2e-5
    assert SolverReturnStatus(status) is SolverReturnStatus.SUCCESS


def test_legacy_final_state_alias_sets_initial_guess_target(spacecraft_problem_params):
    """Legacy configs with ``final_state`` still warm-start toward that target."""
    target = jnp.array([0.25, -0.1, 0.05])
    legacy_params = dict(spacecraft_problem_params)
    legacy_params.pop("initial_guess_final_state")
    legacy_params["final_state"] = target

    problem = OptimalControlProblem(
        dynamics=SpacecraftDynamics(),
        params=legacy_params,
    )
    states, _ = problem.initial_guess()

    assert "final_state" not in problem.params
    assert jnp.allclose(problem.params["initial_guess_final_state"], target)
    assert jnp.allclose(states[-1], target + 1e-6)
    assert jnp.allclose(
        problem.params["reference_state_trajectory"][-1],
        spacecraft_problem_params["reference_state_trajectory"][-1],
    )
