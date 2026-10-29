import gc

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.problem_fixtures import make_drone_params
from turbompc.dynamics.integrators import DiscretizationScheme
from turbompc.problems.obstacle_avoidance import OptimalControlProblemObstacle
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    HessianApproximation,
    TurboMPCSolver,
)
from turbompc.utils.load_params import load_solver_params

HORIZON = 5
START = jnp.array([-1.9, 0.05, 0.0, 0.0, 0.0, 0.0])
GOAL = jnp.zeros(6)
OBSTACLE_CENTER = jnp.array([-0.95, 0.025])
OBSTACLE_RADIUS = 0.3
CONVERGENCE_TOLERANCE = 4.0e-7
MERIT_CONVERGENCE_TOLERANCE = 1.0e-5


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches():
    jax.clear_caches()
    gc.collect()
    yield
    jax.clear_caches()
    gc.collect()


def _solve_drone(linesearch_method):
    obstacle_centers = jnp.tile(OBSTACLE_CENTER[None, None, :], (HORIZON + 1, 1, 1))
    problem_params, dynamics = make_drone_params(
        HORIZON,
        obstacle_centers,
        jnp.array([OBSTACLE_RADIUS]),
    )
    problem_params.update(
        {
            "discretization_resolution": 1.25,
            "discretization_scheme": int(DiscretizationScheme.IMPLICIT),
            "constrain_initial_control": False,
            "initial_state": START,
            "initial_guess_final_state": GOAL,
            "reference_state_trajectory": jnp.tile(GOAL, (HORIZON + 1, 1)),
            "reference_control_trajectory": jnp.zeros((HORIZON + 1, 3)),
            "weights_penalization_reference_state_trajectory": jnp.array(
                [5.0, 5.0, 5.0, 0.1, 0.1, 0.1]
            ),
            "weights_penalization_control_squared": jnp.array([0.1, 0.1, 0.1]),
            "weights_penalization_final_state": jnp.zeros(6),
            "rescale_optimization_variables": False,
        }
    )
    solver_params = dict(load_solver_params("turbompc.yaml"))
    solver_params["linesearch"]["method"] = linesearch_method
    solver_params.update(
        {
            "num_sqp_iteration_max": 15,
            "tol_convergence": CONVERGENCE_TOLERANCE,
        }
    )
    solver = TurboMPCSolver(
        program=OptimalControlProblemObstacle(dynamics, problem_params),
        params=solver_params,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE,
        backward_hessian=HessianApproximation.GAUSS_NEWTON,
    )
    solution = solver.solve(solver.initial_guess(problem_params), problem_params)
    solution.states.block_until_ready()
    return solution


def test_filter_reduces_work_on_nonlinear_drone_obstacle_problem():
    full_step = _solve_drone("none")
    merit = _solve_drone("merit")
    filtered = _solve_drone("filter")

    assert full_step.linesearch_alphas is None
    assert float(full_step.convergence_error) < CONVERGENCE_TOLERANCE
    assert float(merit.convergence_error) < MERIT_CONVERGENCE_TOLERANCE
    assert float(filtered.convergence_error) < CONVERGENCE_TOLERANCE
    assert float(filtered.convergence_error) < float(merit.convergence_error)
    assert int(filtered.num_iter) <= int(full_step.num_iter)
    assert int(filtered.num_iter) < int(merit.num_iter)
    assert int(filtered.admm_iters.sum()) < int(merit.admm_iters.sum())

    filtered_num_iter = int(filtered.num_iter)
    filtered_alphas = np.asarray(filtered.linesearch_alphas)[:filtered_num_iter]
    merit_alphas = np.asarray(merit.linesearch_alphas)[: int(merit.num_iter)]

    linesearch_params = load_solver_params("turbompc.yaml")["linesearch"]
    assert np.all(np.isin(filtered_alphas, linesearch_params["alphas"]))
    assert np.all(np.isin(merit_alphas, linesearch_params["alphas"]))

    positions = np.asarray(filtered.states)[:, :2]
    distances = np.linalg.norm(positions - np.asarray(OBSTACLE_CENTER), axis=1)
    assert distances.min() >= OBSTACLE_RADIUS - 1.0e-8
    assert distances.min() < OBSTACLE_RADIUS + 0.01

    speeds = np.linalg.norm(np.asarray(filtered.states)[:, 3:6], axis=1)
    assert speeds.max() > 0.1

    np.testing.assert_allclose(filtered.states, merit.states, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(
        filtered.controls, merit.controls, rtol=1.0e-5, atol=1.0e-5
    )
