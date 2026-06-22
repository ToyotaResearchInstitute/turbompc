import copy
from time import time

import jax.numpy as jnp
import numpy as np
import pytest
from tests.helpers.assertions import (
    assert_box_bounds,
    assert_equality_residual,
    assert_slack_active,
    assert_solution_nontrivial,
)
from tests.helpers.problem_fixtures import (
    make_drone_params,
    make_linear_params,
    make_spacecraft_params,
    tile_spacecraft_inertia,
)
from tests.helpers.solver_fixtures import sqp_osqp_params, sqp_params
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.obstacle_avoidance import OptimalControlProblemObstacle
from turbompc.problems.optimal_control_problem import (
    OptimalControlProblem,
    OptimalControlProblemSlack,
)
from turbompc.solvers.qp_utils import pack_z
from turbompc.solvers.sqp_diffmpc import SolverReturnStatus, SQPDiffMPCSolver
from turbompc.solvers.sqp_osqp import SQPOSQPSolver
from turbompc.utils.load_params import load_problem_params

EQ_TOL = 1.0e-3
Z_TOL = 1.0e-4
COST_TOL = 1.0e-4


@pytest.mark.parametrize("horizon", [2, 3])
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_sqp_osqp_linear_smoke(horizon, implicit, bounded):
    dynamics, params = make_linear_params(
        horizon, implicit, bounded, rescale=False, rescaling="unit"
    )
    problem = OptimalControlProblem(dynamics=dynamics, params=copy.deepcopy(params))
    solver = SQPOSQPSolver(program=problem)

    sol = solver.solve(problem_params=params)

    assert_solution_nontrivial(sol)
    assert sol.states.shape == (horizon + 1, dynamics.num_states)
    assert sol.controls.shape == (horizon + 1, dynamics.num_controls)
    assert_equality_residual(problem, sol, params, tol=EQ_TOL)


def test_sqp_osqp_respects_box_bounds_spacecraft():
    params = make_spacecraft_params(horizon=6)
    problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    solver = SQPOSQPSolver(problem)

    sol = solver.solve(problem_params=params)

    assert sol.status == 0

    assert_box_bounds(problem, sol, params, tol=1e-6)
    assert_equality_residual(problem, sol, params, tol=1e-3)


def test_sqp_osqp_control_rate_penalty_smooths_controls():
    dynamics, base_params = make_linear_params(
        6,
        False,
        False,
        rescale=False,
        rescaling="unit",
        initial_state=jnp.zeros((4,)),
    )
    nx = dynamics.num_states
    nu = dynamics.num_controls

    N = base_params["horizon"]
    reference_controls = jnp.stack(
        [jnp.array([(-1.0) ** t, (-1.0) ** (t + 1)]) for t in range(N)]
        + [jnp.array([(-1.0) ** (N - 1), (-1.0) ** N])]
    )

    base_params = dict(base_params)
    base_params["reference_control_trajectory"] = reference_controls
    base_params["penalize_control_reference"] = True
    base_params["weights_penalization_reference_state_trajectory"] = jnp.zeros((nx,))
    base_params["weights_penalization_control_rate"] = jnp.zeros((nu,))

    params_no_rd = dict(base_params)
    params_rd = dict(base_params)
    params_rd["weights_penalization_control_rate"] = jnp.ones((nu,)) * 10.0

    problem = OptimalControlProblem(dynamics=dynamics, params=base_params)

    solver_no_rd = SQPOSQPSolver(program=problem)
    sol_no_rd = solver_no_rd.solve(problem_params=params_no_rd)

    solver_rd = SQPOSQPSolver(program=problem)
    sol_rd = solver_rd.solve(problem_params=params_rd)

    du_no_rd = sol_no_rd.controls[1:] - sol_no_rd.controls[:-1]
    du_rd = sol_rd.controls[1:] - sol_rd.controls[:-1]

    smoothness_no_rd = jnp.sum(du_no_rd**2)
    smoothness_rd = jnp.sum(du_rd**2)

    assert smoothness_rd < smoothness_no_rd


def test_sqp_osqp_respects_initial_control_constraint():
    init_control = jnp.array([0.3, -0.4])
    dynamics, params = make_linear_params(
        5,
        False,
        False,
        rescale=False,
        rescaling="unit",
        initial_state=jnp.zeros((4,)),
    )
    params["weights_penalization_control_squared"] = (
        jnp.ones((dynamics.num_controls,)) * 0.1
    )
    params["constrain_initial_control"] = True
    params["initial_control"] = init_control

    problem = OptimalControlProblem(dynamics=dynamics, params=copy.deepcopy(params))
    solver = SQPOSQPSolver(program=problem)

    sol = solver.solve(problem_params=params)

    assert sol.status == 0
    assert np.allclose(
        np.asarray(sol.controls[0]), np.asarray(init_control), rtol=1e-6, atol=1e-6
    )


def test_sqp_osqp_linesearch_runs():
    dynamics, params = make_linear_params(
        4,
        False,
        False,
        rescale=False,
        rescaling="unit",
        initial_state=jnp.zeros((4,)),
    )
    params["penalize_control_reference"] = True

    solver_params = sqp_osqp_params(
        tol=1.0e-6,
        sqp_iters=3,
        linesearch=True,
        linesearch_alphas=[0.3, 0.7, 1.0],
    )

    problem = OptimalControlProblem(dynamics=dynamics, params=params)
    solver = SQPOSQPSolver(program=problem, params=solver_params)
    sol = solver.solve(problem_params=params)

    assert sol.status == 0
    assert sol.linesearch_alphas is not None
    assert len(sol.linesearch_alphas) == sol.num_iter
    assert np.isfinite(sol.convergence_error)
    assert sol.convergence_error <= solver_params["tol_convergence"]


def test_sqp_osqp_matches_unconstrained_sqp_linear():
    dynamics, params = make_linear_params(
        5,
        False,
        False,
        rescale=False,
        rescaling="unit",
        initial_state=jnp.array([0.2, -0.1, 0.05, 5.0]),
    )
    params["weights_penalization_control_squared"] = jnp.zeros((dynamics.num_controls,))

    problem = OptimalControlProblem(dynamics=dynamics, params=params)

    solver_params = sqp_osqp_params(
        tol=1.0e-8, sqp_iters=20, linesearch=False, osqp_eps=1e-8
    )

    solver_osqp = SQPOSQPSolver(program=problem, params=solver_params)
    sol_osqp = solver_osqp.solve(problem_params=params)
    assert sol_osqp.status == 0
    assert sol_osqp.convergence_error <= 1.0e-7

    solver_params = sqp_params(tol=1.0e-8, sqp_iters=20, linesearch=False)

    solver_sqp = SQPDiffMPCSolver(program=problem, params=solver_params)
    sol_sqp = solver_sqp.solve(solver_sqp.initial_guess(params), problem_params=params)
    assert sol_sqp.status == SolverReturnStatus.SUCCESS


def test_sqp_osqp_slack_variables_resolve_infeasible_linear_ineq():
    dynamics, params = make_linear_params(
        3, False, True, rescale=False, rescaling="unit"
    )
    params["state_min_bounds"] = -jnp.ones((dynamics.num_states,)) * 0.05
    params["state_max_bounds"] = jnp.ones((dynamics.num_states,)) * 0.05
    params["use_slack_variables"] = True
    params["slack_penalization_weight"] = 10.0

    problem = OptimalControlProblemSlack(dynamics=dynamics, params=params)
    solver = SQPOSQPSolver(program=problem)
    sol = solver.solve(problem_params=params)

    assert sol.status == 0
    assert sol.slack.shape == (
        params["horizon"] + 1,
        dynamics.num_states + dynamics.num_controls,
    )
    assert_slack_active(sol)

    g, l, u = problem.inequality_constraints(sol.states, sol.controls, params)
    assert jnp.any((g < l - 1.0e-6) | (g > u + 1.0e-6))

    g = (g + sol.slack).reshape(-1)
    l = l.reshape(-1)
    u = u.reshape(-1)
    assert jnp.all(g >= l - 1.0e-6)
    assert jnp.all(g <= u + 1.0e-6)


@pytest.mark.parametrize(
    "case",
    [
        dict(name="linear", dynamics="linear", horizon=5, linesearch=False),
        dict(
            name="spacecraft_nonconvex",
            dynamics="spacecraft",
            horizon=6,
            linesearch=True,
        ),
    ],
)
def test_sqp_osqp_matches_unconstrained_sqp(case):
    if case["dynamics"] == "linear":
        dynamics, params = make_linear_params(
            case["horizon"],
            False,
            False,
            rescale=False,
            rescaling="unit",
            initial_state=jnp.array([0.2, -0.1, 0.05, 5.0]),
        )
        params["weights_penalization_control_squared"] = jnp.zeros(
            (dynamics.num_controls,)
        )
        problem = OptimalControlProblem(dynamics=dynamics, params=params)
    else:
        params = dict(load_problem_params("spacecraft.yaml"))
        params["horizon"] = case["horizon"]
        params["initial_state"] = jnp.array([0.2, -0.1, 0.05])
        params["initial_guess_final_state"] = jnp.zeros((3,))
        params["reference_state_trajectory"] = jnp.zeros((case["horizon"] + 1, 3))
        params["reference_control_trajectory"] = jnp.zeros((case["horizon"] + 1, 3))
        params["weights_penalization_control_squared"] = jnp.zeros((3,))
        params["weights_penalization_control_rate"] = jnp.zeros((3,))
        params["penalize_control_reference"] = False
        tile_spacecraft_inertia(params)
        problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)

    solver_osqp_params = sqp_osqp_params(
        tol=1.0e-6, sqp_iters=20, linesearch=case["linesearch"]
    )
    solver_osqp = SQPOSQPSolver(program=problem, params=solver_osqp_params)
    sol_osqp = solver_osqp.solve(problem_params=params)
    assert sol_osqp.status == 0
    assert sol_osqp.convergence_error <= solver_osqp_params["tol_convergence"]

    solver_sqp_params = sqp_params(
        tol=1.0e-6, sqp_iters=20, linesearch=case["linesearch"]
    )
    solver_sqp = SQPDiffMPCSolver(program=problem, params=solver_sqp_params)
    sol_sqp = solver_sqp.solve(solver_sqp.initial_guess(params), problem_params=params)
    assert sol_sqp.status == SolverReturnStatus.SUCCESS
    assert sol_sqp.convergence_error <= solver_sqp_params["tol_convergence"]

    z_osqp = pack_z(sol_osqp.states, sol_osqp.controls)
    z_sqp = pack_z(sol_sqp.states, sol_sqp.controls)
    max_diff = jnp.max(jnp.abs(z_osqp - z_sqp))
    assert float(max_diff) < Z_TOL

    cost_osqp = problem.cost(sol_osqp.states, sol_osqp.controls, params)
    cost_sqp = problem.cost(sol_sqp.states, sol_sqp.controls, params)
    assert float(jnp.abs(cost_osqp - cost_sqp)) < COST_TOL

    eq_osqp = problem.equality_constraints(sol_osqp.states, sol_osqp.controls, params)
    eq_sqp = problem.equality_constraints(sol_sqp.states, sol_sqp.controls, params)
    assert float(jnp.max(jnp.abs(eq_osqp))) < EQ_TOL
    assert float(jnp.max(jnp.abs(eq_sqp))) < EQ_TOL


def test_sqp_osqp_handles_obstacle_avoidance():
    solver_params = sqp_osqp_params(tol=1e-3, sqp_iters=10, linesearch=True)
    problem_params, dynamics = make_drone_params(
        horizon=load_problem_params("drone.yaml")["horizon"],
        obs_centers=jnp.array([[-1.4, -0.1], [-0.7, 0.3], [-0.3, 0.25]]),
        obs_radii=jnp.array([0.3, 0.2, 0.2]),
    )

    problem = OptimalControlProblemObstacle(dynamics=dynamics, params=problem_params)
    solver = SQPOSQPSolver(program=problem, params=solver_params)
    tt = time()
    solution = solver.solve(problem_params=problem_params)
    elapsed = time() - tt

    g, g_l, g_u = problem.inequality_constraints(
        solution.states, solution.controls, problem.params
    )

    assert solution.status == 0
    assert elapsed < 50.0
    assert solution.convergence_error < 1e-2
    assert_equality_residual(problem, solution, problem.params, tol=EQ_TOL)
    assert jnp.all(g >= g_l - 1e-6) and jnp.all(g <= g_u + 1e-6)
