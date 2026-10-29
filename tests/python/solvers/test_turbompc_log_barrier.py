"""Log-barrier-specific end-to-end TurboMPC SQP tests.

These tests cover what only the barrier has: its globalization
(interior line search), the strict-interior guarantee, one-sided
rows, the log_barrier_kappa -> 0 limit, and barrier-domain recovery.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.constraint_handling_fixtures import (
    OBSTACLE_CENTER,
    OBSTACLE_HORIZON,
    OBSTACLE_RADIUS,
    LowerObstacleProblem,
    assert_interior,
    inequality_violation,
    linear_problem,
    obstacle_problem,
    sqp_solver_params,
)
from turbompc.solvers.status import SQPStatus
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)


def _solve(problem, params, solver_params, guess=None):
    solver = TurboMPCSolver(
        program=problem,
        params=solver_params,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
    )
    return solver, solver.solve(
        solver.initial_guess(params) if guess is None else guess, params
    )


@pytest.mark.parametrize("linesearch", ["none", "merit", "filter"])
@pytest.mark.parametrize(
    ("use_slack", "rescale"), [(False, False), (True, False), (False, True)]
)
def test_linear_bounds_converge_strictly_inside(linesearch, use_slack, rescale):
    problem, params = linear_problem(use_slack=use_slack, rescale=rescale)
    _, solution = _solve(problem, params, sqp_solver_params(linesearch=linesearch))

    assert int(solution.status) == int(SQPStatus.SOLVED)
    assert inequality_violation(problem, solution, params) == 0.0
    if not use_slack:
        assert jnp.all(solution.slack == 0.0)


@pytest.mark.parametrize("linesearch", ["none", "merit", "filter"])
def test_nonlinear_obstacle_converges_strictly_inside(linesearch):
    problem, params = obstacle_problem()
    _, solution = _solve(
        problem, params, sqp_solver_params(linesearch=linesearch, sqp_iterations=60)
    )

    assert int(solution.status) == int(SQPStatus.SOLVED)
    assert inequality_violation(problem, solution, params) == 0.0
    distances = jnp.linalg.norm(solution.states[:, :2] - OBSTACLE_CENTER, axis=-1)
    assert float(jnp.min(distances)) >= OBSTACLE_RADIUS


def test_one_sided_nonlinear_bounds_converge():
    """The obstacle row has a finite lower and an infinite upper bound."""
    problem, params = obstacle_problem(problem_class=LowerObstacleProblem)
    _, solution = _solve(problem, params, sqp_solver_params(sqp_iterations=60))

    assert int(solution.status) == int(SQPStatus.SOLVED)
    assert inequality_violation(problem, solution, params) == 0.0


def test_solution_approaches_projection_as_log_barrier_kappa_decreases():
    problem, params = linear_problem()
    _, reference = _solve(problem, params, sqp_solver_params(0.0))
    assert int(reference.status) == int(SQPStatus.SOLVED)

    errors = []
    for log_barrier_kappa in (1.0e-2, 1.0e-4, 1.0e-6):
        _, solution = _solve(problem, params, sqp_solver_params(log_barrier_kappa))
        assert int(solution.status) == int(SQPStatus.SOLVED)
        errors.append(float(jnp.max(jnp.abs(solution.states - reference.states))))

    # The barrier bias is O(log_barrier_kappa): each 100x smaller
    # log_barrier_kappa shrinks it ~100x.
    assert errors[1] < 0.05 * errors[0]
    assert errors[2] < 0.05 * errors[1]


def test_infeasible_primal_start_gets_an_interior_image():
    """Starting inside the obstacle, ``g(x)`` is outside the bounds.

    The pinned initial state keeps the stage-0 row infeasible, so the SQP
    cannot converge; it must stay finite instead.
    """
    inside = jnp.array([1.0, 0.55, 0.0, 0.0])
    problem, params = obstacle_problem(start=inside)
    states0 = jnp.tile(inside, (OBSTACLE_HORIZON + 1, 1))
    controls0 = jnp.zeros((OBSTACLE_HORIZON + 1, 2))
    values, lower, upper = problem.inequality_constraints(states0, controls0, params)
    values = values.reshape((OBSTACLE_HORIZON + 1, -1))
    lower, upper = lower.reshape(values.shape), upper.reshape(values.shape)
    assert bool(jnp.any((values < lower) | (values > upper)))

    solver, solution = _solve(problem, params, sqp_solver_params(sqp_iterations=5))
    barrier_state = solver._constraint_handler.init_carry(
        states0, controls0, jnp.zeros_like(values), params
    )
    assert_interior(barrier_state.s_g, lower, upper)
    assert int(solution.status) == int(SQPStatus.MAX_ITER)
    assert bool(jnp.all(jnp.isfinite(solution.states)))


def test_warm_start_outside_the_barrier_domain_recovers():
    """The inner proximal accepts any ``z_g``; no warm-start repair is needed."""
    problem, params = obstacle_problem()
    solver, cold = _solve(problem, params, sqp_solver_params(sqp_iterations=60))
    outside = cold.admm_state._replace(z_g=jnp.full_like(cold.admm_state.z_g, 1.0e8))
    warm = solver.solve(cold._replace(admm_state=outside), params)

    assert int(warm.status) == int(SQPStatus.SOLVED)
    np.testing.assert_allclose(warm.states, cold.states, atol=1.0e-6)


def test_nonfinite_warm_start_is_reported_not_repaired():
    problem, params = linear_problem()
    solver, cold = _solve(problem, params, sqp_solver_params(sqp_iterations=3))

    broken_guess = cold._replace(states=jnp.full_like(cold.states, jnp.nan))
    assert int(solver.solve(broken_guess, params).status) == int(SQPStatus.NONFINITE)

    # Unlike an out-of-domain z_g, a NaN one is flagged by the inner ADMM.
    broken_dual = cold._replace(
        admm_state=cold.admm_state._replace(
            z_g=jnp.full_like(cold.admm_state.z_g, jnp.nan)
        )
    )
    assert int(solver.solve(broken_dual, params).status) == int(SQPStatus.QP_FAILED)
