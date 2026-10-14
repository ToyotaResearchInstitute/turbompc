"""Problems and settings shared by the projection and log-barrier tests."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.problem_fixtures import cost_blocks_from_qr, make_linear_params
from turbompc.problems.obstacle_avoidance import OptimalControlProblemObstacle
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.constraint_handling import ConstraintHandling
from turbompc.solvers.qp_data import qpdata_from_ocp_blocks
from turbompc.solvers.qp_utils import ZShape
from turbompc.utils.load_params import load_solver_params

BARRIER_KAPPA = 1.0e-4
SLACK_WEIGHT = 1.0e2


def handlings(barrier_kappa: float = BARRIER_KAPPA):
    """``(constraint_handling, log_barrier_kappa)`` params.

    Projection requires a zero ``log_barrier_kappa``.
    """
    return [
        pytest.param(ConstraintHandling.PROJECTION, 0.0, id="projection"),
        pytest.param(ConstraintHandling.LOG_BARRIER, barrier_kappa, id="log_barrier"),
    ]


HANDLINGS = handlings()


def is_barrier(log_barrier_kappa: float) -> bool:
    return log_barrier_kappa > 0.0


def with_handling(solver_params: dict, log_barrier_kappa: float) -> dict:
    """Return solver params using the handling ``log_barrier_kappa`` selects."""
    params = copy.deepcopy(solver_params)
    params["constraint_handling"] = (
        "log_barrier" if is_barrier(log_barrier_kappa) else "projection"
    )
    params["log_barrier_kappa"] = log_barrier_kappa
    return params


# ---------------------------------------------------------------------------
# Small dense-inequality OCP QP for the ADMM-level tests.
# ---------------------------------------------------------------------------

QP_HORIZON, QP_NUM_STATES, QP_NUM_CONTROLS = 4, 2, 1
QP_PCG_PARAMS = {"max_iter": 200, "tol_epsilon": 1.0e-10}


def _bound_pattern(kind: str, dtype):
    """Per-row bounds for the ``nx + nu`` inequality rows of every stage."""
    num_rows = QP_NUM_STATES + QP_NUM_CONTROLS
    lower = np.full(num_rows, -0.7)
    upper = np.full(num_rows, 0.7)
    if kind == "mixed":
        # Two-sided, upper-only and lower-only rows, plus one unbounded row.
        lower = np.array([-0.7, -np.inf, -0.4])
        upper = np.array([0.7, 0.5, np.inf])
    elif kind != "two_sided":
        raise ValueError(kind)
    lower = np.tile(lower, (QP_HORIZON + 1, 1))
    upper = np.tile(upper, (QP_HORIZON + 1, 1))
    if kind == "mixed":
        lower[1, 0] = -np.inf
        upper[1, 0] = np.inf
    return jnp.asarray(lower, dtype=dtype), jnp.asarray(upper, dtype=dtype)


def make_qp(
    bounds: str = "two_sided",
    *,
    slack_penalization_weight: float = 0.0,
    seed: int = 0,
    dtype=jnp.float64,
):
    """Build a small OCP QP whose unconstrained optimum violates the bounds."""
    rng = np.random.default_rng(seed)
    N, nx, nu = QP_HORIZON, QP_NUM_STATES, QP_NUM_CONTROLS

    def normal(scale, shape):
        return jnp.asarray(rng.normal(scale=scale, size=shape), dtype=dtype)

    eye = jnp.eye(nx, dtype=dtype)
    As, Bs = normal(0.4, (N, nx, nx)), normal(0.4, (N, nx, nu))
    c_dyn = normal(0.1, (N, nx))
    Qmat = jnp.tile(eye[None], (N + 1, 1, 1))
    Rmat = jnp.tile(jnp.eye(nu, dtype=dtype)[None], (N + 1, 1, 1)) * 0.1
    Rd = jnp.tile(jnp.eye(nu, dtype=dtype)[None], (N, 1, 1)) * 0.05
    # Large linear terms push the unconstrained optimum outside the bounds.
    qvec, rvec = normal(2.0, (N + 1, nx)), normal(2.0, (N + 1, nu))
    D, E, q = cost_blocks_from_qr(Qmat, Rmat, Rd, qvec, rvec)
    ineq_l, ineq_u = _bound_pattern(bounds, dtype)

    qp_data = qpdata_from_ocp_blocks(
        D=D,
        E=E,
        q=q,
        A0=jnp.concatenate([eye, jnp.zeros((nx, nu), dtype=dtype)], 1),
        c0=normal(0.1, (nx,)),
        As_next=jnp.tile(-eye[None], (N, 1, 1)),
        Bs_next=jnp.zeros((N, nx, nu), dtype=dtype),
        As=As,
        Bs=Bs,
        c_dyn=c_dyn,
        ineq_blocks=jnp.tile(jnp.eye(nx + nu, dtype=dtype)[None], (N + 1, 1, 1)),
        ineq_l=ineq_l,
        ineq_u=ineq_u,
        slack_penalization_weight=jnp.asarray(slack_penalization_weight, dtype=dtype),
    )
    return qp_data, ZShape(horizon=N, num_states=nx, num_controls=nu)


# ---------------------------------------------------------------------------
# SQP problems.
# ---------------------------------------------------------------------------


def sqp_solver_params(
    log_barrier_kappa: float = BARRIER_KAPPA,
    *,
    linesearch: str = "none",
    sqp_iterations: int = 25,
    admm_iterations: int = 4000,
    tol_convergence: float = 1.0e-6,
    admm_eps: float = 1.0e-10,
):
    params = copy.deepcopy(load_solver_params("turbompc.yaml"))
    params["num_sqp_iteration_max"] = sqp_iterations
    params["tol_convergence"] = tol_convergence
    params["admm"]["max_iter"] = admm_iterations
    params["admm"]["eps_abs"] = admm_eps
    params["admm"]["eps_rel"] = admm_eps
    params["linesearch"]["method"] = linesearch
    return with_handling(params, log_barrier_kappa)


def linear_problem(*, use_slack: bool = False, rescale: bool = False, horizon: int = 5):
    """Double integrator that brakes from unit velocity.

    The optimum saturates the lower control bound early and the lower velocity
    bound late (indices 0 and 2 of the respective bound vectors); the upper
    bounds are inactive.
    """
    dynamics, params = make_linear_params(
        horizon=horizon,
        implicit=False,
        bounded=True,
        rescale=rescale,
        initial_state=jnp.ones((4,), dtype=jnp.float64),
    )
    params = copy.deepcopy(params)
    params["state_min_bounds"] = jnp.array([-10.0, -10.0, 0.93, 0.93])
    params["state_max_bounds"] = jnp.full((4,), 10.0)
    params["control_min_bounds"] = jnp.full((2,), -0.2)
    params["control_max_bounds"] = jnp.full((2,), 0.2)
    if use_slack:
        params["use_slack_variables"] = True
        params["slack_penalization_weight"] = SLACK_WEIGHT
    return OptimalControlProblem(dynamics, params), params


OBSTACLE_HORIZON = 5
_OBSTACLE_START = jnp.array([0.0, 0.0, 0.0, 0.0])
_OBSTACLE_GOAL = jnp.array([2.0, 1.5, 0.0, 0.0])
OBSTACLE_CENTER = jnp.array([1.0, 0.55])
OBSTACLE_RADIUS = 0.55


class LowerObstacleProblem(OptimalControlProblem):
    """Obstacle avoidance as a lower-bounded row with an infinite upper bound."""

    def step_inequality_constraints(self, state, control, params):
        g, g_l, g_u = super().step_inequality_constraints(state, control, params)
        position = state[: self.params["obstacles_dimension"]]
        centers = jnp.asarray(params["obstacles_centers"])
        radii = jnp.asarray(params["obstacles_radii"])

        def clearance(position, center, radius):
            return jnp.linalg.norm(position - center) / (radius + 0.001) - 1.0

        g_obs = jax.vmap(clearance, in_axes=(None, 0, 0))(position, centers, radii)
        return (
            jnp.concatenate([g, g_obs]),
            jnp.concatenate([g_l, jnp.zeros_like(g_obs)]),
            jnp.concatenate([g_u, jnp.full_like(g_obs, jnp.inf)]),
        )


def obstacle_problem(
    problem_class=OptimalControlProblemObstacle, start=_OBSTACLE_START
):
    """Double integrator that must pass a disk between start and goal."""
    horizon = OBSTACLE_HORIZON
    dynamics, params = make_linear_params(
        horizon, implicit=False, bounded=False, initial_state=start
    )
    params = copy.deepcopy(params)
    params.update(
        discretization_resolution=0.1,
        initial_state=start,
        initial_guess_final_state=_OBSTACLE_GOAL,
        reference_state_trajectory=jnp.tile(_OBSTACLE_GOAL, (horizon + 1, 1)),
        weights_penalization_reference_state_trajectory=jnp.array([5.0, 5.0, 0.0, 0.0]),
        weights_penalization_control_squared=jnp.array([0.1, 0.1]),
        weights_penalization_final_state=jnp.zeros((4,)),
        dynamics_state_dot_params={
            "A": jnp.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "B": jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            "b": jnp.zeros((4,)),
        },
        obstacles_centers=jnp.tile(OBSTACLE_CENTER[None, None, :], (horizon + 1, 1, 1)),
        obstacles_radii=jnp.array([OBSTACLE_RADIUS]),
        obstacles_dimension=2,
        rescale_optimization_variables=False,
    )
    return problem_class(dynamics=dynamics, params=params), params


def inequality_violation(problem, solution, params) -> float:
    """Largest bound violation of ``g(x) + slack``."""
    values, lower, upper = problem.inequality_constraints(
        solution.states, solution.controls, params
    )
    values = values.reshape(solution.slack.shape) + solution.slack
    lower, upper = lower.reshape(values.shape), upper.reshape(values.shape)
    return float(jnp.max(jnp.maximum(jnp.maximum(lower - values, values - upper), 0.0)))


def assert_interior(values, lower, upper):
    assert bool(jnp.all(jnp.where(jnp.isfinite(lower), values > lower, True)))
    assert bool(jnp.all(jnp.where(jnp.isfinite(upper), values < upper, True)))
