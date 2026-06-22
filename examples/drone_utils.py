"""Drone obstacle-avoidance utilities for the tutorial notebook.

Consolidates the constants, problem/solver configuration, and closed-loop
rollout helper that were previously spread across three separate scripts.
"""

from __future__ import annotations

import copy
import os
import warnings

import numpy as np

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*scatter inputs have incompatible types.*",
)

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_threefry_partitionable", True)

_JAX_CACHE = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax_turbompc")
)
config.update("jax_compilation_cache_dir", _JAX_CACHE)

import jax.numpy as jnp  # noqa: E402
from turbompc.dynamics.drone_dynamics import (
    DroneDynamics,
    drone_parameters,
    drone_state_dot_parameters,
)
from turbompc.dynamics.integrators import DiscretizationScheme, predict_next_state
from turbompc.problems.obstacle_avoidance import (
    OptimalControlProblemObstacle,
    OptimalControlProblemObstacleSlack,
)
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)
from turbompc.utils.load_params import load_problem_params, load_solver_params
from turbompc.utils.timing import ProblemConfig, _single_step_dynamics_params

# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------

DRONE_NX = 6
DRONE_NU = 3
DRONE_HORIZON = 50
DRONE_DT = 1.0
DRONE_UMAX = 10.0
DRONE_MASS = 32.0
DRONE_DRAG_COEFF = 0.2

OBS_CENTERS = np.array([[-1.4, -0.1], [-0.7, 0.3], [-0.3, 0.25]], dtype=np.float64)
OBS_RADII = np.array([0.3, 0.2, 0.2], dtype=np.float64)
DRONE_X0_BASE = np.array([-1.9, 0.05, 0.2, 0.0, 0.0, 0.0], dtype=np.float64)
DRONE_X0_NOISE_STD = 0.05

DRONE_Q_DIAG = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
DRONE_QN_DIAG = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
DRONE_R_DIAG = np.array([1.0, 1.0, 1.0], dtype=np.float64)

# JAX copies of cost weights (used internally)
_Q_DIAG = jnp.array(DRONE_Q_DIAG)
_R_DIAG = jnp.array(DRONE_R_DIAG)

# Default obstacle layout as JAX arrays
_DEFAULT_OBS_CENTERS = jnp.array(OBS_CENTERS)
_DEFAULT_OBS_RADII = jnp.array(OBS_RADII)


def generate_drone_x0(batch_size: int, seed: int) -> np.ndarray:
    """Return (batch_size, DRONE_NX) initial states with noise around DRONE_X0_BASE."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, DRONE_X0_NOISE_STD, size=(batch_size, DRONE_NX))
    return np.broadcast_to(DRONE_X0_BASE, (batch_size, DRONE_NX)) + noise


# ---------------------------------------------------------------------------
# Problem / solver parameter builders
# ---------------------------------------------------------------------------


def _make_problem_params(
    use_slack: bool,
    slack_weight: float,
    obs_centers: jnp.ndarray,
    obs_radii: jnp.ndarray,
    discretization_scheme: int = 10,
    dt: float = None,
    horizon: int = None,
    rd_weight: float = 0.01,
) -> dict:
    base = dict(load_problem_params("drone.yaml"))
    if horizon is not None:
        base["horizon"] = horizon
    N = int(base["horizon"])
    dynamics = DroneDynamics(drone_parameters)
    nx = dynamics.num_states
    nu = dynamics.num_controls
    base["reference_state_trajectory"] = jnp.zeros((N + 1, nx))
    base["reference_control_trajectory"] = jnp.zeros((N + 1, nu))
    base["obstacles_centers"] = obs_centers
    base["obstacles_radii"] = obs_radii
    base["obstacles_dimension"] = 2
    base["rescale_optimization_variables"] = False
    base["use_slack_variables"] = use_slack
    base["slack_penalization_weight"] = slack_weight
    base["weights_penalization_control_rate"] = [rd_weight] * nu
    base["discretization_scheme"] = discretization_scheme
    # constrain_initial_control is only meaningful for the implicit (trapezoidal) scheme
    if DiscretizationScheme(discretization_scheme) != DiscretizationScheme.IMPLICIT:
        base["constrain_initial_control"] = False
    if dt is not None:
        base["discretization_resolution"] = dt
    base["dynamics_state_dot_params"] = {
        key: jnp.array([value] * N) for key, value in drone_state_dot_parameters.items()
    }
    return base


def _make_solver_params(
    warm_start: bool,
    alpha: float,
    pcg_eps: float,
    sqp_iter: int,
    admm_tol: float,
    admm_max_iter: int = 1000,
    sqp_tol: float = 1e-4,
) -> dict:
    sp = load_solver_params("turbompc.yaml")
    sp["num_sqp_iteration_max"] = sqp_iter
    sp["tol_convergence"] = sqp_tol
    sp["warm_start_backward"] = warm_start
    sp["linesearch"] = True
    sp["linesearch_alphas"] = [0.1, 0.3, 0.7, 1.0]
    sp["admm"]["max_iter"] = admm_max_iter
    sp["admm"]["check_termination_every"] = 5
    sp["admm"]["eps_abs"] = admm_tol
    sp["admm"]["eps_rel"] = admm_tol
    sp["admm"]["relaxation_parameter"] = alpha
    sp["admm"]["pcg"]["tol_epsilon"] = pcg_eps
    return sp


def _reward(state, control):
    return -(jnp.sum(_Q_DIAG * state**2) + jnp.sum(_R_DIAG * control**2))


def _update_per_seed(seed, batch_size, problem_params):
    rng = np.random.default_rng(seed)
    base_x0 = jnp.array(DRONE_X0_BASE)
    noise = jnp.array(rng.normal(0, DRONE_X0_NOISE_STD, size=(batch_size, DRONE_NX)))
    x0 = jnp.broadcast_to(base_x0, (batch_size, DRONE_NX)) + noise
    return {}, x0


# ---------------------------------------------------------------------------
# Public config factory
# ---------------------------------------------------------------------------


def make_drone_config(
    use_slack: bool = True,
    slack_weight: float = 10.0,
    obs_centers: jnp.ndarray = None,
    obs_radii: jnp.ndarray = None,
    warm_start: bool = True,
    alpha: float = 1.6,
    pcg_eps: float = 1e-15,
    sqp_iter: int = 8,
    admm_tol: float = 1e-3,
    admm_max_iter: int = 1000,
    discretization_scheme: int = 10,
    dt: float = None,
    horizon: int = None,
    rd_weight: float = 0.01,
    sqp_tol: float = 1e-4,
) -> ProblemConfig:
    """Build a ProblemConfig for the drone obstacle-avoidance OCP."""
    if obs_centers is None:
        obs_centers = _DEFAULT_OBS_CENTERS
    if obs_radii is None:
        obs_radii = _DEFAULT_OBS_RADII
    problem_class = (
        OptimalControlProblemObstacleSlack
        if use_slack
        else OptimalControlProblemObstacle
    )
    return ProblemConfig(
        dynamics=DroneDynamics(drone_parameters),
        problem_class=problem_class,
        problem_params=_make_problem_params(
            use_slack,
            slack_weight,
            obs_centers,
            obs_radii,
            discretization_scheme,
            dt=dt,
            horizon=horizon,
            rd_weight=rd_weight,
        ),
        solver_params=_make_solver_params(
            warm_start, alpha, pcg_eps, sqp_iter, admm_tol, admm_max_iter, sqp_tol
        ),
        weight_keys=["weights_penalization_reference_state_trajectory"],
        reward_fn=_reward,
        update_per_seed=_update_per_seed,
    )


# ---------------------------------------------------------------------------
# Warm-start helpers
# ---------------------------------------------------------------------------


def _zeros_initial_guess(solver, problem_params: dict):
    """All-states and controls initialised to zero."""
    guess = solver.initial_guess(problem_params)
    return guess._replace(
        states=jnp.zeros_like(guess.states),
        controls=jnp.zeros_like(guess.controls),
    )


def _arc_initial_guess(solver, problem_params: dict, x0: np.ndarray):
    """Replace straight-line initial-guess states with a y-arc that clears all obstacles."""
    guess = solver.initial_guess(problem_params)
    N = guess.states.shape[0] - 1
    arc_peak_y = 0.7
    arc_states = np.array(guess.states)
    for k in range(N + 1):
        t = k / N
        arc_states[k, 0] = x0[0] * (1 - t)
        arc_states[k, 1] = x0[1] * (1 - t) + arc_peak_y * 4 * t * (1 - t)
        arc_states[k, 2] = x0[2] * (1 - t)
    return guess._replace(states=jnp.array(arc_states))


# ---------------------------------------------------------------------------
# Obstacle margin helper
# ---------------------------------------------------------------------------


def _obstacle_margins(state: np.ndarray) -> np.ndarray:
    """Return (n_obs,) margin values h_i = 1 - dist/r  (>0 means inside obstacle)."""
    px, py = state[0], state[1]
    return np.array(
        [
            1.0 - np.sqrt((px - c[0]) ** 2 + (py - c[1]) ** 2 + 1e-12) / (r + 0.001)
            for c, r in zip(OBS_CENTERS, OBS_RADII)
        ]
    )


# ---------------------------------------------------------------------------
# Closed-loop rollout
# ---------------------------------------------------------------------------


def collect_turbompc_drone_trajectory(
    cfg,
    sim_steps: int,
    seed: int = 0,
    fwd_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
    bwd_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
    sqp_iter: int = None,
    arc_init: bool = False,
    zeros_init: bool = False,
    solver=None,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Receding-horizon closed-loop MPC rollout.

    Returns:
        traj:    (sim_steps+1, nx) executed state trajectory
        metrics: dict with quality metrics (obstacle margins, violations, terminal error)
        u_arr:   (sim_steps, nu) applied controls
    """
    problem_params = dict(cfg.problem_params)
    param_updates, initial_states = cfg.update_per_seed(seed, 1, problem_params)
    problem_params.update(param_updates)

    solver_params = copy.deepcopy(cfg.solver_params)
    if sqp_iter is not None:
        solver_params["num_sqp_iteration_max"] = sqp_iter

    if solver is None:
        problem = cfg.problem_class(dynamics=cfg.dynamics, params=problem_params)
        solver = TurboMPCSolver(
            program=problem,
            params=solver_params,
            forward_backend=fwd_backend,
            backward_backend=bwd_backend,
        )

    dt = float(problem_params["discretization_resolution"])
    scheme = DiscretizationScheme(int(problem_params["discretization_scheme"]))
    sim_dyn_params = _single_step_dynamics_params(problem_params)

    state = jnp.array(initial_states[0])
    weights = {k: problem_params[k] for k in cfg.weight_keys}
    if arc_init:
        init_guess = _arc_initial_guess(solver, problem_params, np.array(state))
    elif zeros_init:
        init_guess = _zeros_initial_guess(solver, problem_params)
    else:
        init_guess = solver.initial_guess(problem_params)  # straight-line x0 -> goal
    solution = solver.solve(init_guess, problem_params, weights)

    traj = [np.array(state)]
    u_log = []
    sqp_iters_per_step = [int(solution.num_iter)]
    total_cost = 0.0

    for _ in range(sim_steps):
        # Horizon-shift warm-start: drop step 0, repeat last step.
        shifted_states = jnp.concatenate(
            [solution.states[1:], solution.states[-1:]], axis=0
        )
        shifted_controls = jnp.concatenate(
            [solution.controls[1:], solution.controls[-1:]], axis=0
        )
        shifted_slack = jnp.concatenate(
            [solution.slack[1:], solution.slack[-1:]], axis=0
        )
        warm = solution._replace(
            states=shifted_states,
            controls=shifted_controls,
            slack=shifted_slack,
        )
        solution = solver.solve(
            warm, problem_params, {**weights, "initial_state": state}
        )
        u0 = solution.controls[0]
        # For implicit trapezoidal the OCP dynamics constraint uses u[t+1].
        u1 = solution.controls[1] if scheme == DiscretizationScheme.IMPLICIT else u0
        state = predict_next_state(
            cfg.dynamics, dt, scheme, sim_dyn_params, state, u0, u1
        )
        total_cost += float(
            np.sum(DRONE_Q_DIAG * np.array(state) ** 2)
            + np.sum(DRONE_R_DIAG * np.array(u0) ** 2)
        )
        traj.append(np.array(state))
        u_log.append(np.array(u0))
        sqp_iters_per_step.append(int(solution.num_iter))

    traj_arr = np.array(traj)
    u_arr = np.array(u_log)
    delta_u = np.diff(u_arr, axis=0)
    mean_delta_u = (
        float(np.mean(np.linalg.norm(delta_u, axis=1))) if len(delta_u) > 0 else 0.0
    )

    all_margins = np.array([_obstacle_margins(s) for s in traj_arr])
    max_margin = float(np.max(all_margins))
    violated = all_margins > 1e-4
    n_violations = int(np.sum(violated))
    terminal_err = float(np.linalg.norm(traj_arr[-1]))
    max_sqp = solver_params["num_sqp_iteration_max"]

    metrics = {
        "solver": "turbompc",
        "arc_init": arc_init,
        "zeros_init": zeros_init,
        "seed": seed,
        "sim_steps": sim_steps,
        "n_violations": n_violations,
        "violation_fraction": n_violations / (traj_arr.shape[0] * all_margins.shape[1]),
        "max_violation": max(0.0, max_margin),
        "terminal_state_norm": terminal_err,
        "mean_delta_u": mean_delta_u,
        "obs_margins": all_margins.tolist(),
        "sqp_iters_per_step": sqp_iters_per_step,
        "failure_count": sum(i >= max_sqp for i in sqp_iters_per_step),
        "cost": total_cost,
    }
    return traj_arr, metrics, u_arr
