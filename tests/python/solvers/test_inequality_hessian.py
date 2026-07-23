"""Test for inequality-constraint Lagrangian Hessian (mu^T grad^2 g) in the TurboMPC backward.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import vmap

from tests.helpers.problem_fixtures import make_linear_params
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.problems.obstacle_avoidance import OptimalControlProblemObstacle
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend
from turbompc.utils.load_params import load_solver_params, load_problem_params
from turbompc.utils.gradient_finitediff import gradient_finite_diff


class OptimalControlProblemLowerObstacle(OptimalControlProblem):
    """Obstacle avoidance encoded as a lower-active nonlinear constraint:
    g(x) = ||p - c|| / (r + eps) - 1 >= 0 (g_l = 0, g_u = +1e9)
    The problem is equivalent to OptimalControlProblemObstacle, which uses upper-active
    constraints g(x) = 1 - ||p-c||/(r+eps) <= 0
    """

    def step_inequality_constraints(self, state, control, params):
        g, g_l, g_u = super().step_inequality_constraints(state, control, params)
        position = state[: self.params["obstacles_dimension"]]
        centers = jnp.asarray(params["obstacles_centers"])
        radii = jnp.asarray(params["obstacles_radii"])

        def lower_obstacle(position, obs_center, obs_radius):
            # >= 0 outside the obstacle; lower bound (0) binds at the boundary.
            return jnp.linalg.norm(position - obs_center) / (obs_radius + 0.001) - 1.0

        g_obs = vmap(lower_obstacle, in_axes=(None, 0, 0))(position, centers, radii)
        g_obs_l = jnp.zeros_like(g_obs)
        g_obs_u = 1.0e9 * jnp.ones_like(g_obs)
        return (
            jnp.concatenate([g, g_obs]),
            jnp.concatenate([g_l, g_obs_l]),
            jnp.concatenate([g_u, g_obs_u]),
        )

NX, NU = 4, 2
START = jnp.array([0.0, 0.0, 0.0, 0.0])
GOAL = jnp.array([2.0, 1.5, 0.0, 0.0])         
OBS_C = jnp.array([1.0, 0.55])                  
OBS_R = 0.55
HORIZON, DT = 5, 0.1
QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"
WEIGHT_KEYS = [QK, RK]


def _build(use_full_hessian, ocp_cls=OptimalControlProblemObstacle):
    # complete validated base params (LinearDynamics nx=4, nu=2); override to a 2-D double
    # integrator + obstacle. bounded=False -> control bounds inert (only the obstacle binds).
    dyn, problem_params = make_linear_params(HORIZON, implicit=False, bounded=False, initial_state=START)
    A_c = jnp.array([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
    B_c = jnp.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])
    problem_params = dict(problem_params)
    problem_params["discretization_resolution"] = DT
    problem_params["initial_state"] = START
    problem_params["initial_guess_final_state"] = GOAL
    problem_params["reference_state_trajectory"] = jnp.tile(GOAL, (HORIZON + 1, 1))
    problem_params[QK] = jnp.array([5.0, 5.0, 0.0, 0.0])
    problem_params[RK] = jnp.array([0.1, 0.1])
    problem_params["weights_penalization_final_state"] = jnp.zeros((NX,))
    problem_params["dynamics_state_dot_params"] = {"A": A_c, "B": B_c, "b": jnp.zeros((NX,))}
    problem_params["obstacles_centers"] = jnp.tile(OBS_C[None, None, :], (HORIZON + 1, 1, 1))
    problem_params["obstacles_radii"] = jnp.array([OBS_R])
    problem_params["obstacles_dimension"] = 2
    problem_params["rescale_optimization_variables"] = False
    ocp = ocp_cls(dynamics=dyn, params=problem_params)
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 50
    solver = TurboMPCSolver(
        program=ocp, params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE,
        use_full_hessian=use_full_hessian)
    return solver, problem_params


def _loss(states, controls):
    return 0.5 * jnp.sum((states[:, :2] - GOAL[:2]) ** 2) + 0.5e-2 * jnp.sum(controls ** 2)


def _flat(d):
    return np.concatenate([np.asarray(d[k]).reshape(-1) for k in WEIGHT_KEYS])


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _ad_grad(solver, problem_params, loss_fn=_loss):
    ig = solver.initial_guess(problem_params)
    def loss_w(w):
        sol = solver.solve(ig, problem_params, {**{k: problem_params[k] for k in WEIGHT_KEYS}, **w})
        return loss_fn(sol.states, sol.controls)
    return jax.grad(loss_w)({k: problem_params[k] for k in WEIGHT_KEYS})


def _fd_grad(solver, problem_params, eps_seq=(1e-3, 3e-4, 1e-4), rtol=2e-3, atol=1e-7, loss_fn=_loss):
    """Finite difference check"""
    ig = solver.initial_guess(problem_params)

    def fwd(weights):
        sol = solver.solve(ig, problem_params, weights)
        return loss_fn(sol.states, sol.controls)

    base_w = {k: problem_params[k] for k in WEIGHT_KEYS}
    grads = [gradient_finite_diff(fwd, weights=base_w, eps=eps) for eps in eps_seq]
    grad, flagged = [], False
    for k in WEIGHT_KEYS:
        per_eps = [np.asarray(g[k]).reshape(-1) for g in grads]
        gk = np.zeros(per_eps[0].size)
        for i in range(gk.size):
            vals = [pe[i] for pe in per_eps]
            chosen, ok = vals[-1], False
            for a, b in zip(vals[:-1], vals[1:]):
                if abs(a - b) <= rtol * abs(b) + atol:
                    chosen, ok = a, True; break
            flagged = flagged or (not ok); gk[i] = chosen
        grad.append(gk)
    return np.concatenate(grad), flagged


def test_obstacle_active_and_dynamics_linear():
    """forward converges and the obstacle is active (so mu, grad^2 g != 0)."""
    solver, problem_params = _build(use_full_hessian=True)
    ig = solver.initial_guess(problem_params)
    sol = solver.solve(ig, problem_params, {k: problem_params[k] for k in WEIGHT_KEYS})
    assert float(sol.convergence_error) < 1e-5
    P = np.asarray(sol.states)[:, :2]
    dist = np.linalg.norm(P - np.asarray(OBS_C), axis=1)
    assert dist.min() < OBS_R + 0.02, f"obstacle not engaged (min dist {dist.min():.3f})"


def test_inequality_hessian_matches_fd():
    """use_full_hessian=True (inequality Hessian ON) -> AD matches convergence-checked FD."""
    solver, problem_params = _build(use_full_hessian=True)
    g_ad = _flat(_ad_grad(solver, problem_params))
    g_fd, flagged = _fd_grad(solver, problem_params)
    assert not flagged, "FD did not plateau"
    assert _cos(g_ad, g_fd) > 1 - 1e-4, f"AD vs FD cos={_cos(g_ad, g_fd)}"
    assert _rel(g_ad, g_fd) < 5e-3, f"AD vs FD rel_l2={_rel(g_ad, g_fd)}"


def test_without_inequality_hessian_is_worse():
    """Ablation: use_full_hessian=False (Gauss-Newton inequality) is worse vs FD."""
    solver_on, problem_params = _build(use_full_hessian=True)
    solver_off, _ = _build(use_full_hessian=False)
    g_on = _flat(_ad_grad(solver_on, problem_params))
    g_off = _flat(_ad_grad(solver_off, problem_params))
    g_fd, flagged = _fd_grad(solver_on, problem_params)
    assert not flagged
    rel_on, rel_off = _rel(g_on, g_fd), _rel(g_off, g_fd)
    assert rel_on < 5e-3, f"with-Hessian rel_l2={rel_on}"
    assert rel_off > 5 * rel_on, f"ablation not separated: on={rel_on:.2e} off={rel_off:.2e}"


def test_lower_obstacle_active_and_lower():
    """forward converges, the (lower-active) obstacle is engaged, and the obstacle's
    ADMM dual is NEGATIVE (= lower-active, y_g = -nu_l)"""
    solver, problem_params = _build(use_full_hessian=True, ocp_cls=OptimalControlProblemLowerObstacle)
    ig = solver.initial_guess(problem_params)
    sol = solver.solve(ig, problem_params, {k: problem_params[k] for k in WEIGHT_KEYS})
    assert float(sol.convergence_error) < 1e-5
    P = np.asarray(sol.states)[:, :2]
    dist = np.linalg.norm(P - np.asarray(OBS_C), axis=1)
    assert dist.min() < OBS_R + 0.05, f"obstacle not engaged (min dist {dist.min():.3f})"
    # obstacle is the LAST inequality column; its bound dual must be negative (lower-active).
    y_obs = np.asarray(sol.admm_state.y_g)[:, -1]
    assert y_obs.min() < -1e-6, f"obstacle dual not lower-active (min y_g={y_obs.min():.2e})"
    assert not (y_obs.max() > 1e-6 and y_obs.min() > -1e-6), "expected a lower-active (negative) dual"


def test_lower_active_inequality_hessian_matches_fd():
    """lower-active nonlinear constraint: AD (signed net dual y_g Hessian) matches
    convergence-checked FD. With -sign*y_g the Hessian sign flips for lower-active -> fails."""
    solver, problem_params = _build(use_full_hessian=True, ocp_cls=OptimalControlProblemLowerObstacle)
    g_ad = _flat(_ad_grad(solver, problem_params))
    g_fd, flagged = _fd_grad(solver, problem_params)
    assert not flagged, "FD did not plateau"
    assert _cos(g_ad, g_fd) > 1 - 1e-4, f"lower-active AD vs FD cos={_cos(g_ad, g_fd)}"
    assert _rel(g_ad, g_fd) < 5e-3, f"lower-active AD vs FD rel_l2={_rel(g_ad, g_fd)}"


def test_lower_active_matches_upper_active():
    """lower-active nonlinear constraints match upper-active (same feasible set + cost) in forward primals and AD gradient."""
    solver_up, problem_params_up = _build(use_full_hessian=True, ocp_cls=OptimalControlProblemObstacle)
    solver_lo, problem_params_lo = _build(use_full_hessian=True, ocp_cls=OptimalControlProblemLowerObstacle)
    # forward primals coincide (identical feasible set + cost)
    s_up = solver_up.solve(solver_up.initial_guess(problem_params_up), problem_params_up, {k: problem_params_up[k] for k in WEIGHT_KEYS})
    s_lo = solver_lo.solve(solver_lo.initial_guess(problem_params_lo), problem_params_lo, {k: problem_params_lo[k] for k in WEIGHT_KEYS})
    p_up, p_lo = np.asarray(s_up.states)[:, :2], np.asarray(s_lo.states)[:, :2]
    assert np.linalg.norm(p_up - p_lo) < 1e-3, "upper/lower forward primals should coincide"
    g_up = _flat(_ad_grad(solver_up, problem_params_up))
    g_lo = _flat(_ad_grad(solver_lo, problem_params_lo))
    assert _cos(g_up, g_lo) > 1 - 1e-4, f"lower vs upper AD cos={_cos(g_up, g_lo)} (sign bug?)"
    assert _rel(g_lo, g_up) < 5e-3, f"lower vs upper AD rel_l2={_rel(g_lo, g_up)}"


# --------------------------------------------------------------------------- #
# NONLINEAR DRONE DYNAMICS (quadratic drag, ∇²f != 0) + nonlinear inequality (obstacle).
# --------------------------------------------------------------------------- #
DRONE_NX, DRONE_NU = 6, 3
DRONE_START = jnp.array([-1.9, 0.05, 0.0, 0.0, 0.0, 0.0])
DRONE_GOAL = jnp.zeros(6)                 
DRONE_OBS_C = jnp.array([-0.95, 0.025])   
DRONE_OBS_R = 0.3
DRONE_H, DRONE_DT = 5, 1.0


def _loss_drone(states, controls):
    return 0.5 * jnp.sum((states[:, :2] - DRONE_GOAL[:2]) ** 2) + 0.5e-2 * jnp.sum(controls ** 2)


def _build_drone(use_full_hessian):
    from turbompc.dynamics.drone_dynamics import (
        DroneDynamics, drone_parameters, drone_state_dot_parameters)
    from turbompc.dynamics.integrators import DiscretizationScheme
    dyn = DroneDynamics(drone_parameters)
    N = DRONE_H
    problem_params = dict(load_problem_params("drone.yaml"))
    problem_params["horizon"] = N
    problem_params["discretization_resolution"] = DRONE_DT
    problem_params["discretization_scheme"] = int(DiscretizationScheme.EULER)
    problem_params["constrain_initial_control"] = False
    problem_params["initial_state"] = DRONE_START
    problem_params["initial_guess_final_state"] = DRONE_GOAL
    problem_params["reference_state_trajectory"] = jnp.tile(DRONE_GOAL, (N + 1, 1))
    problem_params["reference_control_trajectory"] = jnp.zeros((N + 1, DRONE_NU))
    problem_params[QK] = jnp.array([5.0, 5.0, 5.0, 0.1, 0.1, 0.1])   
    problem_params[RK] = jnp.array([0.1, 0.1, 0.1])
    problem_params["weights_penalization_final_state"] = jnp.zeros((DRONE_NX,))
    problem_params["obstacles_centers"] = jnp.tile(DRONE_OBS_C[None, None, :], (N + 1, 1, 1))
    problem_params["obstacles_radii"] = jnp.array([DRONE_OBS_R])
    problem_params["obstacles_dimension"] = 2
    problem_params["use_slack_variables"] = False
    problem_params["slack_penalization_weight"] = 0.0
    problem_params["rescale_optimization_variables"] = False
    problem_params["dynamics_state_dot_params"] = {
        k: jnp.tile(jnp.asarray(v, dtype=jnp.float64)[None, :], (N, 1))
        for k, v in drone_state_dot_parameters.items()
    }
    ocp = OptimalControlProblemObstacle(dynamics=dyn, params=problem_params)
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 50
    solver = TurboMPCSolver(
        program=ocp, params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE,
        use_full_hessian=use_full_hessian)
    return solver, problem_params


def test_drone_obstacle_active_and_dynamics_nonlinear():
    """forward convergence check"""
    solver, problem_params = _build_drone(use_full_hessian=True)
    sol = solver.solve(solver.initial_guess(problem_params), problem_params, {k: problem_params[k] for k in WEIGHT_KEYS})
    assert float(sol.convergence_error) < 1e-5
    P = np.asarray(sol.states)[:, :2]
    dist = np.linalg.norm(P - np.asarray(DRONE_OBS_C), axis=1)
    assert dist.min() < DRONE_OBS_R + 0.05, f"obstacle not engaged (min dist {dist.min():.3f})"
    speed = np.linalg.norm(np.asarray(sol.states)[:, 3:6], axis=1)
    assert speed.max() > 1e-2, f"drone not moving -> drag (nonlinear dyn) inactive (vmax {speed.max():.2e})"


def test_drone_without_full_hessian_is_worse():
    """Check full Hessian with finite difference and ablation on the nonlinear drone: Gauss-Newton is worse vs FD than the full Hessian."""
    solver_on, problem_params = _build_drone(use_full_hessian=True)
    solver_off, _ = _build_drone(use_full_hessian=False)
    g_on = _flat(_ad_grad(solver_on, problem_params, loss_fn=_loss_drone))
    g_off = _flat(_ad_grad(solver_off, problem_params, loss_fn=_loss_drone))
    g_fd, flagged = _fd_grad(solver_on, problem_params, loss_fn=_loss_drone)
    assert not flagged
    rel_on, rel_off = _rel(g_on, g_fd), _rel(g_off, g_fd)
    # Compare with finite differences.
    assert rel_on < 5e-3, f"with-Hessian rel_l2={rel_on}"
    # With Hessian gives more accurate gradient.
    assert rel_off > 3 * rel_on, f"ablation not separated: on={rel_on:.2e} off={rel_off:.2e}"
