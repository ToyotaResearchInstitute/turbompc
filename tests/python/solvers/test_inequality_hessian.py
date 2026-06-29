"""Test for inequality-constraint Lagrangian Hessian (mu^T grad^2 g) in the TurboMPC backward.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from tests.helpers.problem_fixtures import make_linear_params
from turbompc.problems.obstacle_avoidance import OptimalControlProblemObstacle
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend
from turbompc.utils.load_params import load_solver_params

NX, NU = 4, 2
START = jnp.array([0.0, 0.0, 0.0, 0.0])
GOAL = jnp.array([2.0, 1.5, 0.0, 0.0])         # off-axis -> no degenerate weight direction
OBS_C = jnp.array([1.0, 0.55])                  # on the start->goal path, firmly in the way
OBS_R = 0.55
HORIZON, DT = 14, 0.18
QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"
WEIGHT_KEYS = [QK, RK]


def _build(use_full_hessian):
    # complete validated base params (LinearDynamics nx=4, nu=2); override to a 2-D double
    # integrator + obstacle. bounded=False -> control bounds inert (only the obstacle binds).
    dyn, pp = make_linear_params(HORIZON, implicit=False, bounded=False, initial_state=START)
    A_c = jnp.array([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
    B_c = jnp.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])
    pp = dict(pp)
    pp["discretization_resolution"] = DT
    pp["initial_state"] = START
    pp["initial_guess_final_state"] = GOAL
    pp["reference_state_trajectory"] = jnp.tile(GOAL, (HORIZON + 1, 1))
    pp[QK] = jnp.array([5.0, 5.0, 0.0, 0.0])
    pp[RK] = jnp.array([0.1, 0.1])
    pp["weights_penalization_final_state"] = jnp.zeros((NX,))
    pp["dynamics_state_dot_params"] = {"A": A_c, "B": B_c, "b": jnp.zeros((NX,))}
    pp["obstacles_centers"] = jnp.tile(OBS_C[None, None, :], (HORIZON + 1, 1, 1))
    pp["obstacles_radii"] = jnp.array([OBS_R])
    pp["obstacles_dimension"] = 2
    pp["rescale_optimization_variables"] = False
    ocp = OptimalControlProblemObstacle(dynamics=dyn, params=pp)
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 50
    solver = TurboMPCSolver(
        program=ocp, params=sp,
        forward_backend=ForwardBackend.ADMM_FUSED_CUDSS,
        backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
        use_full_hessian=use_full_hessian)
    return solver, pp


def _loss(states, controls):
    return 0.5 * jnp.sum((states[:, :2] - GOAL[:2]) ** 2) + 0.5e-2 * jnp.sum(controls ** 2)


def _flat(d):
    return np.concatenate([np.asarray(d[k]).reshape(-1) for k in WEIGHT_KEYS])


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _ad_grad(solver, pp):
    ig = solver.initial_guess(pp)
    def loss_w(w):
        sol = solver.solve(ig, pp, {**{k: pp[k] for k in WEIGHT_KEYS}, **w})
        return _loss(sol.states, sol.controls)
    return jax.grad(loss_w)({k: pp[k] for k in WEIGHT_KEYS})


def _fd_grad(solver, pp, eps_seq=(1e-3, 3e-4, 1e-4), rtol=2e-3, atol=1e-7):
    ig = solver.initial_guess(pp)
    def loss_of_w(w):
        sol = solver.solve(ig, pp, {**{k: pp[k] for k in WEIGHT_KEYS}, **w})
        return float(_loss(sol.states, sol.controls))
    base = {k: np.array(pp[k], float) for k in WEIGHT_KEYS}
    grad, flagged = [], False
    for k in WEIGHT_KEYS:
        gk = np.zeros(base[k].size)
        for i in range(base[k].size):
            vals = []
            for eps in eps_seq:
                ap = base[k].copy().reshape(-1); ap[i] += eps
                am = base[k].copy().reshape(-1); am[i] -= eps
                wp = {k: jnp.asarray(ap.reshape(base[k].shape))}
                wm = {k: jnp.asarray(am.reshape(base[k].shape))}
                vals.append((loss_of_w(wp) - loss_of_w(wm)) / (2 * eps))
            chosen, ok = vals[-1], False
            for a, b in zip(vals[:-1], vals[1:]):
                if abs(a - b) <= rtol * abs(b) + atol:
                    chosen, ok = a, True; break
            flagged = flagged or (not ok); gk[i] = chosen
        grad.append(gk)
    return np.concatenate(grad), flagged


def test_obstacle_active_and_dynamics_linear():
    """Sanity: forward converges and the obstacle is active (so mu, grad^2 g != 0)."""
    solver, pp = _build(use_full_hessian=True)
    ig = solver.initial_guess(pp)
    sol = solver.solve(ig, pp, {k: pp[k] for k in WEIGHT_KEYS})
    assert float(sol.convergence_error) < 1e-5
    P = np.asarray(sol.states)[:, :2]
    dist = np.linalg.norm(P - np.asarray(OBS_C), axis=1)
    assert dist.min() < OBS_R + 0.02, f"obstacle not engaged (min dist {dist.min():.3f})"


def test_inequality_hessian_matches_fd():
    """use_full_hessian=True (inequality Hessian ON) -> AD matches convergence-checked FD."""
    solver, pp = _build(use_full_hessian=True)
    g_ad = _flat(_ad_grad(solver, pp))
    g_fd, flagged = _fd_grad(solver, pp)
    assert not flagged, "FD did not plateau"
    assert _cos(g_ad, g_fd) > 1 - 1e-4, f"AD vs FD cos={_cos(g_ad, g_fd)}"
    assert _rel(g_ad, g_fd) < 5e-3, f"AD vs FD rel_l2={_rel(g_ad, g_fd)}"


def test_without_inequality_hessian_is_worse():
    """Ablation: use_full_hessian=False (Gauss-Newton inequality) is materially worse vs FD."""
    solver_on, pp = _build(use_full_hessian=True)
    solver_off, _ = _build(use_full_hessian=False)
    g_on = _flat(_ad_grad(solver_on, pp))
    g_off = _flat(_ad_grad(solver_off, pp))
    g_fd, flagged = _fd_grad(solver_on, pp)
    assert not flagged
    rel_on, rel_off = _rel(g_on, g_fd), _rel(g_off, g_fd)
    assert rel_on < 5e-3, f"with-Hessian rel_l2={rel_on}"
    assert rel_off > 5 * rel_on, f"ablation not separated: on={rel_on:.2e} off={rel_off:.2e}"
