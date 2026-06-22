"""APG training — 3D point-mass spacecraft tracking a periodic zig-zag.

Policy: two-layer MLP (6 -> 8 -> 81, zero-init output layer) that maps the
full state s = [pos; vel] to the MPC's position-tracking, velocity-damping,
and control-effort cost weights. Trained by analytic policy gradient (APG)
with gradients flowing through the differentiable solver (SQP-ADMM,
cuDSS-loop forward + DIRECT_CUDSS_FFI backward).

  - log_w[0:3] -> pos_w mult,  log_w[3:6] -> vel_w mult,  log_w[6:9] -> ctrl_w mult
  - lr=3e-3, batch=4, 500 Adam steps, JAX x64.
  - Loss: post-control state vs phase-aligned reference + light ctrl-squared.
  - Init distribution (#3): random phase i0 in one reference cycle, bounded p/v jitter.

Outputs:
  examples/pointmass_rl/outputs/pointmass_apg_500.csv          per-step train loss + eval RMSE
  examples/pointmass_rl/outputs/pointmass_apg_500_theta.npz    trained policy weights
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT))

from jax import config as _jax_cfg

_jax_cfg.update("jax_enable_x64", True)

import time

import jax
import jax.numpy as jnp
import numpy as np
from examples.pointmass_rl.optimizer import adam_init, adam_step
from examples.pointmass_rl.policy import init_policy, policy_apply
from tests.helpers.solver_fixtures import turbompc_solver_params
from turbompc.dynamics.linear_dynamics import LinearDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)

# ============================================================
# 1. plant + OCP (identical)
# ============================================================
NX, NU = 6, 3
DT, MASS, F_BOUND, H_MPC = 0.1, 1.0, 5.0, 8

A_dyn = jnp.zeros((NX, NX)).at[0:3, 3:6].set(jnp.eye(3))
B_dyn = jnp.zeros((NX, NU)).at[3:6, :].set(jnp.eye(3) / MASS)
dynamics = LinearDynamics(
    {
        "num_states": NX,
        "num_controls": NU,
        "names_states": ["x", "y", "z", "vx", "vy", "vz"],
        "names_controls": ["fx", "fy", "fz"],
    }
)


def make_pointmass_params(mpc_horizon: int) -> dict:
    return {
        "horizon": mpc_horizon,
        "discretization_resolution": DT,
        "discretization_scheme": 0,
        "initial_state": jnp.zeros(NX),
        "initial_guess_final_state": jnp.zeros(NX),
        "reference_state_trajectory": jnp.zeros((mpc_horizon + 1, NX)),
        "reference_control_trajectory": jnp.zeros((mpc_horizon + 1, NU)),
        "penalize_control_reference": False,
        "constrain_initial_control": False,
        "initial_control": jnp.zeros(NU),
        "rescale_optimization_variables": False,
        "state_rescaling_min": -jnp.ones(NX),
        "state_rescaling_max": jnp.ones(NX),
        "control_rescaling_min": -jnp.ones(NU),
        "control_rescaling_max": jnp.ones(NU),
        "weights_penalization_reference_state_trajectory": jnp.ones(NX),
        "weights_penalization_final_state": jnp.zeros(NX),
        "weights_penalization_control_squared": jnp.ones(NU),
        "weights_penalization_control_rate": jnp.zeros(NU),
        "state_min_bounds": -jnp.ones(NX) * 5.0,
        "state_max_bounds": jnp.ones(NX) * 5.0,
        "control_min_bounds": -jnp.ones(NU) * F_BOUND,
        "control_max_bounds": jnp.ones(NU) * F_BOUND,
        "use_slack_variables": False,
        "slack_penalization_weight": 0.0,
        "dynamics_state_dot_params": {
            "A": A_dyn,
            "B": B_dyn,
            "b": jnp.zeros((mpc_horizon + 1, NX)),
        },
    }


mpc_params = make_pointmass_params(H_MPC)
ocp = OptimalControlProblem(dynamics=dynamics, params=mpc_params)
sp = turbompc_solver_params(tol=1e-8, admm_max=500)
sp["num_sqp_iteration_max"] = 10
sp["admm"]["pcg"]["max_iter"] = 500
sp["admm"]["pcg"]["tol_epsilon"] = 1e-12

solver = TurboMPCSolver(
    ocp,
    params=sp,
    forward_backend=ForwardBackend.ADMM_FUSED_CUDSS,
    backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
)


# ============================================================
# 2. back-and-forth zig-zag reference
# ============================================================
A_X = 0.1
SEGMENT_STEPS = 4
WAYPOINTS = jnp.array(
    [
        [+A_X, 0.00, 0.00],
        [-A_X, 0.05, 0.05],
        [+A_X, 0.10, 0.10],
        [-A_X, 0.15, 0.15],
        [+A_X, 0.10, 0.10],
        [-A_X, 0.05, 0.05],
    ]
)
REFERENCE_CYCLE = WAYPOINTS[jnp.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1])]
CYCLE_SEGMENTS = REFERENCE_CYCLE.shape[0]


def pos_target_at_step(i):
    seg = (i // SEGMENT_STEPS) % CYCLE_SEGMENTS
    return REFERENCE_CYCLE[seg]


# ============================================================
# 3. policy -> time-varying MPC weights (full state input, per-stage outputs)
# ============================================================
DEFAULT_POS_W = 1.0
DEFAULT_VEL_W = 1.0
DEFAULT_CTRL_W = 1e-3
H_STAGES = H_MPC + 1  # 9 weight rows: stages 0..H_MPC


def theta_to_weights(theta, state_obs):
    """Map full state s ∈ R^6 → time-varying MPC cost weights.

    NN output: (H_STAGES * 9,) log-multipliers, reshaped to (H_STAGES, 9).
    Per stage t: log_w[t, 0:3] → pos_w, log_w[t, 3:6] → vel_w, log_w[t, 6:9] → ctrl_w.
    Zero-init W2 ⇒ log_w = 0 ⇒ multipliers = 1 ⇒ MPC uses prior weights (constant).
    """
    log_w = policy_apply(theta, state_obs).reshape(H_STAGES, 9)  # (9, 9)
    pos_w = DEFAULT_POS_W * jnp.exp(log_w[:, 0:3])  # (9, 3)
    vel_w = DEFAULT_VEL_W * jnp.exp(log_w[:, 3:6])  # (9, 3)
    ctrl_w = DEFAULT_CTRL_W * jnp.exp(log_w[:, 6:9])  # (9, 3)
    state_w = jnp.concatenate([pos_w, vel_w], axis=-1)  # (9, 6)
    return {
        "weights_penalization_reference_state_trajectory": state_w,
        "weights_penalization_control_squared": ctrl_w,
    }


# ============================================================
# 4. rollout + loss (identical structure)
# ============================================================
H = 8
CYCLE_STEPS = CYCLE_SEGMENTS * SEGMENT_STEPS
N_ROLL = 2 * CYCLE_STEPS


def rollout(theta, s0, i0, n_steps):
    def step(s, k):
        gphase = i0 + k
        future_idx = gphase + jnp.arange(H_MPC + 1)
        future_pos = jax.vmap(pos_target_at_step)(future_idx)
        ref_state = jnp.concatenate([future_pos, jnp.zeros_like(future_pos)], axis=-1)
        weights = theta_to_weights(theta, s)  # CHANGED: full state, not s[:3]
        pp = {**mpc_params, "initial_state": s, "reference_state_trajectory": ref_state}
        sol = solver.solve(solver.initial_guess(pp), problem_params=pp, weights=weights)
        u = sol.controls[0]
        s_next = s + DT * (A_dyn @ s + B_dyn @ u)
        return s_next, (s, u)

    final, (states, controls) = jax.lax.scan(step, s0, xs=jnp.arange(n_steps))
    return jnp.concatenate([states, final[None]], axis=0), controls


def apg_loss(theta, s0, i0):
    states, controls = rollout(theta, s0, i0, H)
    pos = states[1:, :3]
    tgt = jax.vmap(pos_target_at_step)(i0 + 1 + jnp.arange(H))
    track = jnp.sum((pos - tgt) ** 2, axis=-1).mean()
    ctrl = 0.001 * jnp.sum(controls**2, axis=-1).mean()
    return track + ctrl


def sample_init(rng):
    r_i, r_p, r_v = jax.random.split(rng, 3)
    i0 = jax.random.randint(r_i, (), 0, CYCLE_STEPS)
    p = pos_target_at_step(i0) + jax.random.uniform(
        r_p, (3,), minval=-0.03, maxval=0.03
    )
    v = jax.random.uniform(r_v, (3,), minval=-0.05, maxval=0.05)
    return jnp.concatenate([p, v]), i0


def batched_loss(theta, rng):
    rngs = jax.random.split(rng, 4)
    s0s, i0s = jax.vmap(sample_init)(rngs)
    return jax.vmap(lambda s, i: apg_loss(theta, s, i))(s0s, i0s).mean()


def rmse_2cyc(theta):
    s0 = jnp.concatenate([pos_target_at_step(0), jnp.zeros(3)])
    states, _ = rollout(theta, s0, 0, N_ROLL)
    pos = states[1:, :3]
    tgt = jax.vmap(pos_target_at_step)(1 + jnp.arange(N_ROLL))
    return float(jnp.sqrt(jnp.mean(jnp.sum((pos - tgt) ** 2, axis=-1))))


# ============================================================
# 5. training
# ============================================================
N_STEPS, LR, EVAL_EVERY = 500, 3e-3, 10
LOG_PATH = OUTPUT_DIR / "pointmass_apg_500.csv"
THETA_PATH = OUTPUT_DIR / "pointmass_apg_500_theta.npz"


adam_step = jax.jit(adam_step)


def main() -> None:
    theta = init_policy(jax.random.PRNGKey(0), obs_dim=6, out_dim=H_STAGES * 9)
    opt_state = adam_init(theta)
    loss_and_grad = jax.value_and_grad(batched_loss)
    loss_and_grad = jax.jit(loss_and_grad)

    print(
        "=== APG training: 3D point-mass spacecraft (full-state, time-varying MPC"
        " weights) ==="
    )
    print(f"  H_env={H}  H_mpc={H_MPC}  batch=4  lr={LR}  steps={N_STEPS}")
    print(f"  policy: 6 -> 8 -> {H_STAGES * 9}   input: s = [pos; vel] in R^6")
    print("  outputs: per-stage log multipliers on (pos, vel, ctrl) cost weights")
    print(f"           reshaped to ({H_STAGES}, 9) — one row per MPC stage")
    print(f"  log: {LOG_PATH.name}\n")
    print("step    train_loss      eval_RMSE_2cyc(m)   grad_norm     time")

    t0 = time.time()
    rmse0 = rmse_2cyc(theta)
    t1 = time.time()
    print(f"   0       —          {rmse0:.4e}            —      {t1-t0:.1f}s  (init eval)")

    with open(LOG_PATH, "w") as f:
        f.write("step,train_loss,eval_rmse,grad_norm,step_time\n")
        f.write(f"0,,{rmse0:.6e},,\n")

    t_wall0 = time.time()
    rng_data = jax.random.PRNGKey(42)
    last_rmse = rmse0
    for step in range(N_STEPS):
        rng_data, sub = jax.random.split(rng_data)
        t0 = time.time()
        L, g = loss_and_grad(theta, sub)
        theta, opt_state = adam_step(theta, g, opt_state, lr=LR)
        jax.block_until_ready((L, g, theta, opt_state))
        t1 = time.time()
        gnorm = float(jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(g))))
        step_time = t1 - t0
        if (step + 1) % EVAL_EVERY == 0 or step == 0:
            rmse_t = rmse_2cyc(theta)
            last_rmse = rmse_t
            print(
                (
                    f"  {step+1:>3}   {float(L):.4e}     {rmse_t:.4e}      {gnorm:.3e}    "
                    f" {step_time:.3f}s"
                ),
                flush=True,
            )
            with open(LOG_PATH, "a") as f:
                f.write(
                    f"{step+1},{float(L):.6e},{rmse_t:.6e},{gnorm:.6e},{step_time:.3f}\n"
                )
        else:
            with open(LOG_PATH, "a") as f:
                f.write(f"{step+1},{float(L):.6e},,{gnorm:.6e},{step_time:.3f}\n")

    np.savez(
        THETA_PATH,
        **{f"leaf_{i}": np.asarray(v) for i, v in enumerate(jax.tree.leaves(theta))},
    )

    t_wall = time.time() - t_wall0
    print(f"\nTotal wall-clock: {t_wall:.0f}s ({t_wall/60:.1f} min)")
    print(f"Final eval RMSE (2cyc):  {last_rmse:.4e} m")
    print(f"Initial eval RMSE (2cyc): {rmse0:.4e} m")
    print(f"Improvement:     {(rmse0 - last_rmse) / rmse0 * 100:.1f}%")
    print(f"saved theta -> {THETA_PATH.name}")


if __name__ == "__main__":
    main()
