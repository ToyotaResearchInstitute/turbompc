"""Closed-loop rollout benchmark: linear dynamics (constrained).

Drop-in replacement for benchmarking/linear-system/benchmark_turbompc_constrained.py
using the generalized turbompc.utils.timing framework.

Run from benchmarking/linear-system/:
    python ../../timing_linear.py --batch_size 8 --sim_steps 50 --horizon 40
"""
import argparse
import os
import sys
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

import jax.numpy as jnp

# Allow running from benchmarking/linear-system or its subdirectories.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turbompc.dynamics.linear_dynamics import LinearDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import (
    BACKWARD_BACKEND_CHOICES,
    FORWARD_BACKEND_CHOICES,
    parse_backward_backend,
    parse_forward_backend,
)
from turbompc.utils.load_params import load_solver_params
from turbompc.utils.timing import ProblemConfig, benchmark_rollout
from utils import (  # noqa: E402
    MPC_TOL,
    N_CTRL,
    N_STATE,
    TURBOMPC_SQP_ITER,
    generate_problem_data,
)

# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------


def _make_problem_params(horizon: int, umax: float) -> dict:
    nx, nu = N_STATE, N_CTRL
    return {
        "horizon": horizon,
        "discretization_resolution": 1.0,
        # Euler (0) — continuous A,B,b with dt=1 reproduce the discrete step
        "discretization_scheme": 0,
        "initial_state": jnp.zeros(nx),
        "initial_guess_final_state": jnp.zeros(nx),
        "reference_state_trajectory": jnp.zeros((horizon + 1, nx)),
        "reference_control_trajectory": jnp.zeros((horizon + 1, nu)),
        "penalize_control_reference": False,
        "rescale_optimization_variables": False,
        "constrain_initial_control": False,
        "initial_control": jnp.zeros(nu),
        "state_rescaling_min": -np.linspace(0.1, 5.0, nx),
        "state_rescaling_max": jnp.ones(nx),
        "control_rescaling_min": -np.linspace(0.2, 3.0, nu),
        "control_rescaling_max": jnp.ones(nu),
        "weights_penalization_reference_state_trajectory": jnp.ones(nx),
        "weights_penalization_final_state": jnp.zeros(nx),
        "weights_penalization_control_squared": jnp.ones(nu),
        "weights_penalization_control_rate": jnp.zeros(nu),
        "state_min_bounds": -jnp.ones(nx) * 1e7,
        "state_max_bounds": jnp.ones(nx) * 1e7,
        "control_min_bounds": -jnp.ones(nu) * umax,
        "control_max_bounds": jnp.ones(nu) * umax,
        # placeholder — overwritten each seed
        "dynamics_state_dot_params": {
            "A": jnp.zeros((nx, nx)),
            "B": jnp.zeros((nx, nu)),
            "b": jnp.zeros(nx),
        },
    }


def _make_solver_params(warm_start: bool, alpha: float, pcg_eps: float) -> dict:
    sp = load_solver_params("turbompc.yaml")
    sp["num_sqp_iteration_max"] = TURBOMPC_SQP_ITER
    sp["tol_convergence"] = MPC_TOL
    sp["warm_start_backward"] = warm_start
    sp["linesearch"] = False
    sp["linesearch_alphas"] = [1.0]
    sp["admm"]["max_iter"] = 1000
    sp["admm"]["check_termination_every"] = 1
    sp["admm"]["eps_abs"] = 1e-4
    sp["admm"]["eps_rel"] = 1e-4
    sp["admm"]["relaxation_parameter"] = alpha
    sp["admm"]["pcg"]["tol_epsilon"] = pcg_eps
    return sp


def _reward(state, control):
    return -(jnp.sum(state**2) + jnp.sum(control**2))


def _update_per_seed(seed, batch_size, problem_params):
    Q, R, A, B, b, x0 = generate_problem_data(batch_size, seed)
    A = jnp.array(A)
    B = jnp.array(B)
    b = jnp.array(b)
    x0 = jnp.array(x0)
    Q = jnp.array(Q)
    R = jnp.array(R)
    # discrete -> continuous: A_cont = A_discrete - I
    updates = {
        "weights_penalization_reference_state_trajectory": jnp.diag(Q),
        "weights_penalization_control_squared": jnp.diag(R),
        "weights_penalization_final_state": jnp.zeros(N_STATE),
        "dynamics_state_dot_params": {
            "A": A - jnp.eye(N_STATE),
            "B": B,
            "b": b,
        },
    }
    return updates, x0


def make_linear_config(
    horizon: int, umax: float, warm_start: bool, alpha: float, pcg_eps: float
) -> ProblemConfig:
    dynamics_params = {
        "verbose": False,
        "num_states": N_STATE,
        "num_controls": N_CTRL,
        "names_states": [f"x{i}" for i in range(N_STATE)],
        "names_controls": [f"u{i}" for i in range(N_CTRL)],
    }
    return ProblemConfig(
        dynamics=LinearDynamics(dynamics_params),
        problem_class=OptimalControlProblem,
        problem_params=_make_problem_params(horizon, umax),
        solver_params=_make_solver_params(warm_start, alpha, pcg_eps),
        weight_keys=["weights_penalization_reference_state_trajectory"],
        reward_fn=_reward,
        update_per_seed=_update_per_seed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Linear constrained rollout benchmark")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--sim_steps", type=int, default=50)
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--umax", type=float, default=10.0)
    parser.add_argument("--pcg_eps", type=float, default=1e-12)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument(
        "--linear_fwd_backend",
        type=str,
        default="admm_jax_loop_pcg",
        choices=FORWARD_BACKEND_CHOICES,
    )
    parser.add_argument(
        "--linear_bwd_backend",
        type=str,
        default="admm_jax_loop_pcg",
        choices=BACKWARD_BACKEND_CHOICES,
    )
    parser.add_argument("--cold_start", action="store_true")
    parser.add_argument("--skip_backward", action="store_true")
    parser.add_argument("--use_full_hessian", action="store_true")
    parser.add_argument("--fd_check", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    args = parser.parse_args()

    warm_start = not args.cold_start
    run_backward = not args.skip_backward
    fwd_backend = parse_forward_backend(args.linear_fwd_backend)
    bwd_backend = parse_backward_backend(args.linear_bwd_backend)

    config = make_linear_config(
        horizon=args.horizon,
        umax=args.umax,
        warm_start=warm_start,
        alpha=args.alpha,
        pcg_eps=args.pcg_eps,
    )

    fwd_times, bwd_times, gradients, admm_iters, costs, _ = benchmark_rollout(
        config=config,
        batch_size=args.batch_size,
        num_sim_steps=args.sim_steps,
        num_repeats=args.num_repeats,
        forward_backend=fwd_backend,
        backward_backend=bwd_backend,
        warm_start=warm_start,
        run_backward=run_backward,
        use_full_hessian=args.use_full_hessian,
        fd_check=args.fd_check,
    )

    if args.save_results:
        ws = "ws" if warm_start else "cs"
        tag = (
            f"linear_{args.batch_size}_{args.horizon}_{N_STATE + N_CTRL}"
            f"_{args.num_repeats}_c_{ws}_pcg={args.pcg_eps}_alpha={args.alpha}"
            f"_umax={args.umax}_backend={fwd_backend.name}-{bwd_backend.name}"
        )
        os.makedirs(f"timing_results/{tag}", exist_ok=True)
        np.save(f"timing_results/{tag}/fwd", fwd_times)
        np.save(f"timing_results/{tag}/bwd", bwd_times)
        np.save(f"timing_results/{tag}/iters", admm_iters)
        if gradients:
            np.save(f"timing_results/{tag}/grad", gradients[-1])
        print("Results saved.")


if __name__ == "__main__":
    main()
