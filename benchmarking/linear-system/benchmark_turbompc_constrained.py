"""TurboMPC constrained rollout benchmark."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Tuple

# Allow imports from parent directory (utils, benchmark_naming, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_threefry_partitionable", True)

import jax.numpy as jnp  # noqa: E402
from benchmark_naming import device_name_for_file, turbompc_dirname  # noqa: E402
from benchmark_problem_setup import build_turbompc_linear_problem  # noqa: E402
from turbompc.problems.optimal_control_problem import (  # noqa: E402
    OptimalControlProblem,
)
from turbompc.solvers.turbompc_solver import (  # noqa: E402
    BACKWARD_BACKEND_CHOICES,
    FORWARD_BACKEND_CHOICES,
    parse_backward_backend,
    parse_forward_backend,
)
from turbompc.utils.load_params import load_solver_params  # noqa: E402
from turbompc.utils.timing import ProblemConfig, benchmark_rollout  # noqa: E402
from utils import TURBOMPC_SQP_ITER, generate_problem_data  # noqa: E402


def _reward(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    return -(jnp.sum(state**2) + jnp.sum(control**2))


def _make_config(
    *,
    horizon: int,
    umax: float,
    pcg_eps: float,
    warm_start: bool,
    alpha: float,
    n_state: int,
    n_ctrl: int,
    admm_max_iter: int = 50,
    tol: float = 1e-3,
) -> ProblemConfig:
    dynamics, problem_params = build_turbompc_linear_problem(
        horizon=horizon, umax=umax, n_state=n_state, n_ctrl=n_ctrl
    )

    solver_params = load_solver_params("turbompc.yaml")
    solver_params["num_sqp_iteration_max"] = TURBOMPC_SQP_ITER
    solver_params["tol_convergence"] = tol
    solver_params["warm_start_backward"] = warm_start
    solver_params["linesearch"] = False
    solver_params["linesearch_alphas"] = [1.0]
    solver_params["admm"]["max_iter"] = admm_max_iter
    solver_params["admm"]["check_termination_every"] = 1
    solver_params["admm"]["eps_abs"] = tol
    solver_params["admm"]["eps_rel"] = tol
    solver_params["admm"]["relaxation_parameter"] = alpha
    solver_params["admm"]["pcg"]["tol_epsilon"] = pcg_eps

    weight_keys = [
        "weights_penalization_reference_state_trajectory",
        "weights_penalization_control_squared",
    ]

    def update_per_seed(
        seed: int, batch_size: int, params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], jnp.ndarray]:
        Q, R, A_matrix, B_matrix, b_vector, initial_states = generate_problem_data(
            batch_size, seed, n_state=n_state, n_ctrl=n_ctrl
        )
        Q = jnp.asarray(Q)
        R = jnp.asarray(R)
        A_matrix = jnp.asarray(A_matrix)
        B_matrix = jnp.asarray(B_matrix)
        b_vector = jnp.asarray(b_vector)
        initial_states = jnp.asarray(initial_states)

        updates = {
            "weights_penalization_reference_state_trajectory": jnp.diag(Q),
            "weights_penalization_control_squared": jnp.diag(R),
            "weights_penalization_final_state": jnp.zeros(n_state),
            "dynamics_state_dot_params": {
                "A": A_matrix - jnp.eye(n_state),
                "B": B_matrix,
                "b": b_vector,
            },
        }
        return updates, initial_states

    return ProblemConfig(
        dynamics=dynamics,
        problem_class=OptimalControlProblem,
        problem_params=problem_params,
        solver_params=solver_params,
        weight_keys=weight_keys,
        reward_fn=_reward,
        update_per_seed=update_per_seed,
    )


def main():
    fwd_choices = FORWARD_BACKEND_CHOICES
    bwd_choices = BACKWARD_BACKEND_CHOICES
    parser = argparse.ArgumentParser(
        description="TurboMPC constrained rollout benchmark"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--sim_steps", type=int, default=50)
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--umax", type=float, default=10.0)
    parser.add_argument(
        "--n_state",
        type=int,
        default=8,
        help="State dimension (overrides utils.N_STATE)",
    )
    parser.add_argument(
        "--n_ctrl",
        type=int,
        default=4,
        help="Control dimension (overrides utils.N_CTRL)",
    )
    parser.add_argument("--pcg_eps", type=float, default=1e-12)

    parser.add_argument(
        "--linear_fwd_backend",
        type=str,
        default="admm_jax_loop_pcg",
        choices=fwd_choices,
        help="Forward backend enum name (ForwardBackend).",
    )
    parser.add_argument(
        "--linear_bwd_backend",
        type=str,
        default="admm_jax_loop_pcg",
        choices=bwd_choices,
        help="Backward backend enum name (BackwardBackend).",
    )

    parser.add_argument("--cold_start", action="store_true")
    parser.add_argument("--skip_backward", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--sqp_iter", type=int, default=TURBOMPC_SQP_ITER)
    parser.add_argument(
        "--admm_max_iter", type=int, default=100, help="ADMM max iterations"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-7,
        help="Solver tolerance (SQP convergence + ADMM eps_abs/eps_rel)",
    )
    parser.add_argument("--use_full_hessian", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument(
        "--save_gradients",
        action="store_true",
        help="Also save grad_Q/grad_R .npy for compare_gradients.py",
    )

    args = parser.parse_args()
    warm_start = not args.cold_start
    run_backward = not args.skip_backward
    fwd_backend = parse_forward_backend(args.linear_fwd_backend)
    bwd_backend = parse_backward_backend(args.linear_bwd_backend)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Num repeats: {args.num_repeats}")
    print(
        f"TurboMPC Rollout: {args.n_state} states, {args.n_ctrl} controls,"
        f" {args.horizon} horizon"
    )
    print(
        f"Simulation: {args.sim_steps} steps, {args.batch_size} batch, umax={args.umax}"
    )
    print(f"Settings: SQP+ADMM (PCG tol={args.pcg_eps}, alpha={args.alpha})")

    turbompc_horizon = args.horizon - 1

    config = _make_config(
        horizon=turbompc_horizon,
        umax=args.umax,
        pcg_eps=args.pcg_eps,
        warm_start=warm_start,
        alpha=args.alpha,
        n_state=args.n_state,
        n_ctrl=args.n_ctrl,
        admm_max_iter=args.admm_max_iter,
        tol=args.tol,
    )

    forward_times, backward_times, gradients, admm_iters, costs, _ = benchmark_rollout(
        config=config,
        batch_size=args.batch_size,
        num_sim_steps=args.sim_steps,
        num_repeats=args.num_repeats,
        forward_backend=fwd_backend,
        backward_backend=bwd_backend,
        warm_start=warm_start,
        run_backward=run_backward,
        use_full_hessian=args.use_full_hessian,
        fd_check=False,
        verbose=True,
    )

    def _stack_repeat_gradients(repeat_gradients):
        if not repeat_gradients:
            return None
        keys = repeat_gradients[0].keys()
        return {
            key: np.stack([g[key] for g in repeat_gradients if g is not None])
            for key in keys
        }

    if args.save_results:
        dirname = turbompc_dirname(
            args.batch_size,
            args.horizon,
            args.n_state + args.n_ctrl,
            args.num_repeats,
            constrained=True,
            warm_start=warm_start,
            pcg_eps=args.pcg_eps,
            alpha=args.alpha,
            umax=args.umax,
            tol=args.tol,
            admm_max_iter=args.admm_max_iter,
            sim_steps=args.sim_steps,
        )
        backend_dirname = f"fwd={args.linear_fwd_backend}_bwd={args.linear_bwd_backend}"
        device = device_name_for_file(jnp.zeros(1).device)
        outdir = os.path.join("timing_results", backend_dirname, dirname)
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, f"turbompc_{device}_fwd"), forward_times)
        np.save(os.path.join(outdir, f"turbompc_{device}_bwd"), backward_times)
        np.save(os.path.join(outdir, f"turbompc_{device}_grad"), gradients)
        np.save(os.path.join(outdir, f"turbompc_{device}_iters"), admm_iters)
        if args.save_gradients and gradients is not None:
            stacked_gradients = _stack_repeat_gradients(gradients)
            np.save(
                os.path.join(outdir, f"turbompc_{device}_grad_Q"),
                stacked_gradients["weights_penalization_reference_state_trajectory"],
            )
            np.save(
                os.path.join(outdir, f"turbompc_{device}_grad_R"),
                stacked_gradients["weights_penalization_control_squared"],
            )
        print("Results saved")


if __name__ == "__main__":
    main()
