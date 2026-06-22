"""TurboMPC gradient accuracy benchmark.

For each solver tolerance in a sweep, runs N_SEEDS independent problem instances
(seeds) and measures the cosine similarity between the AD (automatic differentiation)
gradient and the FD (finite-difference) gradient.

Nominal problem settings (matching run_sweeps_oat_turbompc.sh):
    batch=64, horizon=40, n_state=8, n_ctrl=4, umax=1.0
    fwd=admm_cudss_loop, bwd=direct_cudss_ffi, alpha=1.0
    admm_max_iter=1000, sim_steps=50, use_full_hessian=True

Statistics collected per tolerance:
    - cosine similarity per weight key (Q, R) and concatenated
    - relative L2 error ||g_ad - g_fd|| / ||g_fd||
    - absolute L2 error ||g_ad - g_fd||
Aggregated as mean ± std, median, min over seeds.
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_threefry_partitionable", True)

import jax
import jax.numpy as jnp
from jax import checkpoint, jit, vmap

from benchmark_problem_setup import build_turbompc_linear_problem  # noqa: E402
from turbompc.problems.optimal_control_problem import OptimalControlProblem  # noqa: E402
from turbompc.solvers.turbompc_solver import (  # noqa: E402
    TurboMPCSolver,
    parse_backward_backend,
    parse_forward_backend,
)
from turbompc.utils.gradient_finitediff import gradient_finite_diff  # noqa: E402
from turbompc.utils.load_params import load_solver_params  # noqa: E402
from turbompc.utils.timing import ProblemConfig, build_rollout_fn  # noqa: E402
from utils import TURBOMPC_SQP_ITER, generate_problem_data  # noqa: E402

# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two flat arrays."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + eps
    return float(np.dot(a, b) / denom)


def relative_l2_error(g_ad: np.ndarray, g_fd: np.ndarray, eps: float = 1e-12) -> float:
    """||g_ad - g_fd|| / (||g_fd|| + eps)."""
    return float(np.linalg.norm(g_ad - g_fd) / (np.linalg.norm(g_fd) + eps))


def absolute_l2_error(g_ad: np.ndarray, g_fd: np.ndarray) -> float:
    return float(np.linalg.norm(g_ad - g_fd))


# ---------------------------------------------------------------------------
# Per-seed gradient comparison
# ---------------------------------------------------------------------------

def _reward(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    return -(jnp.sum(state**2) + jnp.sum(control**2))


def _build_solver_params(
    base_params: Dict[str, Any],
    *,
    sqp_iter: int,
    tol: float,
    admm_max_iter: int,
    alpha: float,
    pcg_eps: float,
) -> Dict[str, Any]:
    p = copy.deepcopy(base_params)
    p["num_sqp_iteration_max"] = sqp_iter
    p["tol_convergence"] = tol
    p["warm_start_backward"] = False
    p["linesearch"] = False
    p["linesearch_alphas"] = [1.0]
    p["admm"]["max_iter"] = admm_max_iter
    p["admm"]["check_termination_every"] = 1
    p["admm"]["eps_abs"] = tol
    p["admm"]["eps_rel"] = tol
    p["admm"]["relaxation_parameter"] = alpha
    p["admm"]["pcg"]["tol_epsilon"] = pcg_eps
    return p


def _build_problem_params(
    template: Dict[str, Any],
    *,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    A_matrix: jnp.ndarray,
    B_matrix: jnp.ndarray,
    b_vector: jnp.ndarray,
    n_state: int,
) -> Dict[str, Any]:
    p = dict(template)
    p.update({
        "weights_penalization_reference_state_trajectory": jnp.diag(Q),
        "weights_penalization_control_squared": jnp.diag(R),
        "weights_penalization_final_state": jnp.zeros(n_state),
        "dynamics_state_dot_params": {
            "A": A_matrix - jnp.eye(n_state),
            "B": B_matrix,
            "b": b_vector,
        },
    })
    return p


def _build_rollout_batch(
    dynamics,
    problem_params: Dict[str, Any],
    solver_params: Dict[str, Any],
    weight_keys: List[str],
    initial_states: jnp.ndarray,
    fwd_backend,
    bwd_backend,
    use_full_hessian: bool,
    sim_steps: int,
):
    problem = OptimalControlProblem(dynamics=dynamics, params=dict(problem_params))
    solver = TurboMPCSolver(
        program=problem,
        params=solver_params,
        forward_backend=fwd_backend,
        backward_backend=bwd_backend,
        use_full_hessian=use_full_hessian,
    )
    weights = {k: problem_params[k] for k in weight_keys}
    init_solution = solver.solve(
        solver.initial_guess(problem_params),
        problem_params=problem_params,
        weights=weights,
    )
    rollout_fn = build_rollout_fn(
        config=ProblemConfig(
            dynamics=dynamics,
            problem_class=OptimalControlProblem,
            problem_params=problem_params,
            solver_params=solver_params,
            weight_keys=weight_keys,
            reward_fn=_reward,
            update_per_seed=lambda s, b, p: ({}, initial_states),
        ),
        solver=solver,
        problem_params=problem_params,
        init_solution=init_solution,
        warm_start=False,
        num_sim_steps=sim_steps,
    )
    return jit(vmap(rollout_fn, in_axes=(0, None))), solver, weights


def run_gradient_accuracy_sweep(
    *,
    tolerances: List[float],
    n_seeds: int,
    sqp_iter: int,
    batch_size: int,
    horizon: int,
    n_state: int,
    n_ctrl: int,
    umax: float,
    alpha: float,
    admm_max_iter: int,
    sim_steps: int,
    fwd_backend_str: str,
    bwd_backend_str: str,
    pcg_eps: float,
    fd_eps: float,
    use_full_hessian: bool,
    fd_solver_tol: Optional[float] = None,
    fd_admm_max_iter: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run gradient accuracy sweep and return aggregated statistics.

    When ``fd_solver_tol`` is provided the FD reference gradients are computed
    **once per seed** before the tolerance sweep and cached, so the (expensive)
    FD forward passes are not repeated for every tolerance value.

    Returns a dict keyed by tolerance with per-key and combined stats.
    """
    fwd_backend = parse_forward_backend(fwd_backend_str)
    bwd_backend = parse_backward_backend(bwd_backend_str)

    turbompc_horizon = horizon - 1

    results: Dict[float, Dict[str, Any]] = {}

    weight_keys = [
        "weights_penalization_reference_state_trajectory",
        "weights_penalization_control_squared",
    ]

    # Build dynamics and problem template once (independent of tolerance)
    dynamics, problem_params_template = build_turbompc_linear_problem(
        horizon=turbompc_horizon, umax=umax, n_state=n_state, n_ctrl=n_ctrl
    )
    base_solver_params = load_solver_params("turbompc.yaml")

    # ------------------------------------------------------------------
    # Phase 0: pre-compute FD reference gradients (only when fd_solver_tol
    #          is given — avoids repeating FD once per swept tolerance)
    # ------------------------------------------------------------------
    fd_cache: Dict[int, Dict[str, np.ndarray]] = {}
    if fd_solver_tol is not None:
        fd_admm_iter = fd_admm_max_iter if fd_admm_max_iter is not None else admm_max_iter
        solver_params_fd = _build_solver_params(
            base_solver_params,
            sqp_iter=sqp_iter,
            tol=fd_solver_tol,
            admm_max_iter=fd_admm_iter,
            alpha=alpha,
            pcg_eps=pcg_eps,
        )
        print(f"\n{'=' * 70}")
        print(
            f"Pre-computing FD reference gradients at tol={fd_solver_tol:.0e}, "
            f"admm_max_iter={fd_admm_iter}  ({n_seeds} seeds)"
        )
        print(f"{'=' * 70}")
        for seed in range(n_seeds):
            Q, R, A_matrix, B_matrix, b_vector, initial_states = generate_problem_data(
                batch_size, seed, n_state=n_state, n_ctrl=n_ctrl
            )
            Q, R = jnp.asarray(Q), jnp.asarray(R)
            A_matrix, B_matrix, b_vector = (
                jnp.asarray(A_matrix), jnp.asarray(B_matrix), jnp.asarray(b_vector)
            )
            initial_states = jnp.asarray(initial_states)

            problem_params = _build_problem_params(
                problem_params_template,
                Q=Q, R=R, A_matrix=A_matrix, B_matrix=B_matrix,
                b_vector=b_vector, n_state=n_state,
            )
            rollout_batch_fd, _, weights = _build_rollout_batch(
                dynamics, problem_params, solver_params_fd, weight_keys,
                initial_states, fwd_backend, bwd_backend, use_full_hessian, sim_steps,
            )
            # Warm-up
            _w = rollout_batch_fd(jnp.zeros_like(initial_states), weights)
            jax.block_until_ready(_w)

            g_fd_np = gradient_finite_diff(
                rollout_batch_fd, initial_states, weights=weights, eps=fd_eps
            )
            fd_cache[seed] = g_fd_np

            if verbose:
                print(f"  FD seed={seed:3d}  done")

            del rollout_batch_fd, _w, g_fd_np
            jax.clear_caches()
            gc.collect()

    # ------------------------------------------------------------------
    # Phase 1: sweep AD tolerances
    # ------------------------------------------------------------------
    for tol in tolerances:
        print(f"\n{'=' * 70}")
        print(
            f"Tolerance = {tol:.0e}  |  batch={batch_size}, horizon={horizon}, "
            f"nx={n_state}, nu={n_ctrl}, umax={umax}, seeds={n_seeds}"
        )
        print(f"{'=' * 70}")

        solver_params = _build_solver_params(
            base_solver_params,
            sqp_iter=sqp_iter,
            tol=tol,
            admm_max_iter=admm_max_iter,
            alpha=alpha,
            pcg_eps=pcg_eps,
        )

        # Per-seed stats containers
        seed_stats: Dict[str, List[float]] = {
            "cos_Q": [], "cos_R": [], "cos_all": [],
            "rel_l2_Q": [], "rel_l2_R": [], "rel_l2_all": [],
            "abs_l2_Q": [], "abs_l2_R": [], "abs_l2_all": [],
        }

        for seed in range(n_seeds):
            Q, R, A_matrix, B_matrix, b_vector, initial_states = generate_problem_data(
                batch_size, seed, n_state=n_state, n_ctrl=n_ctrl
            )
            Q, R = jnp.asarray(Q), jnp.asarray(R)
            A_matrix, B_matrix, b_vector = (
                jnp.asarray(A_matrix), jnp.asarray(B_matrix), jnp.asarray(b_vector)
            )
            initial_states = jnp.asarray(initial_states)

            problem_params = _build_problem_params(
                problem_params_template,
                Q=Q, R=R, A_matrix=A_matrix, B_matrix=B_matrix,
                b_vector=b_vector, n_state=n_state,
            )

            rollout_batch, _solver, weights = _build_rollout_batch(
                dynamics, problem_params, solver_params, weight_keys,
                initial_states, fwd_backend, bwd_backend, use_full_hessian, sim_steps,
            )

            def rollout_for_grad(w):
                costs, _ = rollout_batch(initial_states, w)
                return jnp.sum(costs)

            rollout_grad = jit(jax.grad(checkpoint(rollout_for_grad)))

            # JIT warm-up
            _warmup = rollout_batch(jnp.zeros_like(initial_states), weights)
            jax.block_until_ready(_warmup)

            # AD gradients
            g_ad = rollout_grad(weights)
            jax.block_until_ready(g_ad)
            g_ad_np = {k: np.array(g_ad[k]) for k in weight_keys}

            # FD gradients: use cache if available, otherwise compute fresh
            if fd_cache:
                g_fd_np = fd_cache[seed]
            else:
                g_fd_np = gradient_finite_diff(
                    rollout_batch, initial_states, weights=weights, eps=fd_eps
                )

            # ---- Compute metrics per weight key ----
            g_ad_Q = g_ad_np["weights_penalization_reference_state_trajectory"]
            g_fd_Q = g_fd_np["weights_penalization_reference_state_trajectory"]
            g_ad_R = g_ad_np["weights_penalization_control_squared"]
            g_fd_R = g_fd_np["weights_penalization_control_squared"]

            g_ad_all = np.concatenate([g_ad_Q.ravel(), g_ad_R.ravel()])
            g_fd_all = np.concatenate([g_fd_Q.ravel(), g_fd_R.ravel()])

            cos_Q = cosine_similarity(g_ad_Q, g_fd_Q)
            cos_R = cosine_similarity(g_ad_R, g_fd_R)
            cos_all = cosine_similarity(g_ad_all, g_fd_all)
            rel_Q = relative_l2_error(g_ad_Q, g_fd_Q)
            rel_R = relative_l2_error(g_ad_R, g_fd_R)
            rel_all = relative_l2_error(g_ad_all, g_fd_all)
            abs_Q = absolute_l2_error(g_ad_Q, g_fd_Q)
            abs_R = absolute_l2_error(g_ad_R, g_fd_R)
            abs_all = absolute_l2_error(g_ad_all, g_fd_all)

            seed_stats["cos_Q"].append(cos_Q)
            seed_stats["cos_R"].append(cos_R)
            seed_stats["cos_all"].append(cos_all)
            seed_stats["rel_l2_Q"].append(rel_Q)
            seed_stats["rel_l2_R"].append(rel_R)
            seed_stats["rel_l2_all"].append(rel_all)
            seed_stats["abs_l2_Q"].append(abs_Q)
            seed_stats["abs_l2_R"].append(abs_R)
            seed_stats["abs_l2_all"].append(abs_all)

            if verbose:
                print(
                    f"  seed={seed:3d}  "
                    f"cos(Q)={cos_Q:.4f}  cos(R)={cos_R:.4f}  cos(all)={cos_all:.4f}  "
                    f"rel_l2(all)={rel_all:.2e}"
                )

            del rollout_batch, rollout_for_grad, rollout_grad, _warmup
            del g_ad, g_ad_np
            if not fd_cache:
                del g_fd_np
            jax.clear_caches()
            gc.collect()

        # ---- Aggregate statistics ----
        agg: Dict[str, Any] = {"n_seeds": n_seeds}
        for metric, vals in seed_stats.items():
            arr = np.array(vals)
            agg[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "values": arr,
            }

        results[tol] = agg

        # Pretty-print summary for this tolerance
        print(f"\n  --- Summary for tol={tol:.0e} (n={n_seeds} seeds) ---")
        print(
            f"  {'Metric':<20}  {'mean':>8}  {'std':>8}  {'median':>8}  "
            f"{'min':>8}  {'max':>8}"
        )
        for metric in ["cos_Q", "cos_R", "cos_all", "rel_l2_Q", "rel_l2_R", "rel_l2_all"]:
            s = agg[metric]
            print(
                f"  {metric:<20}  {s['mean']:8.4f}  {s['std']:8.4f}  "
                f"{s['median']:8.4f}  {s['min']:8.4f}  {s['max']:8.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_results(
    results: Dict[float, Any],
    outdir: str,
    args: argparse.Namespace,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Save full results as .npz per tolerance
    for tol, agg in results.items():
        tol_str = f"{tol:.0e}".replace("-", "n")
        fname = os.path.join(outdir, f"tol_{tol_str}.npz")
        save_dict: Dict[str, Any] = {"n_seeds": agg["n_seeds"]}
        for metric, stats in agg.items():
            if isinstance(stats, dict) and "values" in stats:
                save_dict[metric] = stats["values"]
                save_dict[f"{metric}_mean"] = stats["mean"]
                save_dict[f"{metric}_std"] = stats["std"]
                save_dict[f"{metric}_median"] = stats["median"]
                save_dict[f"{metric}_min"] = stats["min"]
                save_dict[f"{metric}_max"] = stats["max"]
        np.savez(fname, **save_dict)

    # Save a compact CSV summary table
    csv_path = os.path.join(outdir, "summary.csv")
    header = (
        "tol,"
        "cos_all_mean,cos_all_std,cos_all_median,cos_all_min,"
        "cos_Q_mean,cos_Q_std,"
        "cos_R_mean,cos_R_std,"
        "rel_l2_all_mean,rel_l2_all_std,"
        "rel_l2_Q_mean,rel_l2_R_mean,"
        "n_seeds"
    )
    rows = [header]
    for tol in sorted(results.keys()):
        agg = results[tol]
        row = ",".join([
            f"{tol:.0e}",
            f"{agg['cos_all']['mean']:.6f}",
            f"{agg['cos_all']['std']:.6f}",
            f"{agg['cos_all']['median']:.6f}",
            f"{agg['cos_all']['min']:.6f}",
            f"{agg['cos_Q']['mean']:.6f}",
            f"{agg['cos_Q']['std']:.6f}",
            f"{agg['cos_R']['mean']:.6f}",
            f"{agg['cos_R']['std']:.6f}",
            f"{agg['rel_l2_all']['mean']:.6e}",
            f"{agg['rel_l2_all']['std']:.6e}",
            f"{agg['rel_l2_Q']['mean']:.6e}",
            f"{agg['rel_l2_R']['mean']:.6e}",
            str(agg["n_seeds"]),
        ])
        rows.append(row)
    with open(csv_path, "w") as f:
        f.write("\n".join(rows) + "\n")

    print(f"\nResults saved to {outdir}/")
    print(f"  Per-tolerance .npz files + summary.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TurboMPC gradient accuracy sweep (AD vs FD cosine similarity)"
    )
    # Nominal problem settings
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--sim_steps", type=int, default=50)
    parser.add_argument("--n_state", type=int, default=8)
    parser.add_argument("--n_ctrl", type=int, default=4)
    parser.add_argument("--umax", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--sqp_iter", type=int, default=TURBOMPC_SQP_ITER)
    parser.add_argument("--admm_max_iter", type=int, default=50)
    parser.add_argument("--pcg_eps", type=float, default=1e-12)

    # Accuracy sweep settings
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=[1e-1, 1e-3, 1e-5, 1e-7, 1e-9],
        help="Solver tolerances to sweep (space-separated list)",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=100,
        help="Number of problem instances (seeds) per tolerance",
    )
    parser.add_argument(
        "--fd_eps",
        type=float,
        default=1e-5,
        help="Finite-difference step size",
    )
    parser.add_argument(
        "--fd_solver_tol",
        type=float,
        default=None,
        help=(
            "Optional solver tolerance used only for FD evaluations. "
            "If omitted, uses each swept tolerance."
        ),
    )
    parser.add_argument(
        "--fd_admm_max_iter",
        type=int,
        default=None,
        help=(
            "Optional ADMM max_iter used only for FD evaluations. "
            "If omitted, uses --admm_max_iter."
        ),
    )

    # Backend settings (nominal)
    parser.add_argument(
        "--linear_fwd_backend",
        type=str,
        default="admm_cudss_loop",
    )
    parser.add_argument(
        "--linear_bwd_backend",
        type=str,
        default="direct_cudss_ffi",
    )
    parser.add_argument("--use_full_hessian", action="store_true", default=True)

    # Output
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--outdir", type=str, default="timing_results/gradient_accuracy")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-seed output")

    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("TurboMPC Gradient Accuracy Sweep")
    print(f"  batch={args.batch_size}, horizon={args.horizon}, "
          f"nx={args.n_state}, nu={args.n_ctrl}, umax={args.umax}")
    print(f"  sim_steps={args.sim_steps}, alpha={args.alpha}, "
            f"sqp_iter={args.sqp_iter}, admm_max_iter={args.admm_max_iter}")
    print(f"  fwd={args.linear_fwd_backend}, bwd={args.linear_bwd_backend}")

    print(f"  tolerances={args.tolerances}")
    print(f"  n_seeds={args.n_seeds}, fd_eps={args.fd_eps}")
    if args.fd_solver_tol is not None or args.fd_admm_max_iter is not None:
        print(
            f"  fd_solver_tol={args.fd_solver_tol}, "
            f"fd_admm_max_iter={args.fd_admm_max_iter}"
        )
    print(f"{'=' * 70}")

    results = run_gradient_accuracy_sweep(
        tolerances=args.tolerances,
        n_seeds=args.n_seeds,
        sqp_iter=args.sqp_iter,
        batch_size=args.batch_size,
        horizon=args.horizon,
        n_state=args.n_state,
        n_ctrl=args.n_ctrl,
        umax=args.umax,
        alpha=args.alpha,
        admm_max_iter=args.admm_max_iter,
        sim_steps=args.sim_steps,
        fwd_backend_str=args.linear_fwd_backend,
        bwd_backend_str=args.linear_bwd_backend,
        pcg_eps=args.pcg_eps,
        fd_eps=args.fd_eps,
        use_full_hessian=args.use_full_hessian,
        fd_solver_tol=args.fd_solver_tol,
        fd_admm_max_iter=args.fd_admm_max_iter,
        verbose=not args.quiet,
    )

    # Final summary table across all tolerances
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY  (cosine similarity AD vs FD, concatenated Q+R gradient)")
    print(f"{'=' * 70}")
    print(
        f"  {'tol':>10}  {'cos(all) mean':>14}  {'cos(all) std':>12}  "
        f"{'rel_l2 mean':>12}  {'rel_l2 std':>12}"
    )
    for tol in sorted(results.keys()):
        agg = results[tol]
        print(
            f"  {tol:10.0e}  {agg['cos_all']['mean']:14.6f}  "
            f"{agg['cos_all']['std']:12.6f}  "
            f"{agg['rel_l2_all']['mean']:12.2e}  "
            f"{agg['rel_l2_all']['std']:12.2e}"
        )

    if args.save_results:
        # Include key settings in output dir name
        fd_tag = (
            f"_fdref{args.fd_solver_tol:.0e}"
            if args.fd_solver_tol is not None
            else ""
        )
        subdir = (
            f"fwd={args.linear_fwd_backend}_bwd={args.linear_bwd_backend}/"
            f"b{args.batch_size}_h{args.horizon}_nx{args.n_state}_nu{args.n_ctrl}_"
            f"umax{args.umax}_steps{args.sim_steps}_seeds{args.n_seeds}"
            f"_scp{args.sqp_iter}{fd_tag}"
        )
        outdir = os.path.join(args.outdir, subdir)
        save_results(results, outdir, args)


if __name__ == "__main__":
    main()
