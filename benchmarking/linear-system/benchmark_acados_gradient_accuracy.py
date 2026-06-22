"""acados gradient accuracy benchmark.

For each solver tolerance in a sweep, runs N_SEEDS independent problem instances
(seeds) and measures the cosine similarity between the acados gradient and the
FD (finite-difference) gradient.

Statistics collected per tolerance:
    - cosine similarity per weight key (Q, R) and concatenated
    - relative L2 error ||g_ad - g_fd|| / ||g_fd||
    - absolute L2 error ||g_ad - g_fd||
Aggregated as mean +/- std, median, min over seeds.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import scipy.linalg

from benchmark_acados_constrained import (
    ControlBoundedLqrProblem,
    ControlBounds,
    LinearDiscreteDynamics,
    QuadraticCost,
    create_acados_solver,
    get_num_threads_from_multiprocessing,
)
from utils import generate_problem_data


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


def gradient_finite_diff(
    differentiable_function: Callable[..., np.ndarray | Tuple[np.ndarray, Any]],
    *const_args: Any,
    weights: Dict[str, np.ndarray],
    eps: float = 1e-5,
) -> Dict[str, np.ndarray]:
    """Central-difference gradient estimate w.r.t. weights."""
    grad: Dict[str, np.ndarray] = {}
    for key in weights:
        arr = np.array(weights[key])
        grad_key = np.zeros_like(arr)
        for i in range(grad_key.size):
            idx = np.unravel_index(i, grad_key.shape)
            arr_plus = arr.copy()
            arr_plus[idx] += eps
            arr_minus = arr.copy()
            arr_minus[idx] -= eps
            out_plus = differentiable_function(*const_args, {**weights, key: arr_plus})
            out_minus = differentiable_function(*const_args, {**weights, key: arr_minus})
            val_plus = (out_plus[0] if isinstance(out_plus, tuple) else out_plus).sum().item()
            val_minus = (out_minus[0] if isinstance(out_minus, tuple) else out_minus).sum().item()
            grad_key[idx] = (val_plus - val_minus) / (2 * eps)
        grad[key] = grad_key
    return grad


# ---------------------------------------------------------------------------
# Per-seed gradient comparison
# ---------------------------------------------------------------------------


def _build_seed_problem(
    *,
    batch_size: int,
    seed: int,
    n_state: int,
    n_ctrl: int,
    umax: float,
    horizon: int,
):
    Q, R, A_base, B_base, b_base, x0 = generate_problem_data(
        batch_size, seed, n_state=n_state, n_ctrl=n_ctrl
    )

    quadratic_cost = QuadraticCost(Q, R, np.zeros(n_state), np.zeros(n_ctrl))
    dynamics = LinearDiscreteDynamics(A_base, B_base, b_base)
    u_lower = -umax * np.ones((n_ctrl,), dtype=np.float64)
    u_upper = umax * np.ones((n_ctrl,), dtype=np.float64)
    control_bounds = ControlBounds(u_lower, u_upper)

    problem = ControlBoundedLqrProblem(
        dynamics=dynamics,
        cost=quadratic_cost,
        control_bounds=control_bounds,
        N_horizon=horizon - 1,
    )

    return problem, x0.astype(np.float64), np.asarray(Q), np.asarray(R)


def run_gradient_accuracy_sweep(
    *,
    tolerances: List[float],
    n_seeds: int,
    batch_size: int,
    horizon: int,
    n_state: int,
    n_ctrl: int,
    umax: float,
    sim_steps: int,
    fd_eps: float,
    verbose: bool = True,
) -> Dict[float, Dict[str, Any]]:
    """Run gradient accuracy sweep and return aggregated statistics."""
    results: Dict[float, Dict[str, Any]] = {}
    num_threads = get_num_threads_from_multiprocessing()

    weight_keys = [
        "weights_penalization_reference_state_trajectory",
        "weights_penalization_control_squared",
    ]

    for tol in tolerances:
        print(f"\n{'=' * 70}")
        print(
            f"Tolerance = {tol:.0e}  |  batch={batch_size}, horizon={horizon}, "
            f"nx={n_state}, nu={n_ctrl}, umax={umax}, seeds={n_seeds}"
        )
        print(f"{'=' * 70}")

        seed_stats: Dict[str, List[float]] = {
            "cos_Q": [],
            "cos_R": [],
            "cos_all": [],
            "rel_l2_Q": [],
            "rel_l2_R": [],
            "rel_l2_all": [],
            "abs_l2_Q": [],
            "abs_l2_R": [],
            "abs_l2_all": [],
        }

        for seed in range(n_seeds):
            problem, x0_batch, Q, R = _build_seed_problem(
                batch_size=batch_size,
                seed=seed,
                n_state=n_state,
                n_ctrl=n_ctrl,
                umax=umax,
                horizon=horizon,
            )

            solver, _ = create_acados_solver(
                problem,
                x0_batch,
                seed=seed,
                n_batch=batch_size,
                num_threads=num_threads,
                compute_gradient=True,
                tol=tol,
            )

            p_global_base = np.concatenate(
                (
                    problem.dynamics.A.flatten(order="F"),
                    problem.dynamics.B.flatten(order="F"),
                    problem.dynamics.b,
                )
            )

            def build_p_global(Q_mat: np.ndarray, R_mat: np.ndarray) -> np.ndarray:
                H_mat_val = scipy.linalg.block_diag(Q_mat, R_mat)
                return np.concatenate((p_global_base, H_mat_val.flatten(order="F")))

            def prepare_batch_step(
                current_states: np.ndarray, p_global_values: np.ndarray
            ) -> None:
                for j in range(batch_size):
                    solver.ocp_solvers[j].set(0, "lbx", current_states[j, :])
                    solver.ocp_solvers[j].set(0, "ubx", current_states[j, :])
                    solver.ocp_solvers[j].set_p_global_and_precompute_dependencies(
                        p_global_values
                    )

            def advance_dynamics(
                current_states: np.ndarray, u_applied: np.ndarray
            ) -> np.ndarray:
                return (
                    problem.dynamics.A @ current_states.T
                    + problem.dynamics.B @ u_applied.T
                    + problem.dynamics.b.reshape(-1, 1)
                ).T

            def solve_batch_step(
                current_states: np.ndarray, p_global_values: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
                prepare_batch_step(current_states, p_global_values)
                solver.solve()
                u_sol = solver.get_flat("u").reshape(
                    (batch_size, horizon - 1, problem.nu)
                )
                u_applied = u_sol[:, 0, :]
                new_states = advance_dynamics(current_states, u_applied)
                return u_applied, new_states

            def rollout_with_costs(
                x_init: np.ndarray, Q_mat: np.ndarray, R_mat: np.ndarray
            ) -> float:
                p_global_updated = build_p_global(Q_mat, R_mat)
                current_states = x_init.copy()
                total_cost = 0.0

                for _ in range(sim_steps):
                    u_applied, new_states = solve_batch_step(
                        current_states, p_global_updated
                    )
                    current_states = new_states
                    total_cost += np.sum(new_states**2) + np.sum(u_applied**2)

                return float(total_cost)

            # Gradient computation setup
            nxnu = problem.nx + problem.nu
            h_offset = problem.nx**2 + problem.nx * problem.nu + problem.nx
            q_diag_idx = h_offset + np.array([k * (nxnu + 1) for k in range(problem.nx)])
            r_diag_idx = h_offset + np.array(
                [(problem.nx + k) * (nxnu + 1) for k in range(problem.nu)]
            )

            def rollout_acados_for_grad(x_init: np.ndarray):
                current_states = x_init.copy()
                x_traj = np.zeros((sim_steps + 1, batch_size, problem.nx))
                u_traj = np.zeros((sim_steps, batch_size, problem.nu))
                k_traj = np.zeros((sim_steps, batch_size, problem.nu, problem.nx))
                x_traj[0] = current_states.copy()

                p_global_updated = build_p_global(Q, R)

                for step in range(sim_steps):
                    prepare_batch_step(current_states, p_global_updated)
                    solver.solve()

                    u_sol = solver.get_flat("u").reshape(
                        (batch_size, horizon - 1, problem.nu)
                    )
                    u_applied = u_sol[:, 0, :]

                    k_step = np.zeros((batch_size, problem.nu, problem.nx))
                    solver.setup_qp_matrices_and_factorize(batch_size)
                    for j in range(batch_size):
                        result = solver.ocp_solvers[j].eval_solution_sensitivity(
                            stages=0, with_respect_to="initial_state"
                        )
                        k_step[j] = result["sens_u"]

                    new_states = advance_dynamics(current_states, u_applied)

                    u_traj[step] = u_applied.copy()
                    k_traj[step] = k_step
                    x_traj[step + 1] = new_states.copy()
                    current_states = new_states

                grad_Q = np.zeros(problem.nx)
                grad_R = np.zeros(problem.nu)
                phi_batch = [np.zeros(problem.nx) for _ in range(batch_size)]

                for step in reversed(range(sim_steps)):
                    x_t = x_traj[step]
                    x_next = x_traj[step + 1]
                    u_t = u_traj[step]

                    prepare_batch_step(x_t, p_global_updated)
                    solver.solve()

                    phi_arr = np.stack(phi_batch)
                    all_seeds = (
                        2.0 * u_t
                        + 2.0 * (problem.dynamics.B.T @ x_next.T).T
                        + (problem.dynamics.B.T @ phi_arr.T).T
                    )[:, :, np.newaxis]

                    solver.setup_qp_matrices_and_factorize(batch_size)
                    p_sens_batch = solver.eval_adjoint_solution_sensitivity(
                        seed_x=None, seed_u=[(0, all_seeds)], sanity_checks=False
                    )

                    grad_Q += p_sens_batch[:, 0, q_diag_idx].sum(axis=0)
                    grad_R += p_sens_batch[:, 0, r_diag_idx].sum(axis=0)

                    for j in range(batch_size):
                        k_t = k_traj[step][j]
                        phi = phi_batch[j]
                        bk = problem.dynamics.A + problem.dynamics.B @ k_t
                        phi_batch[j] = (
                            bk.T @ (2.0 * x_next[j] + phi) + 2.0 * k_t.T @ u_t[j]
                        )

                return grad_Q, grad_R

            # Warm-up
            _ = rollout_with_costs(np.zeros_like(x0_batch), Q, R)
            _ = rollout_acados_for_grad(np.zeros_like(x0_batch))

            # AD gradients
            g_ad_q, g_ad_r = rollout_acados_for_grad(x0_batch)
            g_ad_np = {
                "weights_penalization_reference_state_trajectory": np.asarray(g_ad_q),
                "weights_penalization_control_squared": np.asarray(g_ad_r),
            }

            def rollout_for_fd(
                x_init: np.ndarray, weights: Dict[str, np.ndarray]
            ) -> np.ndarray:
                q_diag = np.asarray(weights["weights_penalization_reference_state_trajectory"])
                r_diag = np.asarray(weights["weights_penalization_control_squared"])
                q_mat = np.diag(q_diag)
                r_mat = np.diag(r_diag)
                return np.asarray(rollout_with_costs(x_init, q_mat, r_mat))

            fd_weights = {
                "weights_penalization_reference_state_trajectory": np.diag(Q).copy(),
                "weights_penalization_control_squared": np.diag(R).copy(),
            }
            g_fd_np = gradient_finite_diff(
                rollout_for_fd,
                x0_batch,
                weights=fd_weights,
                eps=fd_eps,
            )

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

            del solver
            gc.collect()

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

        print(f"\n  --- Summary for tol={tol:.0e} (n={n_seeds} seeds) ---")
        print(
            f"  {'Metric':<20}  {'mean':>8}  {'std':>8}  {'median':>8}  "
            f"{'min':>8}  {'max':>8}"
        )
        for metric in [
            "cos_Q",
            "cos_R",
            "cos_all",
            "rel_l2_Q",
            "rel_l2_R",
            "rel_l2_all",
        ]:
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
        row = ",".join(
            [
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
            ]
        )
        rows.append(row)

    with open(csv_path, "w") as f:
        f.write("\n".join(rows) + "\n")

    print(f"\nResults saved to {outdir}/")
    print("  Per-tolerance .npz files + summary.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="acados gradient accuracy sweep (adjoint vs FD cosine similarity)"
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--sim_steps", type=int, default=50)
    parser.add_argument("--n_state", type=int, default=8)
    parser.add_argument("--n_ctrl", type=int, default=4)
    parser.add_argument("--umax", type=float, default=10.0)

    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=[1e-1, 1e-3, 1e-5, 1e-7, 1e-9],
        help="Solver tolerances to sweep (space-separated list)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Single solver tolerance (overrides --tolerances)",
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

    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--outdir", type=str, default="timing_results/gradient_accuracy")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-seed output")

    args = parser.parse_args()
    tolerances = [args.tol] if args.tol is not None else args.tolerances

    print(f"\n{'=' * 70}")
    print("acados Gradient Accuracy Sweep")
    print(
        f"  batch={args.batch_size}, horizon={args.horizon}, "
        f"nx={args.n_state}, nu={args.n_ctrl}, umax={args.umax}"
    )
    print(f"  sim_steps={args.sim_steps}")
    print(f"  tolerances={tolerances}")
    print(f"  n_seeds={args.n_seeds}, fd_eps={args.fd_eps}")
    print(f"{'=' * 70}")

    results = run_gradient_accuracy_sweep(
        tolerances=tolerances,
        n_seeds=args.n_seeds,
        batch_size=args.batch_size,
        horizon=args.horizon,
        n_state=args.n_state,
        n_ctrl=args.n_ctrl,
        umax=args.umax,
        sim_steps=args.sim_steps,
        fd_eps=args.fd_eps,
        verbose=not args.quiet,
    )

    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY  (cosine similarity adjoint vs FD, concatenated Q+R gradient)")
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
        subdir = (
            f"acados/b{args.batch_size}_h{args.horizon}_nx{args.n_state}_nu{args.n_ctrl}_"
            f"umax{args.umax}_steps{args.sim_steps}_seeds{args.n_seeds}"
        )
        outdir = os.path.join(args.outdir, subdir)
        save_results(results, outdir, args)


if __name__ == "__main__":
    main()
