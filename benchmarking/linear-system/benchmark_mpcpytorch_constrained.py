"""Benchmarking script."""
import argparse
import gc
import os
import sys
import time

# Allow imports from parent directory (utils, benchmark_naming, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


import numpy as np
import torch
from benchmark_naming import mpcpytorch_dirname

# MPC.pytorch imports
from mpc import mpc
from mpc.dynamics import AffineDynamics
from mpc.mpc import QuadCost
from utils import (
    DEBUGD,
    MPC_LQR_ITER,
    MPC_MAX_LINES,
    MPC_TOL,
    N_CTRL,
    N_STATE,
    generate_problem_data,
)


def benchmark_mpc_pytorch_rollout(  # noqa: C901
    device,
    N_BATCH,
    sim_steps,
    HORIZON,
    num_repeats,
    umax,
    n_state=None,
    n_ctrl=None,
    tol=None,
    lqr_iter=None,
):
    """Benchmark mpc.pytorch"""
    nx = n_state if n_state is not None else N_STATE
    nu = n_ctrl if n_ctrl is not None else N_CTRL
    solver_tol = tol if tol is not None else MPC_TOL
    solver_lqr_iter = lqr_iter if lqr_iter is not None else MPC_LQR_ITER
    print("=== MPC.pytorch ===")

    device = torch.device(device)
    print(f"Using device: {device}")

    all_times_fwd = []
    all_times_bwd = []
    all_gradients = []
    for seed in range(num_repeats):
        Q, R, A_base, B_base, b_base, x0 = generate_problem_data(
            N_BATCH, seed, n_state=nx, n_ctrl=nu
        )

        # Move to GPU
        Q, R, A_base, B_base, b_base, x0 = [
            torch.from_numpy(x).to(device) for x in [Q, R, A_base, B_base, b_base, x0]
        ]

        # Create cost
        c_batch = torch.zeros(
            HORIZON, N_BATCH, nx + nu, dtype=torch.double, device=device
        )

        # Create bounds
        u_lower_batch = (
            torch.ones(HORIZON, N_BATCH, nu, dtype=torch.double, device=device) * -umax
        )
        u_upper_batch = (
            torch.ones(HORIZON, N_BATCH, nu, dtype=torch.double, device=device) * umax
        )

        dynamics = AffineDynamics(A_base, B_base, b_base)

        # Create solver
        mpc_solver = mpc.MPC(
            n_state=nx,
            n_ctrl=nu,
            T=HORIZON,
            u_lower=u_lower_batch,
            u_upper=u_upper_batch,
            lqr_iter=solver_lqr_iter,
            eps=solver_tol,
            n_batch=N_BATCH,
            max_linesearch_iter=MPC_MAX_LINES,
            verbose=0,
            backprop=True,
            grad_method=mpc.GradMethods.ANALYTIC,
            exit_unconverged=False,
            detach_unconverged=False,
        )

        def rollout_pytorch(Q_weights, x_init):
            current_states = x_init.clone()
            total_costs = 0.0
            for step in range(sim_steps):
                QR_weighted = torch.block_diag(torch.diag(Q_weights), R)
                H_weighted = (
                    QR_weighted.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(HORIZON, N_BATCH, -1, -1)
                )
                quad_cost_weighted = QuadCost(H_weighted, c_batch)

                x_sol, u_sol, _ = mpc_solver(
                    current_states, quad_cost_weighted, dynamics
                )
                u_applied = u_sol[0]
                new_states = (
                    A_base @ current_states.T
                    + B_base @ u_applied.T
                    + b_base.unsqueeze(1)
                ).T
                current_states = new_states
                total_costs += torch.sum(new_states**2) + torch.sum(u_applied**2)
            return total_costs

        def rollout_for_grad(Q_weights):
            return rollout_pytorch(Q_weights, x0).sum()

        # Warmup forward
        _ = rollout_pytorch(
            torch.ones(nx, dtype=torch.double, device=device), torch.zeros_like(x0)
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Time forward
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.monotonic()
        final_cost = rollout_pytorch(torch.diag(Q), x0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.monotonic() - start_time

        print(
            f"fwd time: {total_time*1000:.1f} ms"
            f" ({total_time*1000/(N_BATCH*sim_steps):.3f} ms/step/problem)"
        )
        print(f"cost: {final_cost.sum():.4f}")

        # Clear final_cost to free memory
        final_cost_value = final_cost.sum().item()
        del final_cost
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        def finite_diff_grad(Q_weights, eps=1e-6):
            Qc_weights_np = Q_weights.detach().cpu().numpy()
            grad = np.zeros_like(Qc_weights_np)
            for i in range(len(Q_weights)):
                w_plus = Q_weights.clone()
                w_minus = Q_weights.clone()
                w_plus[i] += eps
                w_minus[i] -= eps
                f_plus = rollout_pytorch(w_plus, x0).item()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear GPU memory
                f_minus = rollout_pytorch(w_minus, x0).item()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear GPU memory
                grad[i] = (f_plus - f_minus) / (2 * eps)
            return grad

        # Warmup backward
        Q_weights = torch.ones(
            nx, dtype=torch.double, device=device, requires_grad=True
        )
        rollout_for_grad(Q_weights).backward()
        Q_weights.grad.zero_()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Time backward
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        grad_start_time = time.monotonic()
        rollout_for_grad(Q_weights).backward()
        # print(f"Q_weights.grad: {Q_weights.grad}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        grad_time = time.monotonic() - grad_start_time

        print(
            f"fwd + bwd time: {grad_time*1000:.1f} ms"
            f" ({grad_time*1000/(N_BATCH*sim_steps):.3f} ms/step/problem)"
        )

        grad_values = Q_weights.grad.cpu().numpy()
        if DEBUGD:
            fd_grad_values = finite_diff_grad(Q_weights)
            print(f"Gradient values: {grad_values}")
            print(f"Gradient has NaN: {np.isnan(grad_values).any()}")
            print("Gradient check:")
            for k in range(len(grad_values)):
                diff = np.abs(grad_values[k] - fd_grad_values[k])
                rel_diff = diff / (np.abs(fd_grad_values[k]) + 1e-8)
                print(
                    f"  [{k}] grad={grad_values[k]:.6e}, fd={fd_grad_values[k]:.6e},"
                    f" abs_diff={diff:.6e}, rel_diff={rel_diff:.6e}"
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        all_times_fwd.append(total_time)
        all_times_bwd.append(grad_time)
        all_gradients.append(grad_values)

        # Clear memory between repeats - be aggressive
        del Q, R, A_base, B_base, b_base, x0
        del c_batch, u_lower_batch, u_upper_batch
        del dynamics, mpc_solver, Q_weights
        if "rollout_pytorch" in locals():
            del rollout_pytorch
        if "rollout_for_grad" in locals():
            del rollout_for_grad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

    return (
        np.array(all_times_fwd),
        np.array(all_times_bwd),
        all_gradients[-1] if all_gradients else None,
    )


def main():
    parser = argparse.ArgumentParser(description="MPC Benchmark")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for benchmark"
    )
    parser.add_argument("--horizon", type=int, default=40, help="MPC horizon")
    parser.add_argument(
        "--sim_steps", type=int, default=50, help="Number of simulation steps"
    )
    parser.add_argument(
        "--num_repeats", type=int, default=1, help="Number of repeats for benchmark"
    )
    parser.add_argument("--umax", type=float, default=10.0, help="Control upper bound")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--n_state",
        type=int,
        default=None,
        help="State dimension (overrides utils.N_STATE)",
    )
    parser.add_argument(
        "--n_ctrl",
        type=int,
        default=None,
        help="Control dimension (overrides utils.N_CTRL)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Solver tolerance (overrides utils.MPC_TOL)",
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Whether to save the results"
    )

    args = parser.parse_args()

    nx = args.n_state if args.n_state is not None else N_STATE
    nu = args.n_ctrl if args.n_ctrl is not None else N_CTRL
    solver_tol = args.tol if args.tol is not None else MPC_TOL

    print(f"\n{'=' * 70}")
    print(f"Num repeats: {args.num_repeats}")
    print(f"MPC.pytorch Rollout: {nx} states, {nu} controls, {args.horizon} horizon")
    print(
        f"Simulation: {args.sim_steps} steps, {args.batch_size} batch, umax={args.umax}"
    )
    print(f"Settings: MPC({MPC_LQR_ITER} iter, tol={solver_tol})")

    forward_times, backward_times, gradients = benchmark_mpc_pytorch_rollout(
        args.device,
        args.batch_size,
        args.sim_steps,
        args.horizon,
        args.num_repeats,
        args.umax,
        n_state=args.n_state,
        n_ctrl=args.n_ctrl,
        tol=args.tol,
    )

    # Save results
    if args.save_results:
        dirname = mpcpytorch_dirname(
            args.batch_size,
            args.horizon,
            nx + nu,
            args.num_repeats,
            constrained=True,
            umax=args.umax,
            tol=solver_tol,
        )
        device = args.device
        os.makedirs(f"timing_results/{dirname}", exist_ok=True)
        np.save(f"timing_results/{dirname}/mpcpytorch_{device}_fwd", forward_times)
        np.save(f"timing_results/{dirname}/mpcpytorch_{device}_bwd", backward_times)
        np.save(f"timing_results/{dirname}/mpcpytorch_{device}_grad", gradients)
        print("Results saved")


if __name__ == "__main__":
    main()
