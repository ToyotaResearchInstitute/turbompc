import argparse
import os
import sys

# Allow imports from parent directory (utils, benchmark_naming, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Optional

import casadi as ca
import numpy as np
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpBatchSolver
from benchmark_naming import acados_dirname
from utils import DEBUGD, N_CTRL, generate_problem_data


# Dataclasses for defining OCP problem:
# https://github.com/FreyJo/differentiable_nmpc/benchmark_diff_mpc/problems.py
@dataclass
class LinearDiscreteDynamics:
    """Linear discrete-time dynamics x_{t+1} = A x_t + B u_t + b."""

    A: np.ndarray
    B: np.ndarray
    b: np.ndarray


@dataclass
class QuadraticCost:
    """Quadratic cost 0.5 x^T Q x + 0.5 u^T R u + x^T r + u^T q."""

    Q: np.ndarray
    R: np.ndarray
    r: np.ndarray
    q: np.ndarray


@dataclass
class ControlBounds:
    """Control bounds u_lower <= u <= u_upper."""

    u_lower: np.ndarray
    u_upper: np.ndarray


@dataclass
class ControlBoundedLqrProblem:
    """Linear quadratic regulator problem with control bounds."""

    dynamics: LinearDiscreteDynamics
    cost: QuadraticCost
    control_bounds: ControlBounds
    N_horizon: int

    def __post_init__(self):
        self.nx, self.nu = self.dynamics.A.shape[0], self.dynamics.B.shape[1]


def control_bounds_np(umax: float) -> tuple[np.ndarray, np.ndarray]:
    lower = -umax * np.ones((N_CTRL,), dtype=np.float64)
    upper = umax * np.ones((N_CTRL,), dtype=np.float64)
    return lower, upper


def create_acados_solver(
    problem: ControlBoundedLqrProblem,
    x0_batch: np.ndarray,
    seed: Optional[int] = None,
    n_batch: int = 1,
    num_threads: Optional[int] = 1,
    compute_gradient: bool = False,
    tol: float = 1e-3,
):
    # Create solver ONCE based on:
    # https://github.com/FreyJo/differentiable_nmpc/benchmark_diff_mpc/diff_acados.py

    # Get dimensions
    N_horizon = problem.N_horizon
    nx, nu = problem.nx, problem.nu

    ocp = AcadosOcp()
    model: AcadosModel = ocp.model
    model.x = ca.SX.sym("x", nx)
    model.u = ca.SX.sym("u", nu)
    model.name = f"linear_mpc_bench_{seed}"

    A_mat = ca.SX.sym("A", nx, nx)
    B_mat = ca.SX.sym("B", nx, nu)
    b = ca.SX.sym("b", nx)
    model.p_global = ca.vertcat(
        *[A_mat.reshape((-1, 1)), B_mat.reshape((-1, 1)), b.reshape((-1, 1))]
    )
    ocp.p_global_values = np.concatenate(
        (
            problem.dynamics.A.flatten(order="F"),
            problem.dynamics.B.flatten(order="F"),
            problem.dynamics.b,
        )
    )

    ocp.solver_options.integrator_type = "DISCRETE"
    model.disc_dyn_expr = A_mat @ model.x + B_mat @ model.u + b

    # Cost
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    H_mat = ca.SX.sym("H", nx + nu, nx + nu)
    xu = ca.vertcat(model.x, model.u)
    ocp.model.cost_expr_ext_cost = ca.mtimes([xu.T, H_mat, xu])
    ocp.model.cost_expr_ext_cost_e = ca.mtimes([model.x.T, H_mat[:nx, :nx], model.x])
    ocp.model.p_global = ca.vertcat(*[ocp.model.p_global, H_mat.reshape((-1, 1))])
    H_mat_val = scipy.linalg.block_diag(problem.cost.Q, problem.cost.R)
    ocp.p_global_values = np.concatenate(
        (ocp.p_global_values, H_mat_val.flatten(order="F"))
    )

    # Constraints
    ocp.constraints.lbu = problem.control_bounds.u_lower
    ocp.constraints.ubu = problem.control_bounds.u_upper
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.x0 = x0_batch[0, :]

    # Solver options
    ocp.solver_options.tf = N_horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.with_batch_functionality = True

    if compute_gradient:
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.with_solution_sens_wrt_params = True
        ocp.solver_options.qp_solver_cond_ric_alg = 0
        ocp.solver_options.qp_solver_ric_alg = (
            0  # classic Riccati required for eval_solution_sensitivity
        )

    ocp.solver_options.tol = tol
    ocp.solver_options.nlp_solver_max_iter = 1

    # Generate code in /tmp to avoid permission issues in Docker
    import tempfile

    build_dir = tempfile.mkdtemp(prefix="acados_")
    ocp.solver_options.build_dir = build_dir

    # Create batch solver ONCE
    solver = AcadosOcpBatchSolver(
        ocp, N_batch_init=n_batch, verbose=False, num_threads_in_batch_solve=num_threads
    )
    return solver, ocp


def get_num_threads_from_multiprocessing():
    import multiprocessing

    num_threads = multiprocessing.cpu_count()
    return num_threads


def benchmark_acados_rollout(
    n_batch,
    sim_steps,
    HORIZON,
    num_repeats: int = 1,
    umax: float = 1.0,
    n_state: int = None,
    n_ctrl: int = None,
    tol: float = 1e-7,
):
    """Benchmark acados rollout time with receding horizon MPC"""
    from utils import N_CTRL as _N_CTRL
    from utils import N_STATE as _N_STATE

    nx = n_state if n_state is not None else _N_STATE
    nu = n_ctrl if n_ctrl is not None else _N_CTRL
    print("=== ACADOS ===")

    all_times_fwd = []
    all_times_bwd = []
    all_gradients = []
    all_costs = []
    num_threads = get_num_threads_from_multiprocessing()

    for seed in range(num_repeats):
        Q, R, A_base, B_base, b_base, x0 = generate_problem_data(
            n_batch, seed, n_state=nx, n_ctrl=nu
        )

        # Problem object
        quadratic_cost = QuadraticCost(Q, R, np.zeros(nx), np.zeros(nu))
        dynamics = LinearDiscreteDynamics(A_base, B_base, b_base)
        u_lower = -umax * np.ones((nu,), dtype=np.float64)
        u_upper = umax * np.ones((nu,), dtype=np.float64)
        control_bounds = ControlBounds(u_lower, u_upper)
        # Convention: args.horizon counts time points, so N control intervals = horizon - 1.
        problem = ControlBoundedLqrProblem(
            dynamics, quadratic_cost, control_bounds, N_horizon=HORIZON - 1
        )
        x0_batch = x0.astype(np.float64)

        # Create solver:
        solver, ocp = create_acados_solver(
            problem,
            x0_batch,
            seed=seed,
            n_batch=n_batch,
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
            for j in range(n_batch):
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
            u_sol = solver.get_flat("u").reshape((n_batch, HORIZON - 1, problem.nu))
            u_applied = u_sol[:, 0, :]
            new_states = advance_dynamics(current_states, u_applied)
            return u_applied, new_states

        def rollout_with_costs(
            x_init: np.ndarray, Q_mat: np.ndarray, R_mat: np.ndarray
        ) -> float:
            """Receding horizon MPC rollout with specified cost matrices."""
            p_global_updated = build_p_global(Q_mat, R_mat)
            current_states = x_init.copy()
            total_cost = 0.0

            for step in range(sim_steps):
                u_applied, new_states = solve_batch_step(
                    current_states, p_global_updated
                )
                current_states = new_states
                total_cost += np.sum(new_states**2) + np.sum(u_applied**2)

            return total_cost

        # Warmup
        _ = rollout_with_costs(np.zeros_like(x0_batch), Q, R)

        # Time forward pass
        start_time = timer()
        final_cost = rollout_with_costs(x0_batch, Q, R)
        total_time = timer() - start_time

        print(
            f"fwd time: {total_time*1000:.1f} ms"
            f" ({total_time*1000/(n_batch*sim_steps):.3f} ms/step/problem)"
        )
        print(f"cost: {final_cost:.4f}")

        all_times_fwd.append(total_time)
        all_costs.append(final_cost)

        # Gradient computation
        # p_global layout: [A.flat(F), B.flat(F), b, H.flat(F)]  (sizes: nx^2, nx*nu, nx, (nx+nu)^2)
        # H = block_diag(Q, R), so diag(H)[: nx] = diag(Q), diag(H)[nx:] = diag(R).
        # From the flat Fortran-order H vector, the k-th diagonal entry is at index k*(nx+nu+1).
        nxnu = problem.nx + problem.nu
        H_offset = (
            problem.nx**2 + problem.nx * problem.nu + problem.nx
        )  # start of H in p_global

        # Indices in p_global for diag(Q) and diag(R)
        Q_diag_idx = H_offset + np.array([k * (nxnu + 1) for k in range(problem.nx)])
        R_diag_idx = H_offset + np.array(
            [(problem.nx + k) * (nxnu + 1) for k in range(problem.nu)]
        )

        def rollout_acados_for_grad(x_init):
            current_states = x_init.copy()
            x_traj = np.zeros((sim_steps + 1, n_batch, problem.nx))
            u_traj = np.zeros((sim_steps, n_batch, problem.nu))
            K_traj = np.zeros((sim_steps, n_batch, problem.nu, problem.nx))
            x_traj[0] = current_states.copy()

            p_global_updated = build_p_global(Q, R)

            for step in range(sim_steps):
                prepare_batch_step(current_states, p_global_updated)
                solver.solve()

                u_sol = solver.get_flat("u").reshape((n_batch, HORIZON - 1, problem.nu))
                u_applied = u_sol[:, 0, :]  # (n_batch, nu)

                K_step = np.zeros((n_batch, problem.nu, problem.nx))
                solver.setup_qp_matrices_and_factorize(n_batch)
                for j in range(n_batch):
                    result = solver.ocp_solvers[j].eval_solution_sensitivity(
                        stages=0, with_respect_to="initial_state"
                    )
                    K_step[j] = result["sens_u"]

                new_states = advance_dynamics(current_states, u_applied)

                u_traj[step] = u_applied.copy()
                K_traj[step] = K_step
                x_traj[step + 1] = new_states.copy()
                current_states = new_states

            grad_Q = np.zeros(problem.nx)
            grad_R = np.zeros(problem.nu)
            phi_batch = [np.zeros(problem.nx) for _ in range(n_batch)]

            for step in reversed(range(sim_steps)):
                x_t = x_traj[step]
                x_next = x_traj[step + 1]
                u_t = u_traj[step]

                prepare_batch_step(x_t, p_global_updated)
                solver.solve()

                # Vectorized seed computation: # (n_batch, nu)
                phi_arr = np.stack(phi_batch)  # (n_batch, nx)
                all_seeds = (
                    2.0 * u_t
                    + 2.0 * (problem.dynamics.B.T @ x_next.T).T
                    + (problem.dynamics.B.T @ phi_arr.T).T
                )[
                    :, :, np.newaxis
                ]  # (n_batch, nu, 1)

                solver.setup_qp_matrices_and_factorize(n_batch)
                p_sens_batch = solver.eval_adjoint_solution_sensitivity(
                    seed_x=None, seed_u=[(0, all_seeds)], sanity_checks=False
                )  # (n_batch, 1, np_global)

                grad_Q += p_sens_batch[:, 0, Q_diag_idx].sum(axis=0)
                grad_R += p_sens_batch[:, 0, R_diag_idx].sum(axis=0)

                # Update phi_batch per-element (K_t varies per batch element)
                for j in range(n_batch):
                    K_t = K_traj[step][j]
                    phi = phi_batch[j]
                    BK = (
                        problem.dynamics.A + problem.dynamics.B @ K_t
                    )  # closed-loop matrix (nx, nx)
                    phi_batch[j] = BK.T @ (2.0 * x_next[j] + phi) + 2.0 * K_t.T @ u_t[j]

            return grad_Q, grad_R

        # Warmup backward
        _ = rollout_acados_for_grad(np.zeros_like(x0_batch))

        # Time backward pass
        grad_start_time = timer()
        grad_Q, grad_R = rollout_acados_for_grad(x0_batch)
        grad_time = timer() - grad_start_time

        print(
            f"fwd + bwd time: {grad_time*1000:.1f} ms"
            f" ({grad_time*1000/(n_batch*sim_steps):.3f} ms/step/problem)"
        )

        if DEBUGD:
            print(f"grad_Q (d(reward)/d(Q_diag)): {grad_Q}")
            print(f"grad_R (d(reward)/d(R_diag)): {grad_R}")

        all_times_bwd.append(grad_time)

        all_gradients.append(
            {
                "weights_penalization_reference_state_trajectory": grad_Q,
                "weights_penalization_control_squared": grad_R,
            }
        )

    return (
        np.array(all_times_fwd),
        np.array(all_times_bwd),
        all_gradients,
        np.array(all_costs),
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
    parser.add_argument(
        "--tol", type=float, default=1e-7, help="Solver termination tolerance"
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Whether to save the results"
    )
    parser.add_argument(
        "--save_gradients",
        action="store_true",
        help="Also save grad_Q/grad_R .npy for compare_gradients.py",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"Num repeats: {args.num_repeats}")
    print(
        f"ACADOS Rollout: {args.n_state} states, {args.n_ctrl} controls,"
        f" {args.horizon} horizon"
    )
    print(
        f"Simulation: {args.sim_steps} steps, {args.batch_size} batch, umax={args.umax}"
    )
    print(f"Settings: tol={args.tol}")

    forward_times, backwd_times, gradients, costs = benchmark_acados_rollout(
        args.batch_size,
        args.sim_steps,
        args.horizon,
        args.num_repeats,
        args.umax,
        n_state=args.n_state,
        n_ctrl=args.n_ctrl,
        tol=args.tol,
    )

    if DEBUGD:
        print("\n=== Summary ===")
        print(f"Cost per repeat: {costs}")
        print(f"Cost sum: {np.sum(costs):.4f}")

    if gradients and args.save_gradients:
        last = gradients[-1]
        print(
            "grad_Q shape:"
            f" {last['weights_penalization_reference_state_trajectory'].shape}  values:"
            f" {last['weights_penalization_reference_state_trajectory']}"
        )
        print(
            f"grad_R shape: {last['weights_penalization_control_squared'].shape}  "
            f"values: {last['weights_penalization_control_squared']}"
        )

    # Save results
    if args.save_results:
        dirname = acados_dirname(
            args.batch_size,
            args.horizon,
            args.n_state + args.n_ctrl,
            args.num_repeats,
            constrained=True,
            umax=args.umax,
            tol=args.tol,
            sim_steps=args.sim_steps,
        )
        device = "cpu"
        os.makedirs(f"timing_results/{dirname}", exist_ok=True)
        np.save(f"timing_results/{dirname}/acados_{device}_fwd", forward_times)
        np.save(f"timing_results/{dirname}/acados_{device}_bwd", backwd_times)
        if args.save_gradients and gradients:
            np.save(
                f"timing_results/{dirname}/acados_{device}_grad_Q",
                np.stack(
                    [
                        g["weights_penalization_reference_state_trajectory"]
                        for g in gradients
                    ]
                ),
            )
            np.save(
                f"timing_results/{dirname}/acados_{device}_grad_R",
                np.stack(
                    [g["weights_penalization_control_squared"] for g in gradients]
                ),
            )
        print("Results saved")


if __name__ == "__main__":
    main()
