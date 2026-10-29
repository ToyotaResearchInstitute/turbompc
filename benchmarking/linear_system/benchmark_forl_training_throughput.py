"""Benchmark linear MPC training throughput with SHAC."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


import numpy as np
import torch
from torch import Tensor, nn

from examples.rl.env.runtime import Transition
from examples.rl.shac.shac import SHAC, SHACConfig
from examples.rl.util.networks import mlp

try:
    from .utils import generate_problem_data
except ImportError:  # pragma: no cover - direct script execution
    from utils import generate_problem_data


HERE = Path(__file__).resolve().parent
BACKENDS = ("acados", "mpcpytorch", "logbarrier")


def _synchronize(device: torch.device, adapter: Any) -> None:
    adapter.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class PhaseMeasurement:
    collection_seconds: float
    update_seconds: float
    total_seconds: float
    actor_loss: float
    actor_gradient_norm: float
    critic_loss: float


def _torch_to_jax(value: Tensor):
    from examples.rl.util.jax_torch import torch_to_jax

    return torch_to_jax(value)


def _jax_to_torch(value: Any, *, like: Tensor) -> Tensor:
    from examples.rl.util.jax_torch import jax_to_torch

    return jax_to_torch(value, like=like)


class _LogBarrierMeanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cost_logits: Tensor, state: Tensor, adapter, chunk_index: int):
        controls, pullback = adapter.forward_chunk(cost_logits, state, int(chunk_index))
        ctx.adapter = adapter
        ctx.pullback = pullback
        ctx.save_for_backward(cost_logits, state)
        return controls

    @staticmethod
    def backward(ctx, control_gradient: Tensor):
        cost_logits, state = ctx.saved_tensors
        cost_gradient, state_gradient = ctx.adapter.backward_chunk(
            ctx.pullback, cost_logits, state, control_gradient
        )
        return cost_gradient, state_gradient, None, None


class BatchedAdapter:
    """Split the training batch into MPC microbatches."""

    def solve(self, costs, state):
        size = self.solver_batch_size
        return torch.cat(
            [
                self.solve_chunk(
                    costs[start : start + size], state[start : start + size], index
                )
                for index, start in enumerate(range(0, self.total_batch_size, size))
            ]
        )


class LogBarrierAdapter(BatchedAdapter):
    name = "logbarrier"

    def __init__(
        self,
        *,
        A: np.ndarray,
        B: np.ndarray,
        b: np.ndarray,
        n_state: int,
        n_ctrl: int,
        mpc_horizon: int,
        solver_batch_size: int,
        total_batch_size: int,
        umax: float,
        tolerance: float,
        admm_max_iterations: int,
        target_kappa: float,
        device: torch.device,
        **_unused: Any,
    ):
        import jax
        import jax.numpy as jnp

        try:
            from .benchmark_turbompc_constrained import _make_config
        except ImportError:  # pragma: no cover - direct script execution
            from benchmark_turbompc_constrained import _make_config
        from turbompc.problems.optimal_control_problem import OptimalControlProblem
        from turbompc.solvers.turbompc_solver import (
            HessianApproximation,
            TurboMPCSolver,
            parse_backward_backend,
            parse_forward_backend,
        )

        # The command-line benchmark requires CUDA. Dense CPU solves are also
        # available here for small numerical adapter tests.
        self.forward_backend = (
            "admm_fused_cudss" if device.type == "cuda" else "admm_jax_loop_jax_dense"
        )
        self.backward_backend = (
            "direct_cudss_ffi" if device.type == "cuda" else "direct_jax_dense"
        )
        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.jnp = jnp
        self.n_state = int(n_state)
        self.n_ctrl = int(n_ctrl)
        self.intervals = int(mpc_horizon) - 1
        self.solver_batch_size = int(solver_batch_size)
        self.total_batch_size = int(total_batch_size)
        self.chunk_count = total_batch_size // solver_batch_size

        benchmark_config = _make_config(
            horizon=mpc_horizon - 1,
            umax=umax,
            pcg_eps=1.0e-12,
            warm_start=True,
            alpha=1.0,
            n_state=n_state,
            n_ctrl=n_ctrl,
            admm_max_iter=admm_max_iterations,
            tol=tolerance,
        )
        benchmark_config.solver_params.update(
            constraint_handling="log_barrier",
            log_barrier_kappa=target_kappa,
            num_sqp_iteration_max=1,
        )
        benchmark_config.solver_params["linesearch"].update(
            # Skip merit search; the barrier still limits steps near bounds.
            method="none",
            fraction_to_boundary=0.99,
        )
        # Only input bounds are constrained in the shared linear benchmark.
        benchmark_config.problem_params["state_min_bounds"] = jnp.full(
            n_state, -jnp.inf
        )
        benchmark_config.problem_params["state_max_bounds"] = jnp.full(n_state, jnp.inf)
        benchmark_config.problem_params["dynamics_state_dot_params"] = {
            "A": jnp.asarray(A) - jnp.eye(n_state),
            "B": jnp.asarray(B),
            "b": jnp.asarray(b),
        }
        self.problem_params = copy.deepcopy(benchmark_config.problem_params)
        problem = OptimalControlProblem(
            dynamics=benchmark_config.dynamics,
            params=self.problem_params,
        )
        self.solver = TurboMPCSolver(
            program=problem,
            params=benchmark_config.solver_params,
            forward_backend=parse_forward_backend(self.forward_backend),
            backward_backend=parse_backward_backend(self.backward_backend),
            backward_hessian=HessianApproximation.FULL_HESSIAN,
        )
        self.q_key = "weights_penalization_reference_state_trajectory"
        self.r_key = "weights_penalization_control_squared"
        single_guess = self.solver.initial_guess(self.problem_params)

        def broadcast(value):
            if value is None:
                return None
            array = jnp.asarray(value)
            return jnp.broadcast_to(array, (self.solver_batch_size,) + array.shape)

        self._cold_guess = jax.tree_util.tree_map(broadcast, single_guess)
        self._guesses = [self._cold_guess] * self.chunk_count

        def weights(cost):
            q_logit = cost[: self.n_state]
            r_logit = cost[self.n_state :]
            return {
                self.q_key: jnp.exp(jnp.clip(q_logit, -8.0, 8.0)),
                self.r_key: jnp.exp(jnp.clip(r_logit, -8.0, 8.0)),
            }

        differentiable_solve = self.solver.get_differentiable_solve_function()

        def solve_batch(cost, state, guess):
            guess = guess._replace(
                states=guess.states.at[:, 0].set(state),
                admm_state=None,
            )

            def solve_one(row_guess, row_cost, row_state):
                params = {**self.problem_params, "initial_state": row_state}
                return differentiable_solve(row_guess, params, weights(row_cost))

            solution = jax.vmap(solve_one)(guess, cost, state)
            return solution.controls[:, 0], solution

        self._solve_batch = jax.jit(solve_batch)
        self._last_ready = None
        self._admm_statuses: list[Any] = []
        self._admm_iterations: list[Any] = []

    def reset(self) -> None:
        self._guesses = [self._cold_guess] * self.chunk_count

    def clear_statistics(self) -> None:
        self._admm_statuses.clear()
        self._admm_iterations.clear()

    def forward_chunk(
        self, cost_logits: Tensor, state: Tensor, chunk_index: int
    ) -> tuple[Tensor, Any]:
        guess = self._guesses[chunk_index]
        # Retain JAX's pullback so Torch backward reuses this forward solve.
        controls, pullback, solution = self.jax.vjp(
            lambda cost, initial_state: self._solve_batch(cost, initial_state, guess),
            _torch_to_jax(cost_logits),
            _torch_to_jax(state),
            has_aux=True,
        )
        warm_solution = solution._replace(admm_state=None)
        self._guesses[chunk_index] = self.jax.tree_util.tree_map(
            self.jax.lax.stop_gradient, warm_solution
        )
        self._admm_statuses.append(solution.status)
        self._admm_iterations.append(self.jnp.sum(solution.admm_iters, axis=-1))
        self._last_ready = solution.states
        return _jax_to_torch(controls, like=state), pullback

    def backward_chunk(
        self,
        pullback: Any,
        cost_logits: Tensor,
        state: Tensor,
        control_gradient: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cost_gradient, state_gradient = pullback(_torch_to_jax(control_gradient))
        self._last_ready = state_gradient
        return (
            _jax_to_torch(cost_gradient, like=cost_logits),
            _jax_to_torch(state_gradient, like=state),
        )

    def solve_chunk(self, costs, state, index):
        return _LogBarrierMeanFunction.apply(costs, state, self, index)

    def synchronize(self) -> None:
        if self._last_ready is not None:
            self.jax.block_until_ready(self._last_ready)

    def metadata(self) -> dict[str, Any]:
        devices = [str(value) for value in self.jax.devices()]
        statuses = np.asarray(self.jax.device_get(self._admm_statuses)).ravel()
        iterations = np.asarray(self.jax.device_get(self._admm_iterations)).ravel()
        if np.any(~np.isin(statuses, (0, 1))):
            raise RuntimeError(
                f"TurboMPC failed with SQP statuses {np.unique(statuses).tolist()}"
            )
        return {
            "jax_version": self.jax.__version__,
            "jax_devices": devices,
            "linear_forward_backend": self.forward_backend,
            "linear_backward_backend": self.backward_backend,
            "sqp_solve_count": int(statuses.size),
            "sqp_iteration_limit_count": int(np.count_nonzero(statuses != 0)),
            "admm_mean_iterations": (
                float(iterations.mean()) if iterations.size else None
            ),
            "admm_max_iterations_observed": (
                int(iterations.max()) if iterations.size else None
            ),
        }


class _AcadosMeanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cost_logits: Tensor, state: Tensor, adapter, chunk_index: int):
        controls, parameters, iterate = adapter.forward_chunk(
            cost_logits, state, int(chunk_index)
        )
        ctx.adapter = adapter
        ctx.parameters = parameters
        ctx.iterate = iterate
        ctx.cost_dtype = cost_logits.dtype
        ctx.state_dtype = state.dtype
        return controls

    @staticmethod
    def backward(ctx, control_gradient: Tensor):
        cost_gradient, state_gradient = ctx.adapter.backward_chunk(
            ctx.parameters, ctx.iterate, control_gradient
        )
        return (
            cost_gradient.to(dtype=ctx.cost_dtype),
            state_gradient.to(dtype=ctx.state_dtype),
            None,
            None,
        )


class AcadosAdapter(BatchedAdapter):
    name = "acados"

    def __init__(
        self,
        *,
        A: np.ndarray,
        B: np.ndarray,
        b: np.ndarray,
        n_state: int,
        n_ctrl: int,
        mpc_horizon: int,
        solver_batch_size: int,
        total_batch_size: int,
        umax: float,
        tolerance: float,
        num_threads: int,
        **_unused: Any,
    ):
        import casadi as ca
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpBatchSolver

        self.n_state = int(n_state)
        self.n_ctrl = int(n_ctrl)
        self.intervals = int(mpc_horizon) - 1
        self.solver_batch_size = int(solver_batch_size)
        self.total_batch_size = int(total_batch_size)
        self.chunk_count = total_batch_size // solver_batch_size
        self._warm_iterates: list[Any | None] = [None] * self.chunk_count
        self._statuses: list[int] = []

        ocp = AcadosOcp()
        model = AcadosModel()
        model.name = f"linear_shac_n{mpc_horizon}_m{solver_batch_size}_{os.getpid()}"
        model.x = ca.MX.sym("x", n_state)
        model.u = ca.MX.sym("u", n_ctrl)
        p_x0 = ca.MX.sym("p_x0", n_state)
        p_cost = ca.MX.sym("p_cost", n_state + n_ctrl)
        model.p_global = ca.vertcat(p_x0, p_cost)
        model.disc_dyn_expr = A @ model.x + B @ model.u + b
        q = ca.exp(ca.fmin(8.0, ca.fmax(-8.0, p_cost[:n_state])))
        r = ca.exp(ca.fmin(8.0, ca.fmax(-8.0, p_cost[n_state:])))
        model.cost_expr_ext_cost = ca.dot(q, model.x**2) + ca.dot(r, model.u**2)
        model.cost_expr_ext_cost_e = ca.dot(q, model.x**2)
        model.con_h_expr_0 = model.x - p_x0
        ocp.model = model

        intervals = self.intervals
        ocp.solver_options.N_horizon = intervals
        ocp.solver_options.tf = float(intervals)
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.constraints.lbu = -float(umax) * np.ones(n_ctrl)
        ocp.constraints.ubu = float(umax) * np.ones(n_ctrl)
        ocp.constraints.idxbu = np.arange(n_ctrl)
        ocp.constraints.lh_0 = np.zeros(n_state)
        ocp.constraints.uh_0 = np.zeros(n_state)
        ocp.constraints.constr_type_0 = "BGH"
        ocp.p_global_values = np.zeros(2 * n_state + n_ctrl)
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.with_batch_functionality = True
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.qp_solver_cond_ric_alg = 0
        ocp.solver_options.qp_solver_ric_alg = 0
        ocp.solver_options.nlp_solver_max_iter = 1
        ocp.solver_options.tol = float(tolerance)
        if hasattr(ocp, "code_gen_options"):
            # acados_template through the Figure 9 image.
            ocp.code_gen_options.with_solution_sens_wrt_params_adj = True
            code_generation = ocp.code_gen_options
        else:
            # acados_template >= 0.5.4 renamed the code-generation object and
            # moved the parametric-sensitivity switch to solver options.
            ocp.solver_options.with_solution_sens_wrt_params = True
            code_generation = ocp.code_gen_opts
        build_directory = tempfile.mkdtemp(prefix="acados_linear_shac_")
        code_generation.code_export_directory = build_directory
        if hasattr(code_generation, "json_file"):
            json_file = str(Path(build_directory) / "acados_ocp.json")
            code_generation.json_file = json_file
        else:
            json_file = str(Path(build_directory) / "acados_ocp.json")
        self.batch_solver = AcadosOcpBatchSolver(
            ocp,
            N_batch_init=solver_batch_size,
            num_threads_in_batch_solve=num_threads,
            json_file=json_file,
            verbose=False,
        )

    def reset(self) -> None:
        self._warm_iterates = [None] * self.chunk_count
        if hasattr(self.batch_solver, "reset"):
            self.batch_solver.reset()

    def clear_statistics(self) -> None:
        self._statuses.clear()

    def forward_chunk(
        self, cost_logits: Tensor, state: Tensor, chunk_index: int
    ) -> tuple[Tensor, np.ndarray, Any]:
        if cost_logits.device.type != "cpu" or state.device.type != "cpu":
            raise ValueError("acados SHAC tensors must be on CPU")
        previous = self._warm_iterates[chunk_index]
        if previous is not None:
            self.batch_solver.set_iterate(previous)
        parameters = np.ascontiguousarray(
            torch.cat((state, cost_logits), dim=-1).detach().numpy(),
            dtype=np.float64,
        )
        self.batch_solver.set_p_global_and_precompute_dependencies(parameters)
        returned_status = self.batch_solver.solve()
        status_value = getattr(self.batch_solver, "status", returned_status)
        if status_value is not None:
            status = np.asarray(status_value).reshape(-1)
            self._statuses.extend(int(value) for value in status)
            fatal = ~np.isin(status, (0, 2))
            if np.any(fatal):
                raise RuntimeError(f"acados solve failed: {status.tolist()}")
        if hasattr(self.batch_solver, "get"):
            controls = self.batch_solver.get(0, "u")
        else:
            controls = self.batch_solver.get_flat("u").reshape(
                self.solver_batch_size, self.intervals, self.n_ctrl
            )[:, 0]
        controls = np.ascontiguousarray(controls, dtype=np.float64)
        if not np.all(np.isfinite(controls)):
            raise RuntimeError("acados returned non-finite controls")
        iterate = self.batch_solver.get_flat_iterate()
        self._warm_iterates[chunk_index] = iterate
        return torch.as_tensor(controls, dtype=state.dtype), parameters, iterate

    def backward_chunk(
        self, parameters: np.ndarray, iterate: Any, control_gradient: Tensor
    ) -> tuple[Tensor, Tensor]:
        self.batch_solver.set_p_global_and_precompute_dependencies(parameters)
        self.batch_solver.set_iterate(iterate)
        self.batch_solver.setup_qp_matrices_and_factorize(self.solver_batch_size)
        seed = np.ascontiguousarray(control_gradient.detach().numpy())[:, :, None]
        sensitivity = self.batch_solver.eval_adjoint_solution_sensitivity(
            seed_x=None,
            seed_u=[(0, seed)],
            with_respect_to="p_global",
            sanity_checks=False,
        )[:, 0, :]
        if not np.all(np.isfinite(sensitivity)):
            raise RuntimeError("acados returned non-finite adjoint sensitivity")
        state_gradient = sensitivity[:, : self.n_state]
        cost_gradient = sensitivity[:, self.n_state :]
        return torch.from_numpy(cost_gradient), torch.from_numpy(state_gradient)

    def solve_chunk(self, costs, state, index):
        return _AcadosMeanFunction.apply(costs, state, self, index)

    @staticmethod
    def synchronize() -> None:
        return None

    def metadata(self) -> dict[str, Any]:
        statuses = np.asarray(self._statuses, dtype=np.int64)
        return {
            "acados_allowed_statuses": [0, 2],
            "acados_status_counts": {
                str(int(value)): int(np.count_nonzero(statuses == value))
                for value in np.unique(statuses)
            },
            "acados_nonzero_status_rows": int(np.count_nonzero(statuses)),
            "acados_max_status": int(statuses.max()) if statuses.size else 0,
        }


class MPCPyTorchAdapter(BatchedAdapter):
    name = "mpcpytorch"

    def __init__(
        self,
        *,
        A: np.ndarray,
        B: np.ndarray,
        b: np.ndarray,
        n_state: int,
        n_ctrl: int,
        mpc_horizon: int,
        solver_batch_size: int,
        total_batch_size: int,
        umax: float,
        tolerance: float,
        device: torch.device,
        **_unused: Any,
    ):
        dependency = HERE.parent / "mpc.pytorch"
        if str(dependency) not in sys.path:
            sys.path.insert(0, str(dependency))
        from mpc import mpc
        from mpc.dynamics import AffineDynamics

        self.mpc = mpc
        self.n_state = int(n_state)
        self.n_ctrl = int(n_ctrl)
        self.horizon = int(mpc_horizon)
        self.solver_batch_size = int(solver_batch_size)
        self.total_batch_size = int(total_batch_size)
        self.chunk_count = total_batch_size // solver_batch_size
        self.device = device
        self.dtype = torch.float64
        self.A = torch.as_tensor(A, dtype=self.dtype, device=device)
        self.B = torch.as_tensor(B, dtype=self.dtype, device=device)
        self.b = torch.as_tensor(b, dtype=self.dtype, device=device)
        self.dynamics = AffineDynamics(self.A, self.B, self.b)
        lower = torch.full(
            (self.horizon, solver_batch_size, n_ctrl),
            -float(umax),
            dtype=self.dtype,
            device=device,
        )
        upper = torch.full_like(lower, float(umax))
        self.controller = mpc.MPC(
            n_state=n_state,
            n_ctrl=n_ctrl,
            T=self.horizon,
            u_lower=lower,
            u_upper=upper,
            lqr_iter=1,
            eps=float(tolerance),
            back_eps=float(tolerance),
            n_batch=solver_batch_size,
            max_linesearch_iter=1,
            verbose=0,
            backprop=True,
            grad_method=mpc.GradMethods.ANALYTIC,
            exit_unconverged=False,
            detach_unconverged=False,
        )
        self._warm_controls: list[Tensor | None] = [None] * self.chunk_count

    def reset(self) -> None:
        self._warm_controls = [None] * self.chunk_count

    @staticmethod
    def clear_statistics() -> None:
        return None

    def solve_chunk(
        self, cost_logits: Tensor, state: Tensor, chunk_index: int
    ) -> Tensor:
        q = torch.exp(torch.clamp(cost_logits[:, : self.n_state], -8.0, 8.0))
        r = torch.exp(torch.clamp(cost_logits[:, self.n_state :], -8.0, 8.0))
        diagonal = torch.cat((q, r), dim=-1)
        diagonal = diagonal.unsqueeze(0).expand(self.horizon, -1, -1)
        matrix = 2.0 * torch.diag_embed(diagonal)
        linear = torch.zeros_like(diagonal)
        cost = self.mpc.QuadCost(matrix, linear)
        self.controller.u_init = self._warm_controls[chunk_index]
        _, controls, _ = self.controller(state, cost, self.dynamics)
        self._warm_controls[chunk_index] = controls.detach()
        return controls[0]

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def metadata(self) -> dict[str, Any]:
        return {"mpcpytorch_lqr_iterations": 1}


def make_adapter(backend: str, **kwargs: Any):
    if backend == "logbarrier":
        return LogBarrierAdapter(**kwargs)
    if backend == "acados":
        return AcadosAdapter(**kwargs)
    if backend == "mpcpytorch":
        return MPCPyTorchAdapter(**kwargs)
    raise ValueError(f"unknown backend {backend!r}")


class LinearRuntime:
    """Linear dynamics and an MPC policy for SHAC."""

    def __init__(self, adapter, matrices, initial_state, umax, device):
        self.adapter, self.umax, self.device = adapter, umax, device
        self.dtype = torch.float64
        self.A, self.B, self.b = (
            torch.as_tensor(value, dtype=self.dtype, device=device)
            for value in matrices
        )
        self.initial_state = torch.as_tensor(
            initial_state, dtype=self.dtype, device=device
        )
        self.observation_dim, self.action_dim = self.A.shape[0], self.B.shape[1]
        self.done = torch.zeros(len(initial_state), dtype=torch.bool, device=device)
        self.valid = torch.ones_like(self.done)

    def reset(self, batch_size, seed):
        self.adapter.reset()
        return self.initial_state.clone()

    @staticmethod
    def observe(state):
        return state

    @staticmethod
    def detach(state):
        return state.detach()

    def policy(self, parameters, state, *, retain):
        return SimpleNamespace(
            mean=self.adapter.solve(parameters, state), valid=self.valid
        )

    def squash_action(self, action):
        return action.clamp(-self.umax, self.umax)

    def step(self, state, action, policy):
        next_state = state @ self.A.T + action @ self.B.T + self.b
        reward = -(next_state.square().sum(-1) + action.square().sum(-1))
        return Transition(
            next_state, next_state, reward, self.done, self.done, self.valid
        )

    @staticmethod
    def reset_done(state, done, seed):
        return state


class LinearSHACBenchmark(SHAC):
    """Benchmark SHAC with a selected MPC solver."""

    def __init__(
        self,
        *,
        backend,
        total_batch_size,
        solver_batch_size,
        mpc_horizon,
        training_horizon,
        n_state,
        n_ctrl,
        umax,
        tolerance,
        admm_max_iterations,
        target_kappa,
        critic_iterations,
        critic_minibatches,
        seed,
        device,
        num_threads,
        exploration_std=0.05,
    ):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        _, _, A, B, b, initial_state = generate_problem_data(
            total_batch_size, seed, n_state=n_state, n_ctrl=n_ctrl
        )
        # Initialize in float64 on CPU to preserve identical weights across solvers.
        self.actor = mlp(
            n_state, (256, 256), n_state + n_ctrl, "relu", 0.5, dtype=torch.float64
        ).to(device)
        self.actor.register_buffer(
            "log_std",
            torch.full(
                (n_ctrl,), exploration_std, dtype=torch.float64, device=device
            ).log(),
        )
        self.critic = nn.Sequential(
            mlp(n_state, (256, 256), 1, "tanh", dtype=torch.float64),
            nn.Flatten(0),
        ).to(device)
        self.device = device
        self.adapter = make_adapter(
            backend,
            A=A,
            B=B,
            b=b,
            n_state=n_state,
            n_ctrl=n_ctrl,
            mpc_horizon=mpc_horizon,
            solver_batch_size=solver_batch_size,
            total_batch_size=total_batch_size,
            umax=umax,
            tolerance=tolerance,
            admm_max_iterations=admm_max_iterations,
            target_kappa=target_kappa,
            device=device,
            num_threads=num_threads,
        )
        runtime = LinearRuntime(self.adapter, (A, B, b), initial_state, umax, device)
        settings = SHACConfig(
            seed=seed,
            num_envs=total_batch_size,
            training_horizon=training_horizon,
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-3,
            anneal_lr=False,
            critic_iterations=critic_iterations,
            critic_minibatches=critic_minibatches,
            max_gradient_norm=100.0,
            reward_clip=500.0,
        )
        super().__init__(runtime, self.actor, self.critic, settings)

    @staticmethod
    def _guard(value):
        # Benchmarks must fail on invalid adjoints instead of sanitizing them.
        return value

    def update(self, rollout):
        metrics = super().update(rollout)
        return {
            key: metrics[source]
            for key, source in (
                ("actor_loss", "train/actor_loss"),
                ("actor_gradient_norm", "train/actor_gradient_norm"),
                ("critic_loss", "train/value_loss"),
            )
        }

    def measure(self):
        _synchronize(self.device, self.adapter)
        start = time.perf_counter()
        rollout = self.collect()
        _synchronize(self.device, self.adapter)
        after_collection = time.perf_counter()
        metrics = self.update(rollout)
        _synchronize(self.device, self.adapter)
        after_update = time.perf_counter()
        return PhaseMeasurement(
            collection_seconds=after_collection - start,
            update_seconds=after_update - after_collection,
            total_seconds=after_update - start,
            **metrics,
        )


def _device(backend: str, requested: str) -> torch.device:
    if backend == "acados":
        if requested not in ("auto", "cpu"):
            raise ValueError("acados runs on CPU")
        return torch.device("cpu")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type != "cuda":
        raise ValueError(f"{backend} throughput benchmark requires CUDA")
    return device


def _median(measurements: list[PhaseMeasurement], field: str) -> float:
    return float(statistics.median(getattr(value, field) for value in measurements))


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    positive = {
        "total batch size": args.total_batch_size,
        "solver batch size": args.solver_batch_size,
        "MPC horizon": args.mpc_horizon,
        "training horizon": args.training_horizon,
        "state dimension": args.n_state,
        "control dimension": args.n_ctrl,
        "critic iterations": args.critic_iterations,
        "critic minibatches": args.critic_minibatches,
        "repeats": args.repeats,
        "thread count": args.num_threads,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"positive values required for: {', '.join(invalid)}")
    if args.mpc_horizon < 2:
        raise ValueError("MPC horizon must contain at least two time points")
    if args.warmups < 1:
        raise ValueError("at least one complete warm-up is required")
    if not args.umax > 0.0:
        raise ValueError("umax must be positive")
    if not np.isfinite(args.exploration_std) or args.exploration_std <= 0:
        raise ValueError("exploration standard deviation must be finite and positive")
    if args.total_batch_size % args.solver_batch_size:
        raise ValueError("solver batch size must divide the total SHAC batch")
    device = _device(args.backend, args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    benchmark = LinearSHACBenchmark(
        backend=args.backend,
        total_batch_size=args.total_batch_size,
        solver_batch_size=args.solver_batch_size,
        mpc_horizon=args.mpc_horizon,
        training_horizon=args.training_horizon,
        n_state=args.n_state,
        n_ctrl=args.n_ctrl,
        umax=args.umax,
        tolerance=args.tolerance,
        admm_max_iterations=args.admm_max_iterations,
        target_kappa=args.target_kappa,
        critic_iterations=args.critic_iterations,
        critic_minibatches=args.critic_minibatches,
        seed=args.seed,
        device=device,
        num_threads=min(args.num_threads, args.solver_batch_size),
        exploration_std=args.exploration_std,
    )
    for index in range(args.warmups):
        print(f"warmup {index + 1}/{args.warmups}", flush=True)
        benchmark.measure()
    print("warmup complete", flush=True)
    benchmark.adapter.clear_statistics()
    measurements = []
    for index in range(args.repeats):
        print(f"starting measured repeat {index + 1}/{args.repeats}", flush=True)
        value = benchmark.measure()
        measurements.append(value)
        print(
            (
                f"repeat {index + 1}/{args.repeats}: "
                f"collect={value.collection_seconds:.4f}s "
                f"update={value.update_seconds:.4f}s "
                f"total={value.total_seconds:.4f}s"
            ),
            flush=True,
        )
    collection = _median(measurements, "collection_seconds")
    update = _median(measurements, "update_seconds")
    total = _median(measurements, "total_seconds")
    problem_steps = args.total_batch_size * args.training_horizon
    return {
        "schema_version": 3,
        "backend": args.backend,
        "device": str(device),
        "dtype": "float64",
        "total_batch_size": args.total_batch_size,
        "solver_batch_size": args.solver_batch_size,
        "mpc_horizon": args.mpc_horizon,
        "training_horizon": args.training_horizon,
        "n_state": args.n_state,
        "n_ctrl": args.n_ctrl,
        "umax": args.umax,
        "solver_tolerance": args.tolerance,
        "admm_max_iterations": (
            args.admm_max_iterations if args.backend == "logbarrier" else None
        ),
        "target_kappa": args.target_kappa if args.backend == "logbarrier" else None,
        "critic_iterations": args.critic_iterations,
        "critic_minibatches": args.critic_minibatches,
        "exploration_std": args.exploration_std,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "median_collection_seconds": collection,
        "median_update_seconds": update,
        "median_total_seconds": total,
        "training_problem_steps_per_second": problem_steps / total,
        "measurements": [asdict(value) for value in measurements],
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cpu_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        **benchmark.adapter.metadata(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--total-batch-size", type=int, default=1024)
    parser.add_argument("--solver-batch-size", type=int, required=True)
    parser.add_argument("--mpc-horizon", type=int, default=40)
    parser.add_argument("--training-horizon", type=int, default=50)
    parser.add_argument("--n-state", type=int, default=8)
    parser.add_argument("--n-ctrl", type=int, default=4)
    parser.add_argument("--umax", type=float, default=10.0)
    parser.add_argument(
        "--exploration-std",
        type=float,
        default=0.05,
        help="Fixed Gaussian action standard deviation, independent of umax",
    )
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    parser.add_argument("--admm-max-iterations", type=int, default=1000)
    parser.add_argument("--target-kappa", type=float, default=1.0e-4)
    parser.add_argument("--critic-iterations", type=int, default=16)
    parser.add_argument("--critic-minibatches", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    result = run_case(args)
    encoded = json.dumps(result, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(f"RESULT_JSON={encoded}", flush=True)
    return result


if __name__ == "__main__":
    main()
