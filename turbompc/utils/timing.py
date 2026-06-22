"""Generalized closed-loop rollout benchmarking utilities."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np
from jax import checkpoint, jit, vmap
from turbompc.dynamics.base_dynamics import Dynamics
from turbompc.dynamics.integrators import DiscretizationScheme, predict_next_state
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)
from turbompc.utils.gradient_finitediff import gradient_finite_diff

# ---------------------------------------------------------------------------
# ProblemConfig
# ---------------------------------------------------------------------------


@dataclass
class ProblemConfig:
    dynamics: Dynamics
    problem_class: Type[OptimalControlProblem]
    problem_params: Dict[str, Any]
    solver_params: Dict[str, Any]
    weight_keys: List[str]
    reward_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    update_per_seed: Callable[
        [int, int, Dict[str, Any]],
        Tuple[Dict[str, Any], jnp.ndarray],
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_step_dynamics_params(problem_params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract single-step dynamics params from (possibly time-tiled) OCP params."""
    dyn_params = problem_params.get("dynamics_state_dot_params")
    if not dyn_params:
        return {}
    horizon = int(problem_params["horizon"])
    out: Dict[str, Any] = {}
    for key, value in dyn_params.items():
        arr = jnp.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] in (horizon, horizon + 1):
            out[key] = arr[0]
        else:
            out[key] = arr
    return out


# ---------------------------------------------------------------------------
# Rollout builder
# ---------------------------------------------------------------------------


def build_rollout_fn(
    config: ProblemConfig,
    solver: TurboMPCSolver,
    problem_params: Dict[str, Any],
    init_solution: Any,
    warm_start: bool,
    num_sim_steps: int,
) -> Callable:
    """Return a single-environment rollout fn: (state, weights) -> (cost, admm_iters)."""
    dt = float(problem_params["discretization_resolution"])
    scheme = DiscretizationScheme(int(problem_params["discretization_scheme"]))
    dynamics = config.dynamics
    reward_fn = config.reward_fn
    sim_dyn_params = _single_step_dynamics_params(problem_params)

    def rollout(state: jnp.ndarray, weights: Dict[str, Any]):
        initial_carry = (state, jax.lax.stop_gradient(init_solution), jnp.asarray(0.0))

        def rollout_step(carry, _):
            current_state, current_solution, running_cost = carry
            new_weights = {**weights, "initial_state": current_state}
            solution = solver.solve(current_solution, problem_params, new_weights)
            u_applied = solution.controls[0]
            # For implicit trapezoidal, the OCP uses u[t+1] in the dynamics constraint;
            u_next = (
                solution.controls[1]
                if scheme == DiscretizationScheme.IMPLICIT
                else u_applied
            )
            new_state = predict_next_state(
                dynamics, dt, scheme, sim_dyn_params, current_state, u_applied, u_next
            )
            new_running_cost = running_cost - reward_fn(new_state, u_applied)
            # warm_start=False should be a true cold-start rollout: keep using the fixed initial guess.
            # This keeps AD and FD consistent in gradient-accuracy benchmarks.
            next_solution = solution if warm_start else current_solution
            return (new_state, next_solution, new_running_cost), solution.admm_iters

        # Checkpoint each scan step to trade compute for memory.
        # Without this, JAX stores all intermediate states (50 steps × SQP × ADMM)
        # for the backward pass, causing OOM at large batch × horizon.
        final_carry, admm_iters = jax.lax.scan(
            jax.checkpoint(rollout_step), initial_carry, None, length=num_sim_steps
        )
        _, _, final_cost = final_carry
        return final_cost, admm_iters

    return rollout


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------


def benchmark_rollout(
    config: ProblemConfig,
    batch_size: int,
    num_sim_steps: int,
    num_repeats: int,
    forward_backend: ForwardBackend = ForwardBackend.ADMM_JAX_LOOP_PCG,
    backward_backend: BackwardBackend = BackwardBackend.ADMM_JAX_LOOP_PCG,
    warm_start: bool = True,
    run_backward: bool = True,
    use_full_hessian: bool = False,
    fd_check: bool = False,
    verbose: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[List[Optional[Dict[str, np.ndarray]]]],
    np.ndarray,
    np.ndarray,
    "TurboMPCSolver",
]:
    """Closed-loop rollout timing benchmark.

    Returns (fwd_times, bwd_times, seed_gradients, admm_iters, costs, solver).
    The solver is returned so callers can reuse its compiled XLA kernels.
    """
    problem_label = (
        f"{type(config.dynamics).__name__}/"
        f"{getattr(config.problem_class, '__name__', str(config.problem_class))}"
    )
    label = f"fwd={forward_backend.name} bwd={backward_backend.name}"
    if verbose:
        nx = config.dynamics.num_states
        nu = config.dynamics.num_controls
        h = config.problem_params["horizon"]
        print(f"\n{'=' * 70}")
        print(f"TurboMPC Rollout  [{problem_label}]  backend={label}")
        print(
            f"  nx={nx}, nu={nu}, control_intervals={h}, sim_steps={num_sim_steps}, "
            f"batch={batch_size}, repeats={num_repeats}"
        )
        print(f"  warm_start={warm_start}  run_backward={run_backward}")

    problem_params = dict(config.problem_params)
    problem = config.problem_class(dynamics=config.dynamics, params=problem_params)
    solver = TurboMPCSolver(
        program=problem,
        params=config.solver_params,
        forward_backend=forward_backend,
        backward_backend=backward_backend,
        use_full_hessian=use_full_hessian,
    )

    all_times_fwd: List[float] = []
    all_times_bwd: List[float] = []
    all_gradients: List[Optional[Dict[str, np.ndarray]]] = []
    all_admm_iters: List[list] = []
    all_costs: List[float] = []

    for seed in range(num_repeats):
        param_updates, initial_states = config.update_per_seed(
            seed, batch_size, problem_params
        )
        problem_params.update(param_updates)
        weights = {k: problem_params[k] for k in config.weight_keys}

        init_solution = solver.solve(
            solver.initial_guess(problem_params),
            problem_params=problem_params,
            weights=weights,
        )

        rollout = build_rollout_fn(
            config, solver, problem_params, init_solution, warm_start, num_sim_steps
        )
        rollout_batch = jit(vmap(rollout, in_axes=(0, None)))

        def rollout_for_grad(w):
            costs, _ = rollout_batch(initial_states, w)
            return jnp.sum(costs)

        if run_backward:
            rollout_grad = jit(jax.grad(checkpoint(rollout_for_grad)))

        # JIT warm-up
        out = rollout_batch(jnp.zeros_like(initial_states), weights)
        jax.block_until_ready(out)

        # Forward timing
        t0 = time.monotonic()
        final_costs, num_admm_iters = rollout_batch(initial_states, weights)
        jax.block_until_ready(final_costs)
        fwd_time = time.monotonic() - t0

        # Backward timing
        if run_backward:
            r = rollout_grad(weights)
            jax.block_until_ready(r)
            t0 = time.monotonic()
            gradients = rollout_grad(weights)
            jax.block_until_ready(gradients)
            bwd_time = time.monotonic() - t0

            if fd_check:
                fd_grads = gradient_finite_diff(
                    rollout_batch, initial_states, weights=weights
                )
                print(f"  --- FD check [seed={seed}] ---")
                for k in gradients:
                    g_ad = np.array(gradients[k])
                    g_fd = fd_grads[k]
                    diff = np.abs(g_ad - g_fd)
                    rel = diff / (np.abs(g_fd) + 1e-8)
                    for i in range(g_ad.size):
                        idx = np.unravel_index(i, g_ad.shape)
                        print(
                            f"  {k}{list(idx)}  AD={g_ad[idx]:.4e}  FD={g_fd[idx]:.4e}"
                            f"  abs={diff[idx]:.2e}  rel={rel[idx]:.2e}"
                        )
        else:
            gradients = None
            bwd_time = 0.0

        if verbose:
            n = max(1, batch_size * num_sim_steps)
            print(
                f"  [seed={seed}]  fwd={fwd_time*1000:.1f} ms "
                f" ({fwd_time*1000/n:.3f} ms/step/batch) "
                f" cost={float(final_costs.sum()):.4f}"
            )
            if run_backward:
                print(
                    f"  [seed={seed}]  fwd + bwd={bwd_time*1000:.1f} ms"
                    f"  ({bwd_time*1000/n:.3f} ms/step/batch)"
                )

        all_times_fwd.append(fwd_time)
        all_times_bwd.append(bwd_time)
        all_costs.append(float(final_costs.sum()))
        all_admm_iters.append(num_admm_iters.flatten().tolist())
        all_gradients.append(
            {k: np.array(gradients[k]) for k in gradients}
            if gradients is not None
            else None
        )

        jax.clear_caches()
        gc.collect()

    return (
        np.array(all_times_fwd),
        np.array(all_times_bwd),
        all_gradients if run_backward else None,
        np.array(all_admm_iters),
        np.array(all_costs),
        solver,
    )
