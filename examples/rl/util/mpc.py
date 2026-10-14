"""Batched TurboMPC solves and retained implicit gradients."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from turbompc.solvers.status import SQPStatus
from turbompc.solvers.turbompc_solver import HessianApproximation, TurboMPCSolver
from turbompc.utils.load_params import load_solver_params


@dataclass
class MPCResult:
    """Batched MPC plans, validity masks, and fallbacks."""

    mean: Any
    states: Any
    controls: Any
    iterate: Any
    cache: Any
    valid: Any
    stats: dict[str, Any]


class TurboSeed(NamedTuple):
    """Primal-only warm start: ``(states, controls, valid)`` arrays."""

    states: Any
    controls: Any
    valid: Any


@dataclass
class _TurboCache:
    """Per-row JAX pullbacks retained during the forward solve."""

    batch_size: int
    pullbacks: Any
    raw_cost: jax.Array
    context: jax.Array
    valid: jax.Array


def _final_completed_sqp_stat(history, num_iter, requested) -> jax.Array:
    """Select the last completed SQP statistic for each row."""
    values = jnp.asarray(history)
    iterations = jnp.asarray(num_iter, dtype=jnp.int32)
    requested = jnp.asarray(requested, dtype=jnp.bool_)
    index = jnp.clip(iterations - 1, 0, values.shape[1] - 1)
    final = jnp.take_along_axis(values, index[:, None], axis=1)[:, 0]
    return jnp.where(
        requested & (iterations > 0),
        final,
        jnp.asarray(jnp.nan, dtype=values.dtype),
    )


def _replace_inactive_batched_rows(tree, active, template_index):
    """Replace inactive pytree rows with a selected active row."""
    batch_size = active.shape[0]

    def replace(value):
        if value is None:
            return None
        value = jnp.asarray(value)
        if value.ndim == 0 or value.shape[0] != batch_size:
            return value
        mask = active.reshape((batch_size,) + (1,) * (value.ndim - 1))
        template = jnp.take(value, template_index, axis=0)
        return jnp.where(mask, value, template)

    return jax.tree_util.tree_map(replace, tree, is_leaf=lambda value: value is None)


class MPCBackend:
    """Batch MPC solves and adjoints for a supplied problem."""

    supports_plan_adjoint = True

    def __init__(self, config, problem):
        if not jax.config.jax_enable_x64:
            raise ValueError(
                "MPC requires JAX x64; configure it before constructing the runtime"
            )
        self.problem = problem
        solver_config = config["solver"]
        parameters = dict(load_solver_params("turbompc.yaml"))
        for key, value in solver_config.items():
            if key in {
                "forward_backend",
                "backward_backend",
                "use_slack",
                "slack_weight",
            }:
                continue
            if isinstance(value, dict):
                parameters[key] = {**parameters.get(key, {}), **value}
            else:
                parameters[key] = value
        self.solver = TurboMPCSolver(
            problem.make_ocp(),
            parameters,
            forward_backend=solver_config["forward_backend"],
            backward_backend=solver_config["backward_backend"],
            backward_hessian=HessianApproximation.FULL_HESSIAN,
        )
        self.horizon, self.nx, self.nu = problem.N, problem.nx, problem.nu
        self.cost_dim = problem.cost_dim
        self.admm_max_iterations = int(parameters["admm"]["max_iter"])
        self._step_cache = {}

        def safe_batched_plan_vjp(pullbacks, raw_cost, context, active, dx, du):
            zeros = (jnp.zeros_like(raw_cost), jnp.zeros_like(context))

            def apply(_):
                # Each pullback leaf has a leading batch axis from vmap. Replacing
                # rejected rows prevents singular/NaN adjoint systems, even at zero RHS.
                safe = jax.lax.cond(
                    jnp.all(active),
                    lambda _: pullbacks,
                    lambda _: _replace_inactive_batched_rows(
                        pullbacks, active, jnp.argmax(active.astype(jnp.int32))
                    ),
                    operand=None,
                )
                return jax.vmap(lambda pb, x, u: pb((x, u)))(safe, dx, du)

            gradients = jax.lax.cond(
                jnp.any(active), apply, lambda _: zeros, operand=None
            )
            return tuple(
                jnp.where(active[:, None] & jnp.isfinite(value), value, 0.0)
                for value in gradients
            )

        self._safe_batched_plan_vjp = jax.jit(safe_batched_plan_vjp)

    def zero_seed(self, batch_size: int) -> TurboSeed:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        batch = int(batch_size)
        return TurboSeed(
            jnp.zeros((batch, self.horizon + 1, self.nx)),
            jnp.zeros((batch, self.horizon + 1, self.nu)),
            jnp.zeros((batch,), dtype=jnp.bool_),
        )

    @staticmethod
    def _require_seed(seed: Any, batch_size: int | None = None) -> TurboSeed:
        if not isinstance(seed, TurboSeed):
            raise TypeError("TurboMPC requires its opaque batched warm start")
        valid = jnp.asarray(seed.valid, dtype=jnp.bool_)
        if valid.ndim != 1:
            raise ValueError("TurboMPC seed validity must be rank one")
        if batch_size is not None and valid.shape != (int(batch_size),):
            raise ValueError("TurboMPC seed batch dimension does not agree")
        return TurboSeed(seed.states, seed.controls, valid)

    def set_seed_initial_control(self, seed: Any, control: Any) -> TurboSeed:
        """Update a warm start's initial control."""
        seed = self._require_seed(seed)
        batch = int(seed.valid.shape[0])
        control = jnp.asarray(control, dtype=jnp.float64)
        if control.shape != (batch, self.nu):
            raise ValueError(
                "initial seed control must have shape "
                f"({batch}, {self.nu}), got {control.shape}"
            )
        return TurboSeed(
            seed.states,
            jnp.asarray(seed.controls).at[:, 0].set(control),
            seed.valid,
        )

    def invalidate_seed_rows(self, seed: Any, mask) -> TurboSeed:
        seed = self._require_seed(seed)
        return TurboSeed(
            seed.states,
            seed.controls,
            seed.valid & ~jnp.asarray(mask, dtype=jnp.bool_),
        )

    def select_seed_rows(
        self,
        preferred: Any,
        fallback: Any,
        mask,
    ) -> TurboSeed:
        """Select warm-start rows with a mask."""
        preferred = self._require_seed(preferred)
        batch = int(preferred.valid.shape[0])
        fallback = self._require_seed(fallback, batch)
        mask = jnp.asarray(mask, dtype=jnp.bool_)
        if mask.shape != (batch,):
            raise ValueError(
                f"seed selection mask must have shape ({batch},), got {mask.shape}"
            )
        return TurboSeed(
            jnp.where(mask[:, None, None], preferred.states, fallback.states),
            jnp.where(mask[:, None, None], preferred.controls, fallback.controls),
            jnp.where(mask, preferred.valid, fallback.valid),
        )

    def encode_seed(self, seed: Any) -> jax.Array:
        """Flatten warm-start rows for PPO storage."""
        seed = self._require_seed(seed)
        batch = int(seed.valid.shape[0])
        states = jnp.asarray(seed.states, dtype=jnp.float64)
        controls = jnp.asarray(seed.controls, dtype=jnp.float64)
        expected_states = (batch, self.horizon + 1, self.nx)
        expected_controls = (batch, self.horizon + 1, self.nu)
        if states.shape != expected_states or controls.shape != expected_controls:
            raise ValueError("TurboMPC seed plan dimensions do not agree")
        return jnp.concatenate(
            [
                states.reshape(batch, -1),
                controls.reshape(batch, -1),
                seed.valid.astype(jnp.float64)[:, None],
            ],
            axis=1,
        )

    def decode_seed(self, encoded: Any) -> TurboSeed:
        """Reconstruct warm starts from PPO storage."""
        encoded = jnp.asarray(encoded, dtype=jnp.float64)
        state_size = (self.horizon + 1) * self.nx
        control_size = (self.horizon + 1) * self.nu
        feature_size = state_size + control_size + 1
        if encoded.ndim != 2 or encoded.shape[1] != feature_size:
            raise ValueError(
                f"encoded TurboMPC seed must have shape (B, {feature_size})"
            )
        batch = int(encoded.shape[0])
        states = encoded[:, :state_size].reshape(batch, self.horizon + 1, self.nx)
        controls = encoded[:, state_size : state_size + control_size].reshape(
            batch, self.horizon + 1, self.nu
        )
        valid = encoded[:, -1] > 0.5
        return TurboSeed(states, controls, valid)

    def _step_for(self, batch: int):
        if batch in self._step_cache:
            return self._step_cache[batch]
        problem, solver = self.problem, self.solver

        def plan(cost, context, states, controls):
            params, _ = problem.params_from_context(context)
            weights = problem.cost_weights(cost, context)
            states = states.at[0].set(params["initial_state"])
            if problem.constrain_initial_control:
                controls = controls.at[0].set(params["initial_control"])
            guess = solver.initial_guess(params)._replace(
                states=states, controls=controls, admm_state=None
            )
            solution = solver.solve(guess, params, weights)
            return (solution.states, solution.controls), solution

        def retain_row(cost, context, states, controls):
            _, pullback, solution = jax.vjp(
                lambda c, x: plan(c, x, states, controls), cost, context, has_aux=True
            )
            return solution, pullback

        def step(
            costs,
            context,
            seed_states,
            seed_controls,
            seed_valid,
            fallback_valid,
            active,
            *,
            retain,
        ):
            params, _ = problem.params_from_context(context)
            # Reference rollouts are only needed by cold or fallback rows.
            ref_states, ref_controls = jax.lax.cond(
                jnp.all(seed_valid & fallback_valid),
                lambda _: (seed_states, seed_controls),
                lambda _: problem.build_mpc_inputs(costs, context)[2:],
                operand=None,
            )
            # Keep seed validity separate from availability of a fallback plan.
            fallback_states = jnp.where(
                fallback_valid[:, None, None], seed_states, ref_states
            )
            fallback_controls = jnp.where(
                fallback_valid[:, None, None], seed_controls, ref_controls
            )
            use = seed_valid[:, None, None]
            states = jnp.where(use, seed_states, ref_states)
            cold_controls = ref_controls
            if problem.constrain_initial_control:
                cold_controls = cold_controls.at[:, 0].set(params["initial_control"])
            controls = jnp.where(use, seed_controls, cold_controls)
            initial_controls = controls[:, 0]
            controls = problem.clip_control(controls)
            controls = controls.at[:, 0].set(initial_controls)
            states = states.at[:, 0].set(
                problem.mpc_warm_start_state(states[:, 0], context)
            )
            if retain:
                solution, pullbacks = jax.vmap(retain_row)(
                    costs, context, states, controls
                )
            else:
                inputs = (costs, context, states, controls)
                # Keep inactive evaluation rows from slowing active solves.
                solve_inputs = jax.lax.cond(
                    jnp.all(active),
                    lambda _: inputs,
                    lambda _: _replace_inactive_batched_rows(
                        inputs, active, jnp.argmax(active.astype(jnp.int32))
                    ),
                    operand=None,
                )
                _, solution = jax.vmap(plan)(*solve_inputs)
                pullbacks = None
            status_accepted = (solution.status == jnp.int32(SQPStatus.SOLVED)) | (
                solution.status == jnp.int32(SQPStatus.MAX_ITER)
            )
            valid = (
                active
                & status_accepted
                & jnp.all(jnp.isfinite(solution.states), axis=(1, 2))
                & jnp.all(jnp.isfinite(solution.controls), axis=(1, 2))
            )
            executed = problem.clip_control(solution.controls[:, 0])
            # Failed solves retain the problem's finite warm/reference plan.
            fallback_mean = fallback_controls[:, 0]
            mean = jnp.where(valid[:, None], executed, fallback_mean)
            safe_states = jnp.where(
                valid[:, None, None], solution.states, fallback_states
            )
            safe_controls = jnp.where(
                valid[:, None, None], solution.controls, fallback_controls
            )
            admm_iterations = jnp.where(
                active,
                jnp.sum(solution.admm_iters, axis=-1).astype(jnp.int32),
                0,
            )
            stats = {
                "inner_iterations": admm_iterations,
                "sqp_iterations": solution.num_iter,
                "status": solution.status,
            }
            s = solution.stats
            histories = (
                ("sqp_convergence_error", s.convergence_errors),
                ("nlp_equality", s.equality_violations),
                ("nlp_inequality", s.inequality_violations),
                ("admm_primal", s.qp_stats.primal_residual),
                ("admm_dual", s.qp_stats.dual_residual),
            )
            for key, history in histories:
                stats[key] = _final_completed_sqp_stat(
                    history, solution.num_iter, active
                )
            return solution, safe_states, safe_controls, mean, valid, stats, pullbacks

        self._step_cache[batch] = jax.jit(step, static_argnames=("retain",))
        return self._step_cache[batch]

    def solve_device(
        self,
        costs,
        context,
        seed,
        retain=True,
        active=None,
        history_seed=None,
    ) -> MPCResult:
        """Solve an MPC batch using ``seed`` for initialization.

        ``history_seed`` identifies rows with a fallback plan.
        """
        batch_costs = jnp.asarray(costs, dtype=jnp.float64)
        batch_context = jnp.asarray(context, dtype=jnp.float64)
        if batch_costs.ndim != 2:
            raise ValueError("costs must have shape (B, cost_dim)")
        batch = batch_costs.shape[0]
        if batch <= 0 or batch_costs.shape != (batch, self.cost_dim):
            raise ValueError(f"costs must have shape ({batch}, {self.cost_dim})")
        if batch_context.ndim < 2 or batch_context.shape[0] != batch:
            raise ValueError("solve batch dimensions do not agree")
        seed = self._require_seed(seed, batch)
        history_seed = (
            seed if history_seed is None else self._require_seed(history_seed, batch)
        )
        active_rows = (
            jnp.ones((batch,), dtype=jnp.bool_)
            if active is None
            else jnp.asarray(active, dtype=jnp.bool_)
        )
        if active_rows.shape != (batch,):
            raise ValueError(
                f"active must have shape ({batch},), got {active_rows.shape}"
            )

        # A jitted function's cache key distinguishes committed from
        # uncommitted input shardings.  Actor costs arrive through DLPack as
        # committed device arrays, while ``zero_seed`` is deliberately
        # device-neutral; outputs from the first solve/advance are committed.
        # Put every step input on one explicit sharding so the cold call and
        # all later warm-start calls reuse the same compiled executable.
        step_sharding = batch_costs.sharding
        batch_costs = jax.device_put(batch_costs, step_sharding)
        batch_context = jax.device_put(batch_context, step_sharding)
        seed_states = jax.device_put(
            jnp.asarray(seed.states, dtype=jnp.float64), step_sharding
        )
        seed_controls = jax.device_put(
            jnp.asarray(seed.controls, dtype=jnp.float64), step_sharding
        )
        seed_valid = jax.device_put(seed.valid, step_sharding)
        fallback_valid = jax.device_put(history_seed.valid, step_sharding)
        active_rows = jax.device_put(active_rows, step_sharding)

        step = self._step_for(batch)

        solution, states, controls, mean, valid, stats, pullbacks = step(
            batch_costs,
            batch_context,
            seed_states,
            seed_controls,
            seed_valid,
            fallback_valid,
            active_rows,
            retain=retain,
        )
        iterate = TurboSeed(solution.states, solution.controls, valid)
        cache = None
        if retain:
            cached_valid = jnp.asarray(valid, dtype=jnp.bool_)
            cache = _TurboCache(
                batch_size=batch,
                pullbacks=pullbacks,
                raw_cost=batch_costs,
                context=batch_context,
                valid=cached_valid,
            )
        return MPCResult(
            mean=mean,
            states=states,
            controls=controls,
            iterate=iterate,
            cache=cache,
            valid=valid,
            stats=stats,
        )

    def vjp_plan_device(
        self,
        cache: _TurboCache,
        state_gradient,
        control_gradient,
    ) -> dict[str, jax.Array]:
        """Apply batched implicit adjoints to retained solutions."""
        if not isinstance(cache, _TurboCache):
            raise TypeError("TurboMPC vjp requires its retained forward cache")
        state_gradient = jnp.asarray(state_gradient, dtype=jnp.float64)
        control_gradient = jnp.asarray(control_gradient, dtype=jnp.float64)
        batch = cache.batch_size
        expected_x = (batch, self.horizon + 1, self.nx)
        expected_u = (batch, self.horizon + 1, self.nu)
        if state_gradient.shape != expected_x:
            raise ValueError(
                f"state_gradient must have shape {expected_x}, got "
                f"{state_gradient.shape}"
            )
        if control_gradient.shape != expected_u:
            raise ValueError(
                f"control_gradient must have shape {expected_u}, got "
                f"{control_gradient.shape}"
            )

        active = jnp.asarray(cache.valid, dtype=jnp.bool_)
        if active.shape != (batch,):
            raise ValueError("retained validity mask has the wrong shape")

        state_gradient = jnp.where(active[:, None, None], state_gradient, 0.0)
        control_gradient = jnp.where(active[:, None, None], control_gradient, 0.0)
        cost_gradient, context_gradient = self._safe_batched_plan_vjp(
            cache.pullbacks,
            jnp.asarray(cache.raw_cost),
            jnp.asarray(cache.context),
            active,
            state_gradient,
            control_gradient,
        )
        return {
            "costs": cost_gradient,
            "context": context_gradient,
        }
