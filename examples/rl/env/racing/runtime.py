"""Racing episodes with direct and MPC policies.

This code uses synthetic data and parameters, and does not represent any manufacturer's product.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import torch

from ...util.jax_torch import MPC, PlantStep, jax_to_torch, torch_to_jax
from ...util.training import zero_nonfinite_gradient
from ..runtime import EvaluationProtocol, Transition
from .env import PATH_EVOLUTION, SPEED
from .pd import PIDParameters, ReferencePDController


@dataclass
class RacingState:
    state: torch.Tensor
    applied_control: torch.Tensor
    steps: torch.Tensor
    off_track_count: torch.Tensor
    physics: Any = None
    warm_seed: Any = None
    last_successful_seed: Any = None
    progress_since_success: torch.Tensor | None = None
    controller_memory: torch.Tensor | None = None


@dataclass
class Policy:
    mean: torch.Tensor
    controls: torch.Tensor
    valid: torch.Tensor
    metrics: dict
    result: Any = None
    pd_control: torch.Tensor | None = None
    controller_memory: torch.Tensor | None = None


def make_controller(config, dt):
    pd = config.get("controller", {}).get("pd", {})
    if not pd.get("enabled", False):
        return None
    longitudinal = pd.get("longitudinal_pid", {})

    def parameters(name):
        values = longitudinal.get(name, {})
        return PIDParameters(
            *(
                values.get(key, default)
                for key, default in (
                    ("kp", 0.0),
                    ("ki", 0.0),
                    ("kd", 0.0),
                    ("integral_limit", 1.0),
                    ("command_limit_nm", 1.0),
                )
            )
        )

    return ReferencePDController(
        lateral_kp=pd.get("lateral_kp", 0.0),
        lateral_kd=pd.get("lateral_kd", 0.0),
        dt=dt,
        throttle_deadband=longitudinal.get("throttle_deadband_mps", 0.2),
        brake_deadband=longitudinal.get("brake_deadband_mps", 0.4),
        throttle=parameters("throttle"),
        brake=parameters("brake"),
        brake_rear_fraction=longitudinal.get("brake_rear_fraction", 0.5),
        longitudinal_enabled=longitudinal.get("enabled", True),
        compile_kernel=pd.get("compile", True),
    )


class RacingResetDraws(NamedTuple):
    """Sampled values for a batch of episode resets."""

    start_position: Any
    state_noise: Any
    physics_factors: Any


@dataclass(frozen=True)
class RacingRandomization:
    """Distributions for starting states and physics factors."""

    start_mode: str
    start_range: tuple[float, float]
    physics_names: tuple[str, ...]
    physics_bounds: Any

    @classmethod
    def from_config(
        cls,
        section: Mapping[str, Any],
        *,
        valid_physics_names: tuple[str, ...],
        default_start_range: tuple[float, float],
    ) -> "RacingRandomization":
        start_mode = str(section.get("start", "off")).strip().lower()
        if start_mode not in {"off", "uniform"}:
            raise ValueError("domain_randomization.start must be off or uniform")

        start_range_value = section.get("start_range", list(default_start_range))
        if (
            not isinstance(start_range_value, (tuple, list))
            or len(start_range_value) != 2
        ):
            raise ValueError("domain_randomization.start_range must contain two values")
        start_range = (float(start_range_value[0]), float(start_range_value[1]))
        if not start_range[0] < start_range[1]:
            raise ValueError("domain_randomization.start_range must be increasing")

        physics_value = section.get("physics", {})
        if not isinstance(physics_value, Mapping):
            raise TypeError("domain_randomization.physics must be a mapping")
        physics = dict(physics_value)
        allowed = set(valid_physics_names)
        unknown = set(physics) - allowed
        if unknown:
            raise ValueError(
                "domain_randomization.physics names are not local vehicle "
                f"parameters: {sorted(unknown)}"
            )
        names = tuple(sorted(physics))
        bounds_rows: list[list[float]] = []
        for name in names:
            value = physics[name]
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError(
                    "domain_randomization.physics bounds for "
                    f"{name!r} must contain two values"
                )
            low, high = float(value[0]), float(value[1])
            if not low <= high:
                raise ValueError(
                    "domain_randomization.physics bounds for "
                    f"{name!r} must be increasing"
                )
            bounds_rows.append([low, high])
        bounds = jnp.asarray(bounds_rows, dtype=jnp.float64) if names else None
        return cls(
            start_mode=start_mode,
            start_range=start_range,
            physics_names=names,
            physics_bounds=bounds,
        )

    def start_positions(
        self,
        key: Any,
        batch_size: int,
        *,
        evaluation: bool,
        initial_s: float,
        jitter: float,
    ) -> Any:
        """Sample starting positions from the configured distribution."""
        if evaluation:
            return jnp.full((batch_size,), initial_s, dtype=jnp.float64)
        if self.start_mode == "off":
            return initial_s + jax.random.uniform(
                key,
                (batch_size,),
                minval=-jitter,
                maxval=jitter,
                dtype=jnp.float64,
            )
        low, high = self.start_range
        return jax.random.uniform(
            key,
            (batch_size,),
            minval=low,
            maxval=high,
            dtype=jnp.float64,
        )

    def evaluation_physics(self, batch_size: int) -> Any:
        """Return nominal physics factors clipped to the configured bounds."""
        if self.physics_bounds is None:
            return None
        low = self.physics_bounds[:, 0]
        high = self.physics_bounds[:, 1]
        nominal = jnp.clip(jnp.ones_like(low), low, high)
        return jnp.broadcast_to(nominal, (int(batch_size), nominal.shape[0]))

    def sample_physics(self, key: Any, batch_size: int) -> Any:
        """Sample physics factors for each environment."""
        if self.physics_bounds is None:
            return None
        draw = jax.random.uniform(
            key,
            (batch_size, len(self.physics_names)),
            dtype=jnp.float64,
        )
        low = self.physics_bounds[:, 0]
        high = self.physics_bounds[:, 1]
        return low + draw * (high - low)


class RacingRuntime:
    action_dim = 3
    evaluation_sum_metrics = ("progress",)

    def __init__(self, config, env, problem=None, backend=None):
        self.env, self.problem, self.backend = env, problem, backend
        self.policy_mode = config["policy"]
        if self.policy_mode not in {"mpc", "direct"}:
            raise ValueError("policy must be mpc or direct")
        if (backend is None) != (self.policy_mode == "direct"):
            raise ValueError("MPC policies require a backend; direct policies do not")
        self.device = torch.device(config["training"].get("device", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; explicitly select device=cpu for a direct run"
            )
        self.dtype = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float64): torch.float64,
        }[np.dtype(env.continuous_dtype)]
        for name in (
            "max_episode_steps",
            "lap_start",
            "lap_length",
            "lateral_index",
            "speed_index",
            "stall_reset_speed",
            "off_track_grace_steps",
            "curvature_preview_scale",
        ):
            setattr(self, name, getattr(env, name))
        self.plan_nodes = env.N + 1
        self.state_scale, self.control_scale = (
            self._tensor(env.state_scale),
            self._tensor(env.control_scale),
        )
        self.control_low, self.control_high = (
            self._tensor(env.control_min),
            self._tensor(env.control_max),
        )
        for name in ("preview_path", "preview_curvature", "preview_speed"):
            setattr(self, name, self._tensor(getattr(env, name)))
        self.preview_offsets = torch.arange(
            env.N + 1, device=self.device, dtype=self.dtype
        ) * float(env.ds)
        settings = config["env"]
        self.initial_s = float(settings.get("initial_s", 37.4))
        self.reset_s_jitter = float(settings.get("reset_s_jitter", 0.0))
        self.reset_noise = jnp.asarray(
            settings.get("reset_state_noise", [0.0] * 6), dtype=jnp.float64
        )
        self.randomization = RacingRandomization.from_config(
            settings.get("domain_randomization", {}),
            valid_physics_names=tuple(env.physics_parameter_names),
            default_start_range=(0.0, self.lap_start + self.lap_length),
        )
        self.physics_factor_names = self.randomization.physics_names
        self.bound_violation_weight = float(settings["bound_violation_weight"])
        self.plant = PlantStep(env.transition)
        if backend is not None:

            def advance(iterate, previous, accepted, next_state, distance, done):
                history = backend.select_seed_rows(iterate, previous, accepted)
                return history, problem.advance(history, next_state, distance, done)

            self._advance = jax.jit(advance)
        self.mpc = MPC(backend) if backend else None
        self.controller = make_controller(config, env.dt) if backend is None else None
        self.pd_enabled = self.controller is not None
        self._context_includes_control = bool(
            problem and problem.constrain_initial_control
        )
        self.policy_parameter_dim = problem.cost_dim if problem else self.action_dim
        self.controller_memory_dim = (
            ReferencePDController.memory_dim if backend is None else 0
        )
        self.observation_dim = env.observation_dim + self.controller_memory_dim
        self.action_low = (
            self.control_low
            if backend
            else -torch.ones(self.action_dim, device=self.device, dtype=self.dtype)
        )
        self.action_high = self.control_high if backend else -self.action_low
        if backend is None:
            self.action_scale = self._tensor(config["controller"]["action_scale"])
            if self.action_scale.shape != (3,) or not bool(
                torch.all(torch.isfinite(self.action_scale) & (self.action_scale > 0))
            ):
                raise ValueError(
                    "controller.action_scale must contain three positive values"
                )
            scale = (
                self.controller.memory_scale(float(env.state_scale[SPEED]))
                if self.controller
                else (1.0,) * self.controller_memory_dim
            )
            self.controller_memory_scale = self._tensor(scale)
            columns = (
                "r",
                "V",
                "beta",
                "rear_wr",
                "delta_cmd",
                "engine_torque_cmd",
                "rear_brake_torque_cmd",
                "front_brake_torque_cmd",
            )
            self._reference_columns = {
                name: self._tensor(env.track.table[name]) for name in columns
            }
        anchor_state, anchor_control = env.reset(
            jnp.asarray(self.initial_s, dtype=jnp.float64),
            jnp.zeros(6, dtype=jnp.float64),
        )
        if backend:
            self._actor_anchor_observation = np.array(
                env.observe(anchor_state, anchor_control), copy=True
            )
            self._initial_policy_parameters = np.array(
                problem.initial_cost_logits(), copy=True
            )
        else:
            state = self._tensor(anchor_state)[None]
            applied = self._tensor(anchor_control)[None]
            steps = torch.zeros(1, device=self.device, dtype=torch.int64)
            anchor_state = RacingState(
                state,
                applied,
                steps,
                steps,
                controller_memory=self._initial_memory(state),
            )
            self._actor_anchor_observation = (
                self.observe(anchor_state)[0].detach().cpu().numpy()
            )
            self._initial_policy_parameters = np.zeros(
                3, dtype=np.float32 if self.dtype == torch.float32 else np.float64
            )

    def _tensor(self, value, *, dtype=None):
        return torch.as_tensor(
            np.array(value, copy=True), device=self.device, dtype=dtype or self.dtype
        )

    def actor_anchor_observation(self):
        return self._actor_anchor_observation

    def initial_policy_parameters(self):
        return self._initial_policy_parameters

    def policy_sigma_from_physical(self, sigma):
        return (
            np.asarray(sigma) / self.action_scale.cpu().numpy()
            if self.backend is None
            else np.asarray(sigma)
        )

    def evaluation_protocol(self):
        randomize = bool(self.randomization.physics_names)
        return EvaluationProtocol(
            {"evaluation": True, "from_start": True, "randomize_physics": randomize},
            {"evaluation": True, "randomize_physics": randomize},
            retire=True,
            fixed_seed=True,
        )

    def reset(
        self,
        batch_size,
        seed,
        *,
        evaluation=False,
        from_start=False,
        randomize_physics=None,
    ):
        state, applied, steps, physics = self._fresh_state(
            batch_size,
            seed,
            evaluation=evaluation,
            from_start=from_start,
            randomize_physics=randomize_physics,
        )
        warm = self.backend.zero_seed(batch_size) if self.backend else None
        return RacingState(
            state,
            applied,
            steps,
            torch.zeros_like(steps),
            physics,
            warm,
            warm,
            (
                torch.zeros(batch_size, device=self.device, dtype=self.dtype)
                if self.backend
                else None
            ),
            self._initial_memory(state) if self.backend is None else None,
        )

    def detach(self, state):
        return replace(
            state,
            **{
                key: value.detach()
                for key, value in vars(state).items()
                if isinstance(value, torch.Tensor)
            },
        )

    def reference_speed(self, path):
        return self._track_interpolate(self.preview_speed, path.detach())

    def _initial_memory(self, state):
        if self.controller is None:
            return state.new_zeros((*state.shape[:-1], self.controller_memory_dim))
        memory = self.controller.initial_memory(state)
        memory[..., 2] = (
            self.reference_speed(state[..., PATH_EVOLUTION]) - state[..., SPEED]
        )
        return memory

    def observe(self, state):
        path = state.state[..., PATH_EVOLUTION]
        curvature = self._track_interpolate(
            self.preview_curvature, path[..., None] + self.preview_offsets
        )
        if curvature.requires_grad:
            curvature.register_hook(zero_nonfinite_gradient)
        phase = 2.0 * torch.pi * (path - self.lap_start) / self.lap_length
        parts = [
            (state.state / self.state_scale)[..., :PATH_EVOLUTION],
            torch.sin(phase)[..., None],
            torch.cos(phase)[..., None],
            curvature / self.curvature_preview_scale,
            self._clip_physical(state.applied_control) / self.control_scale,
        ]
        if state.controller_memory is not None:
            parts.append(state.controller_memory / self.controller_memory_scale)
        return torch.cat(parts, dim=-1)

    def _clip_physical(self, value):
        return torch.maximum(torch.minimum(value, self.control_high), self.control_low)

    def squash_action(self, value):
        return torch.maximum(torch.minimum(value, self.action_high), self.action_low)

    def _policy_context(self, state):
        return (
            torch.cat([state.state, self._clip_physical(state.applied_control)], -1)
            if self._context_includes_control
            else state.state
        )

    def replay_inputs(self, state):
        if self.backend is None:
            return {}
        seed = jnp.concatenate(
            [
                self.backend.encode_seed(state.warm_seed),
                self.backend.encode_seed(state.last_successful_seed),
            ],
            axis=-1,
        )
        return {
            "context": self._policy_context(state).detach(),
            "seed": jax_to_torch(seed, like=state.state).detach(),
        }

    def _decode_inputs(self, inputs):
        context = inputs["context"]
        seed = torch_to_jax(inputs["seed"])
        warm, history = (
            self.backend.decode_seed(value) for value in jnp.split(seed, 2, axis=-1)
        )
        if self._context_includes_control:
            applied = torch_to_jax(context[:, -self.action_dim :])
            warm, history = (
                self.backend.set_seed_initial_control(value, applied)
                for value in (warm, history)
            )
        return context, warm, history

    def replay(self, parameters, inputs, *, retain):
        if self.backend is None:
            valid = torch.isfinite(parameters).all(-1)
            return torch.where(valid[:, None], parameters, 0.0), valid
        context, warm, history = self._decode_inputs(inputs)
        if retain:
            mean = self.mpc.mean_only(parameters, context, warm, history_seed=history)
            result = self.mpc.pop_result()
            valid = jax_to_torch(result.valid, like=parameters)
        else:
            parameters_jax, context_jax = torch_to_jax((parameters, context))
            result = self.backend.solve_device(
                parameters_jax,
                context_jax,
                warm,
                retain=False,
                history_seed=history,
            )
            mean, valid = jax_to_torch((result.mean, result.valid), like=parameters)
        valid = valid.bool()
        return torch.where(valid[:, None], mean, 0.0), valid

    def policy(self, parameters, state, *, retain, active=None, inputs=None):
        active = (
            torch.ones_like(state.steps, dtype=torch.bool) if active is None else active
        )
        if self.backend is None:
            return self._direct_policy(parameters, state, active)
        context, warm, history = (
            self._decode_inputs(inputs)
            if inputs is not None
            else (
                self._policy_context(state),
                state.warm_seed,
                state.last_successful_seed,
            )
        )
        if retain:
            mean, controls = self.mpc(parameters, context, warm, history_seed=history)
            result = self.mpc.pop_result()
            valid, iterations = jax_to_torch(
                (result.valid, result.stats["inner_iterations"]), like=parameters
            )
        else:
            parameters_jax, context_jax, active_jax = torch_to_jax(
                (parameters, context, active)
            )
            result = self.backend.solve_device(
                parameters_jax,
                context_jax,
                warm,
                retain=False,
                history_seed=history,
                active=active_jax,
            )
            mean, controls, valid, iterations = jax_to_torch(
                (
                    result.mean,
                    result.controls,
                    result.valid,
                    result.stats["inner_iterations"],
                ),
                like=parameters,
            )
        valid = valid.bool()
        metrics = {
            "admm_iterations": iterations,
            "admm_cap_hit_fraction": iterations >= self.backend.admm_max_iterations,
        }
        return Policy(mean, controls, valid, metrics, result=result)

    def _direct_policy(self, parameters, state, active):
        finite = (
            torch.isfinite(parameters).all(-1)
            & torch.isfinite(state.state).all(-1)
            & torch.isfinite(state.controller_memory).all(-1)
        )
        mean = torch.where(torch.isfinite(parameters).all(-1)[:, None], parameters, 0.0)
        safe_state = torch.where(
            torch.isfinite(state.state).all(-1)[:, None], state.state, 0.0
        )
        safe_memory = torch.where(
            torch.isfinite(state.controller_memory).all(-1)[:, None],
            state.controller_memory,
            0.0,
        )
        query = safe_state[..., PATH_EVOLUTION]
        reference = torch.stack(
            [
                *(
                    self._track_interpolate(self._reference_columns[name], query)
                    for name in ("r", "V", "beta", "rear_wr")
                ),
                *([torch.zeros_like(query)] * 3),
            ],
            -1,
        )
        controls = torch.stack(
            [
                self._track_interpolate(self._reference_columns["delta_cmd"], query),
                self._track_interpolate(
                    self._reference_columns["engine_torque_cmd"], query
                )
                + self._track_interpolate(
                    self._reference_columns["rear_brake_torque_cmd"], query
                ),
                self._track_interpolate(
                    self._reference_columns["front_brake_torque_cmd"], query
                ),
            ],
            -1,
        )
        output = (
            self.controller(
                safe_state, reference, self._clip_physical(controls), safe_memory
            )
            if self.controller
            else None
        )
        prior = output.control if output else torch.zeros_like(mean)
        memory = output.memory if output else safe_memory
        valid = (
            active
            & finite
            & torch.isfinite(prior).all(-1)
            & torch.isfinite(memory).all(-1)
        )
        return Policy(
            mean,
            mean[:, None].repeat(1, self.plan_nodes, 1),
            valid,
            {},
            pd_control=prior,
            controller_memory=memory,
        )

    def step(self, state, action, policy, *, active=None):
        active = (
            torch.ones_like(state.steps, dtype=torch.bool) if active is None else active
        )
        accepted = active & policy.valid
        if self.backend:
            executed = torch.where(accepted[:, None], action, policy.controls[:, 0])
            controls = policy.controls.clone()
            controls[:, 0] = executed
            input_valid = torch.ones_like(active)
            saturated = torch.zeros_like(action, dtype=torch.bool)
        else:
            action_finite = torch.isfinite(action).all(-1)
            learned = self.action_scale * self.squash_action(
                torch.where(action_finite[:, None], action, 0.0)
            )
            prior = policy.pd_control
            composition_finite = torch.isfinite(learned).all(-1) & torch.isfinite(
                prior
            ).all(-1)
            unclipped = torch.where(
                composition_finite[:, None], learned, 0.0
            ) + torch.where(composition_finite[:, None], prior, 0.0)
            command = self._clip_physical(unclipped)
            controls = command[:, None].repeat(1, self.plan_nodes, 1)
            saturated = (unclipped < self.control_low) | (unclipped > self.control_high)
            input_valid = policy.valid & action_finite & composition_finite
        (
            next_state,
            task_reward,
            bound_violation,
            _stall,
            off_low,
            off_high,
            spin_out,
            applied,
        ) = self.plant(
            state.state,
            controls,
            physics=self._physics_for_env(state.physics),
        )
        valid = (
            active
            & input_valid
            & torch.isfinite(next_state).all(-1)
            & torch.isfinite(task_reward)
            & torch.isfinite(bound_violation)
        )
        if self.backend is None:
            valid = valid & torch.isfinite(applied).all(-1)
        off_track = off_low | off_high
        off_count = torch.where(
            off_track,
            state.off_track_count + 1,
            torch.zeros_like(state.off_track_count),
        )
        off_expired = off_count > max(int(self.off_track_grace_steps), 0)
        stall_reset = (
            next_state[..., SPEED].detach() < self.stall_reset_speed
            if self.stall_reset_speed > 0
            else torch.zeros_like(active)
        )
        failure = active & (stall_reset | spin_out | off_expired)
        timeout = (
            active & valid & ((state.steps + 1 >= self.max_episode_steps) | failure)
        )
        done = timeout | (active & ~valid)
        reward = (
            task_reward
            - self.bound_violation_weight * bound_violation
            - float(self.env.failure_penalty) * (failure & valid).to(task_reward.dtype)
        )
        reward = torch.where(
            valid,
            reward,
            reward.new_full(reward.shape, -float(self.env.failure_penalty)),
        )
        reward = torch.where(active, reward, 0.0)
        progress = (
            next_state[..., PATH_EVOLUTION] - state.state[..., PATH_EVOLUTION]
        ).detach()
        next_state = torch.where(active[:, None], next_state, state.state)
        applied = torch.where(active[:, None], applied, state.applied_control)
        progress = torch.where(active, progress, 0.0)
        new = replace(
            state,
            state=next_state,
            applied_control=applied.detach(),
            steps=torch.where(active, state.steps + 1, state.steps),
            off_track_count=torch.where(
                active, off_count, state.off_track_count
            ).detach(),
        )
        if self.backend:
            distance = torch.where(
                accepted, progress, state.progress_since_success + progress
            )
            distance = torch.where(active, distance, state.progress_since_success)
            accepted_jax, state_jax, distance_jax, done_jax = torch_to_jax(
                (accepted, next_state, distance, done | ~active)
            )
            history, warm = self._advance(
                policy.result.iterate,
                state.last_successful_seed,
                accepted_jax,
                state_jax,
                distance_jax,
                done_jax,
            )
            new = replace(
                new,
                warm_seed=warm,
                last_successful_seed=history,
                progress_since_success=distance.detach(),
            )
        else:
            new = replace(
                new,
                controller_memory=torch.where(
                    valid[:, None], policy.controller_memory, state.controller_memory
                ),
            )
        metrics = {
            **policy.metrics,
            "progress": progress,
            "abs_lateral_deviation": next_state[..., self.lateral_index].abs().detach(),
            "abs_velocity_deviation": (
                next_state[..., SPEED].detach()
                - self.reference_speed(next_state[..., PATH_EVOLUTION])
            ).abs(),
            "stall_reset": stall_reset & done,
            "off_track": off_expired & done,
            "spin_out": spin_out & done,
            "invalid": active & ~valid,
            "mpc_solve_failed": (
                active & ~accepted if self.backend else torch.zeros_like(active)
            ),
            "physical_saturation_fraction": saturated.to(self.dtype).mean(-1),
        }
        return Transition(new, self.observe(new), reward, done, timeout, valid, metrics)

    def reset_done(
        self, state, done, seed, *, evaluation=False, randomize_physics=None
    ):
        if not bool(done.any()):
            return state
        fresh, applied, steps, physics = self._fresh_state(
            done.shape[0],
            seed,
            evaluation=evaluation,
            randomize_physics=randomize_physics,
        )
        new = replace(
            state,
            state=torch.where(done[:, None], fresh.detach(), state.state),
            applied_control=torch.where(
                done[:, None], applied.detach(), state.applied_control
            ),
            steps=torch.where(done, steps, state.steps),
            off_track_count=torch.where(
                done, torch.zeros_like(state.off_track_count), state.off_track_count
            ),
        )
        if physics is not None:
            new.physics = jnp.where(torch_to_jax(done)[:, None], physics, state.physics)
        if self.backend:
            new.warm_seed = self.backend.invalidate_seed_rows(
                state.warm_seed, torch_to_jax(done)
            )
            new.last_successful_seed = self.backend.invalidate_seed_rows(
                state.last_successful_seed, torch_to_jax(done)
            )
            new.progress_since_success = torch.where(
                done, 0.0, state.progress_since_success
            )
        else:
            new.controller_memory = torch.where(
                done[:, None],
                self._initial_memory(fresh).detach(),
                state.controller_memory,
            )
        return new

    def trace_frame(self, state, transition, policy):
        return {
            "state": transition.state.state.detach().cpu().numpy(),
            "control": transition.state.applied_control.detach().cpu().numpy(),
            "reward": transition.reward.detach().cpu().numpy(),
        }

    def _sample_reset_draws(
        self,
        seed: int,
        batch_size: int,
        *,
        evaluation: bool,
        from_start: bool,
        randomize_physics: bool | None = None,
    ) -> RacingResetDraws:
        """Sample starting positions, state noise, and physics factors."""
        key = jax.random.PRNGKey(int(seed))
        key_s, key_x, key_p = jax.random.split(key, 3)
        if from_start:
            start_position = jnp.full((batch_size,), self.initial_s, dtype=jnp.float64)
        else:
            start_position = self.randomization.start_positions(
                key_s,
                batch_size,
                evaluation=evaluation,
                initial_s=self.initial_s,
                jitter=self.reset_s_jitter,
            )
        state_noise = jax.random.uniform(
            key_x,
            (batch_size, 6),
            minval=-self.reset_noise,
            maxval=self.reset_noise,
            dtype=jnp.float64,
        )
        if from_start:
            state_noise = jnp.zeros_like(state_noise)
        if randomize_physics is None:
            randomize_physics = not evaluation
        physics_factors = (
            self.randomization.sample_physics(key_p, batch_size)
            if randomize_physics
            else self.randomization.evaluation_physics(batch_size)
        )
        return RacingResetDraws(
            start_position=start_position,
            state_noise=state_noise,
            physics_factors=physics_factors,
        )

    def _fresh_state(
        self,
        batch_size: int,
        seed: int,
        *,
        evaluation: bool,
        from_start: bool = False,
        randomize_physics: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        draws = self._sample_reset_draws(
            seed,
            int(batch_size),
            evaluation=evaluation,
            from_start=from_start,
            randomize_physics=randomize_physics,
        )
        state, applied = self.env.reset(draws.start_position, draws.state_noise)
        steps = torch.zeros(
            (int(batch_size),),
            device=self.device,
            dtype=torch.int64,
        )
        return self._tensor(state), self._tensor(applied), steps, draws.physics_factors

    def _physics_for_env(self, factors: Any) -> Mapping[str, Any] | None:
        """Convert the factor matrix to a dynamics parameter pytree."""
        if factors is None:
            return None
        factors = jnp.asarray(factors)
        return {
            name: factors[..., index]
            for index, name in enumerate(self.physics_factor_names)
        }

    def _track_interpolate(
        self,
        values: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        query = torch.where(
            query < self.lap_start + self.lap_length,
            query,
            self.lap_start + torch.remainder(query - self.lap_start, self.lap_length),
        )
        query = query.clamp(self.preview_path[0], self.preview_path[-1]).contiguous()
        upper = torch.searchsorted(self.preview_path, query, right=True).clamp(
            1, self.preview_path.numel() - 1
        )
        lower = upper - 1
        path_lower = self.preview_path[lower]
        path_upper = self.preview_path[upper]
        fraction = (query - path_lower) / (path_upper - path_lower)
        return values[lower] + fraction * (values[upper] - values[lower])
