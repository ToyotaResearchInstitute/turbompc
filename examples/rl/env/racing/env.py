"""Racing rewards and transitions using the shared vehicle model.

This code uses synthetic data and parameters, and does not represent any manufacturer's product.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .dynamics import (
    VEHICLE_PARAMETERS,
    RacingSimulationEnvironment,
    interpolate_control,
)

# Local TimeVehicleDynamics state layout.
YAW_RATE = 0
SPEED = 1
SIDESLIP = 2
WHEEL_SPEED = 3
LATERAL_DEVIATION = 4
RELATIVE_HEADING = 5
PATH_EVOLUTION = 6
PLANT_NX = 7

# The local path-domain OCP has the same dimension and replaces the final
# physical path coordinate with elapsed time.
MPC_NX = 7
MPC_LATERAL_DEVIATION = LATERAL_DEVIATION
NU = 3
PLANT_NU = NU

LAP_START = 293.0
LAP_LENGTH = 258.5
CURVATURE_PREVIEW_SCALE = 0.05

# Local example actuator box: steering [rad], merged signed rear torque and
# front brake [Nm].
CONTROL_MIN = jnp.asarray([-0.5, -4.0e3, -5.0e3], dtype=jnp.float64)
CONTROL_MAX = jnp.asarray([0.5, 370.0, 0.0], dtype=jnp.float64)
CONTROL_SCALE = jnp.asarray([0.5, 5.0e3, 5.0e3], dtype=jnp.float64)
MPC_STATE_SCALE = jnp.asarray([1.0, 30.0, 0.5, 30.0, 5.0, 1.0, 10.0], dtype=jnp.float64)
OBS_STATE_SCALE = MPC_STATE_SCALE[:PATH_EVOLUTION]

_TRAJECTORY_COLUMNS = (
    "s",
    "k",
    "V",
    "r",
    "beta",
    "rear_wr",
    "delta_cmd",
    "engine_torque_cmd",
    "rear_brake_torque_cmd",
    "front_brake_torque_cmd",
    "E",
    "N",
    "phi",
    "e_min",
    "e_max",
)
_TRAJECTORY_PATH = Path(__file__).resolve().parents[3] / "racing_trajectory.csv"
_trajectory_numpy = np.genfromtxt(_TRAJECTORY_PATH, delimiter=",", names=True)
TRAJECTORY = {
    name: jnp.asarray(_trajectory_numpy[name], dtype=jnp.float64)
    for name in _TRAJECTORY_COLUMNS
}

# The table contains an out-lap and repeated steady laps.  Guard these
# constants against silent changes in the local example data.
_lap_grid = np.linspace(LAP_START, LAP_START + LAP_LENGTH, 512)
for _column in ("E", "N", "V"):
    _lap_mismatch = np.abs(
        np.interp(
            _lap_grid,
            _trajectory_numpy["s"],
            _trajectory_numpy[_column],
        )
        - np.interp(
            _lap_grid + LAP_LENGTH,
            _trajectory_numpy["s"],
            _trajectory_numpy[_column],
        )
    ).max()
    if _lap_mismatch > 0.05:
        raise RuntimeError(
            "racing_trajectory.csv steady-lap copies disagree on "
            f"{_column!r} by {_lap_mismatch:.3f}; LAP_START/LAP_LENGTH "
            "no longer match the local example table"
        )


def wrap_path_evolution(path_evolution):
    """Wrap positions beyond the first steady lap."""
    path_evolution = jnp.asarray(path_evolution)
    wrapped = LAP_START + jnp.mod(path_evolution - LAP_START, LAP_LENGTH)
    return jnp.where(
        path_evolution < LAP_START + LAP_LENGTH,
        path_evolution,
        wrapped,
    )


class Track:
    """Track references with an initial out-lap."""

    def __init__(self, table: Mapping[str, Any] | None = None):
        source = TRAJECTORY if table is None else table
        self.table = {name: jnp.asarray(values) for name, values in source.items()}
        self.lap_start = LAP_START
        self.lap_length = LAP_LENGTH
        self.max_s = float(self.table["s"][-1])

    def wrap(self, path_evolution):
        return wrap_path_evolution(path_evolution)

    def interpolate(self, path_evolution, name: str):
        """Interpolate a reference column within the table endpoints."""
        if name not in self.table:
            raise KeyError(f"unknown racing trajectory column: {name}")
        query = self.wrap(path_evolution)
        query = jnp.clip(query, self.table["s"][0], self.table["s"][-1])
        return jnp.interp(query, self.table["s"], self.table[name])

    def references_at(self, path_evolution):
        """Return the reference states and controls."""
        path_evolution = jnp.asarray(path_evolution)
        zero = jnp.zeros_like(path_evolution)
        states = jnp.stack(
            [
                self.interpolate(path_evolution, "r"),
                self.interpolate(path_evolution, "V"),
                self.interpolate(path_evolution, "beta"),
                self.interpolate(path_evolution, "rear_wr"),
                zero,
                zero,
                zero,
            ],
            axis=-1,
        )
        controls = jnp.stack(
            [
                self.interpolate(path_evolution, "delta_cmd"),
                self.interpolate(path_evolution, "engine_torque_cmd")
                + self.interpolate(path_evolution, "rear_brake_torque_cmd"),
                self.interpolate(path_evolution, "front_brake_torque_cmd"),
            ],
            axis=-1,
        )
        return states, jnp.clip(controls, CONTROL_MIN, CONTROL_MAX)

    def state_bounds(self, path_evolution, minimum, maximum):
        """Combine state bounds with the track corridor."""
        path_evolution = jnp.asarray(path_evolution)
        output_shape = path_evolution.shape + (MPC_NX,)
        minimum = jnp.broadcast_to(jnp.asarray(minimum), output_shape)
        maximum = jnp.broadcast_to(jnp.asarray(maximum), output_shape)
        minimum = minimum.at[..., MPC_LATERAL_DEVIATION].set(
            self.interpolate(path_evolution, "e_min")
        )
        maximum = maximum.at[..., MPC_LATERAL_DEVIATION].set(
            self.interpolate(path_evolution, "e_max")
        )
        return minimum, maximum

    def corridor(self, path_evolution):
        return (
            self.interpolate(path_evolution, "e_min"),
            self.interpolate(path_evolution, "e_max"),
        )

    def curvature(self, path_evolution):
        return self.interpolate(path_evolution, "k")


def normalized_bound_violation_squared(
    states,
    lateral_minimum,
    lateral_maximum,
    minimum_speed,
    scale,
    maximum_heading,
):
    """Compute the squared normalized penalty for bound violations."""
    states = jnp.asarray(states)
    scale = jnp.asarray(scale)
    speed_below = jax.nn.relu(
        (jnp.asarray(minimum_speed) - states[..., SPEED]) / scale[SPEED]
    )
    lateral_below = jax.nn.relu(
        (jnp.asarray(lateral_minimum) - states[..., LATERAL_DEVIATION])
        / scale[LATERAL_DEVIATION]
    )
    lateral_above = jax.nn.relu(
        (states[..., LATERAL_DEVIATION] - jnp.asarray(lateral_maximum))
        / scale[LATERAL_DEVIATION]
    )
    heading_over = jax.nn.relu(
        (jnp.abs(states[..., RELATIVE_HEADING]) - maximum_heading)
        / scale[RELATIVE_HEADING]
    )
    return (
        jnp.square(speed_below)
        + jnp.square(lateral_below)
        + jnp.square(lateral_above)
        + jnp.square(heading_over)
    )


class RacingEnv:
    """Racing simulation and task rewards."""

    # Historical minimal SHAC rollouts quantize state/reward/next-observation
    # at the JAX/Torch bridge after every transition.
    continuous_dtype = np.dtype(np.float32)
    action_dim = NU
    lateral_index = LATERAL_DEVIATION
    speed_index = SPEED
    physics_parameter_names = tuple(VEHICLE_PARAMETERS)

    def __init__(self, config: Mapping[str, Any]):
        try:
            env = config["env"]
        except KeyError as error:
            raise ValueError("racing_minimal requires an env configuration") from error
        if not isinstance(env, Mapping):
            raise TypeError("env configuration must be a mapping")
        self.dt = float(env.get("dt", 0.1))
        self.sim_dt = float(env.get("sim_dt", self.dt))
        self.sim_steps_per_plan = round(self.dt / self.sim_dt)
        if abs(self.sim_steps_per_plan * self.sim_dt - self.dt) > 1.0e-9:
            raise ValueError("env.dt must be an integer multiple of sim_dt")
        self.ds = float(env.get("ds", 3.0))
        self.N = int(env.get("horizon", 32))
        self.simulator_substeps = int(env.get("simulator_substeps", 1))
        self.simulator_total_substeps = (
            self.sim_steps_per_plan * self.simulator_substeps
        )
        self.max_episode_steps = int(env.get("max_episode_steps", 300))
        if self.dt <= 0.0 or self.ds <= 0.0 or self.N < 1:
            raise ValueError("dt, ds, and horizon must be positive")
        if self.simulator_substeps < 1 or self.max_episode_steps < 1:
            raise ValueError(
                "simulator_substeps and max_episode_steps must be positive"
            )
        self.progress_weight = float(env.get("progress_weight", 1.0))
        self.progress_reward_mode = (
            str(env.get("progress_reward_mode", "linear")).strip().lower()
        )
        if self.progress_reward_mode not in {"linear", "reference_linear"}:
            raise ValueError(
                "env.progress_reward_mode must be 'linear' or 'reference_linear'"
            )
        self.tracking_weight = float(env.get("tracking_weight", 1.0))
        tracking_scales = np.asarray(env["tracking_weight_scales"], dtype=np.float64)
        if (
            tracking_scales.shape != (PATH_EVOLUTION,)
            or not np.all(np.isfinite(tracking_scales))
            or np.any(tracking_scales <= 0.0)
        ):
            raise ValueError(
                "env.tracking_weight_scales must contain six positive finite values"
            )
        self.tracking_weight_scales = jnp.asarray(tracking_scales)
        self.failure_penalty = float(env.get("failure_penalty", 50.0))
        self.failure_min_speed = float(env.get("failure_min_speed", 5.0))
        self.failure_max_heading = float(env.get("failure_max_heading", np.pi / 3.0))
        self.stall_reset_speed = float(env.get("stall_reset_speed", 0.0))
        self.off_track_grace_steps = int(env.get("off_track_grace_steps", 0))
        if self.off_track_grace_steps < 0:
            raise ValueError("env.off_track_grace_steps must be >= 0")
        if not np.isfinite(self.failure_min_speed):
            raise ValueError("env.failure_min_speed must be finite")
        if not np.isfinite(self.failure_max_heading):
            raise ValueError("env.failure_max_heading must be finite")
        if not np.isfinite(self.stall_reset_speed) or self.stall_reset_speed < 0.0:
            raise ValueError("env.stall_reset_speed must be finite and >= 0")

        self.control_min = jnp.asarray(
            env.get("control_min_bounds", CONTROL_MIN), dtype=jnp.float64
        )
        self.control_max = jnp.asarray(
            env.get("control_max_bounds", CONTROL_MAX), dtype=jnp.float64
        )
        if self.control_min.shape != (NU,) or self.control_max.shape != (NU,):
            raise ValueError(f"env.control_min/max_bounds must have shape ({NU},)")
        if bool(jnp.any(self.control_min >= self.control_max)):
            raise ValueError("env control bounds must satisfy min < max")
        self.state_min = jnp.asarray(
            env.get(
                "state_min_bounds",
                [-1.0e9, 0.5, -1.0e9, -1.0e9, -4.0, -1.0e9, -1.0e9],
            ),
            dtype=jnp.float64,
        )
        self.state_max = jnp.asarray(
            env.get(
                "state_max_bounds",
                [1.0e9, 30.0, 1.0e9, 1.0e9, 4.0, 1.0e9, 1.0e9],
            ),
            dtype=jnp.float64,
        )
        if self.state_min.shape != (PLANT_NX,) or self.state_max.shape != (PLANT_NX,):
            raise ValueError(f"env.state_min/max_bounds must have shape ({PLANT_NX},)")
        if bool(jnp.any(self.state_min >= self.state_max)):
            raise ValueError("env state bounds must satisfy min < max")

        self.track = Track()
        self.lap_start = self.track.lap_start
        self.lap_length = self.track.lap_length
        self.track_length = self.track.max_s
        self.preview_path = self.track.table["s"]
        self.preview_curvature = self.track.table["k"]
        self.preview_speed = self.track.table["V"]
        self.curvature_preview_scale = CURVATURE_PREVIEW_SCALE
        self.state_scale = jnp.concatenate(
            [OBS_STATE_SCALE, jnp.asarray([self.track_length])]
        )
        self.control_scale = CONTROL_SCALE
        self.observation_dim = PLANT_NX + NU + self.N + 2
        self.reset_min_speed = float(env.get("reset_min_speed", 0.0))
        hard_clamp = env.get("lateral_hard_clamp")
        self.lateral_hard_clamp = None if hard_clamp is None else float(hard_clamp)
        if self.lateral_hard_clamp is not None and self.lateral_hard_clamp <= 0:
            raise ValueError("env.lateral_hard_clamp must be positive")
        self.simulator = RacingSimulationEnvironment(
            dt=self.dt,
            substeps=self.simulator_total_substeps,
            curvature_at=self.track.curvature,
        )

    def clip_control(self, controls):
        return jnp.clip(jnp.asarray(controls), self.control_min, self.control_max)

    def path_knots(self, plant_state):
        plant_state = jnp.asarray(plant_state)
        return plant_state[..., PATH_EVOLUTION, None] + self.ds * jnp.arange(self.N + 1)

    def observe(self, plant_state, applied_control):
        """Build normalized observations with track preview and lap phase."""
        plant_state = jnp.asarray(plant_state)
        scaled = plant_state / self.state_scale
        phase = (
            2.0
            * jnp.pi
            * (plant_state[..., PATH_EVOLUTION] - self.lap_start)
            / self.lap_length
        )
        state = jnp.concatenate(
            [
                scaled[..., :PATH_EVOLUTION],
                jnp.sin(phase)[..., None],
                jnp.cos(phase)[..., None],
            ],
            axis=-1,
        )
        curvature_preview = (
            self.track.curvature(self.path_knots(plant_state)) / CURVATURE_PREVIEW_SCALE
        )
        control = self.clip_control(applied_control) / self.control_scale
        return jnp.concatenate([state, curvature_preview, control], axis=-1)

    def reset(self, start_position, state_noise):
        """Initialize state and control from the supplied reset samples."""
        start = jnp.asarray(start_position, dtype=jnp.float64)
        state, control = self._plant_state_at(start)
        noise = jnp.asarray(state_noise, dtype=jnp.float64)
        expected_shape = state.shape[:-1] + (PATH_EVOLUTION,)
        if noise.shape != expected_shape:
            raise ValueError(
                f"state_noise must have shape {expected_shape}, got {noise.shape}"
            )
        state = state.at[..., :PATH_EVOLUTION].add(noise)
        return state, control

    def transition(
        self,
        state,
        planned_controls,
        physics_factors: Mapping[str, Any] | None = None,
    ):
        """Return (next state, reward, violation), failure flags, and control."""
        states = jnp.asarray(state)
        controls = self.clip_control(planned_controls)
        knots = self.path_knots(states)
        # Preserve the local training environment's established nested
        # projection: the transition clips the public plan, then the plant
        # receives a separately projected copy.  Values are unchanged, but
        # the second projection is part of the historical gradient exactly
        # on an actuator face.
        simulator_controls = self.clip_control(controls)
        next_states = self.simulator.step(
            states, simulator_controls, knots, physics_factors
        )
        if self.lateral_hard_clamp is not None:
            next_states = next_states.at[..., LATERAL_DEVIATION].set(
                jnp.clip(
                    next_states[..., LATERAL_DEVIATION],
                    -self.lateral_hard_clamp,
                    self.lateral_hard_clamp,
                )
            )
        applied = interpolate_control(next_states[..., PATH_EVOLUTION], knots, controls)

        progress = next_states[..., PATH_EVOLUTION] - states[..., PATH_EVOLUTION]
        if self.progress_reward_mode == "reference_linear":
            reference_start, _ = self.reference_trajectory_at(
                states[..., PATH_EVOLUTION]
            )
            reference_progress = self.dt * reference_start[..., SPEED]
            progress = progress - reference_progress
        progress_reward = self.progress_weight * progress
        reference, _ = self.reference_trajectory_at(next_states[..., PATH_EVOLUTION])
        normalized_error = (
            next_states[..., :PATH_EVOLUTION] - reference[..., :PATH_EVOLUTION]
        ) / self.tracking_weight_scales
        reference_error = jnp.mean(jnp.square(normalized_error), axis=-1)

        lateral_minimum, lateral_maximum = self.track.corridor(
            next_states[..., PATH_EVOLUTION]
        )
        stall, off_track_low, off_track_high, spin_out = self.failure_masks(
            next_states, lateral_minimum, lateral_maximum
        )
        bound_violation_squared = normalized_bound_violation_squared(
            next_states,
            lateral_minimum,
            lateral_maximum,
            self.failure_min_speed,
            self.state_scale,
            maximum_heading=(
                self.failure_max_heading if self.failure_max_heading > 0.0 else jnp.inf
            ),
        )
        task_reward = progress_reward - self.tracking_weight * reference_error
        return (
            (next_states, task_reward, bound_violation_squared),
            (stall, off_track_low, off_track_high, spin_out),
            applied,
        )

    def failure_masks(self, states, lateral_minimum, lateral_maximum):
        stall = states[..., SPEED] < self.failure_min_speed
        off_track_low = states[..., LATERAL_DEVIATION] < lateral_minimum
        off_track_high = states[..., LATERAL_DEVIATION] > lateral_maximum
        spin_out = (
            jnp.abs(states[..., RELATIVE_HEADING]) > self.failure_max_heading
            if self.failure_max_heading > 0.0
            else jnp.zeros_like(stall)
        )
        return stall, off_track_low, off_track_high, spin_out

    def _plant_state_at(self, path_evolution):
        path_evolution = jnp.asarray(path_evolution)
        reference, control = self.reference_trajectory_at(path_evolution)
        state = reference.at[..., PATH_EVOLUTION].set(path_evolution)
        if self.reset_min_speed > 0.0:
            state = state.at[..., SPEED].set(
                jnp.maximum(state[..., SPEED], self.reset_min_speed)
            )
            state = state.at[..., WHEEL_SPEED].set(
                jnp.maximum(state[..., WHEEL_SPEED], self.reset_min_speed)
            )
        return state, control

    def reference_trajectory_at(self, path_evolution):
        states, controls = self.track.references_at(path_evolution)
        return states, self.clip_control(controls)


__all__ = [
    "CONTROL_MAX",
    "CONTROL_MIN",
    "CONTROL_SCALE",
    "CURVATURE_PREVIEW_SCALE",
    "LAP_LENGTH",
    "LAP_START",
    "LATERAL_DEVIATION",
    "MPC_LATERAL_DEVIATION",
    "MPC_NX",
    "MPC_STATE_SCALE",
    "NU",
    "OBS_STATE_SCALE",
    "PATH_EVOLUTION",
    "PLANT_NU",
    "PLANT_NX",
    "RELATIVE_HEADING",
    "SIDESLIP",
    "SPEED",
    "TRAJECTORY",
    "VEHICLE_PARAMETERS",
    "WHEEL_SPEED",
    "YAW_RATE",
    "RacingEnv",
    "Track",
    "normalized_bound_violation_squared",
    "wrap_path_evolution",
]
