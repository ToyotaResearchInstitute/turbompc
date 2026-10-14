"""Batched simulation with the shared racing vehicle model.

This code uses synthetic data and parameters, and does not represent any manufacturer's product.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.resources import files
from typing import Any

import diffrax
import jax
import jax.numpy as jnp

from turbompc.dynamics.vehicle import TimeVehicleDynamics
from turbompc.utils.load_params import load_params

NX = 7
NU = 3
PATH_EVOLUTION = 6

# Share the demo's nominal parameters; curvature comes from the current track.
VEHICLE_PARAMETERS = load_params(files("turbompc") / "problems/params/racing.yaml")[
    "dynamics_state_dot_params"
]
VEHICLE_PARAMETERS.pop("curvature")

# Diffrax's Tsit5 tableau, represented as Python constants so a fixed number
# of steps can be expressed by one differentiable lax.scan.
_TSIT5_A = tuple(
    tuple(float(value) for value in row) for row in diffrax.Tsit5.tableau.a_lower
)
_TSIT5_B = tuple(float(value) for value in diffrax.Tsit5.tableau.b_sol)


def fixed_step_tsit5(state_dot, initial_state, duration: float, steps: int):
    """Integrate with a fixed number of Tsit5 steps."""
    if int(steps) < 1:
        raise ValueError("Tsit5 steps must be positive")
    step_size = float(duration) / int(steps)
    initial_state = jnp.asarray(initial_state)
    initial_state = initial_state.astype(
        jnp.result_type(initial_state.dtype, state_dot(initial_state).dtype)
    )

    def integrate_step(state, _):
        stages = [state_dot(state)]
        # The last Tsit5 stage is FSAL and has zero solution weight.
        for row in _TSIT5_A[:-1]:
            increment = sum(
                coefficient * stage for coefficient, stage in zip(row, stages)
            )
            stages.append(state_dot(state + step_size * increment))
        next_state = state + step_size * sum(
            weight * stage for weight, stage in zip(_TSIT5_B, stages)
        )
        return next_state, None

    final_state, _ = jax.lax.scan(
        integrate_step, initial_state, None, length=int(steps)
    )
    return final_state


def interpolate_control(path_evolution, path_knots, planned_controls):
    """Interpolate a spatial control plan at batched positions."""
    path_evolution = jnp.asarray(path_evolution)
    path_knots = jnp.asarray(path_knots)
    planned_controls = jnp.asarray(planned_controls)

    def one(position, knots, controls):
        return jax.vmap(jnp.interp, in_axes=(None, None, 0))(
            position, knots, controls.T
        )

    if path_evolution.ndim == 0:
        return one(path_evolution, path_knots, planned_controls)
    values = jax.vmap(one)(
        path_evolution.reshape(-1),
        path_knots.reshape(-1, path_knots.shape[-1]),
        planned_controls.reshape(-1, *planned_controls.shape[-2:]),
    )
    return values.reshape(*path_evolution.shape, NU)


class RacingSimulationEnvironment:
    """Simulate the racing vehicle with spatially interpolated controls."""

    def __init__(
        self,
        *,
        dt: float,
        substeps: int,
        curvature_at: Callable[[Any], Any],
        vehicle_parameters: Mapping[str, float] | None = None,
    ):
        if float(dt) <= 0.0:
            raise ValueError("simulation dt must be positive")
        if int(substeps) < 1:
            raise ValueError("simulation substeps must be positive")
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.curvature_at = curvature_at
        self.vehicle_parameters = dict(
            VEHICLE_PARAMETERS if vehicle_parameters is None else vehicle_parameters
        )
        missing = set(VEHICLE_PARAMETERS) - set(self.vehicle_parameters)
        if missing:
            raise ValueError(f"missing vehicle parameters: {sorted(missing)}")
        self.model = TimeVehicleDynamics()

    def validate_physics_factors(
        self,
        physics_factors: Mapping[str, Any] | None,
    ) -> None:
        """Validate physics factor names."""
        if physics_factors is None:
            return
        if not isinstance(physics_factors, Mapping):
            raise TypeError(
                "physics_factors must be a mapping from vehicle parameter "
                "name to multiplicative factor"
            )
        unknown = set(physics_factors) - set(self.vehicle_parameters)
        if unknown:
            raise ValueError(f"unknown vehicle physics factor names: {sorted(unknown)}")

    def parameters_at(
        self,
        path_evolution,
        physics_factors: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply physics factors to the nominal dynamics parameters."""
        parameters: dict[str, Any] = {
            **self.vehicle_parameters,
            "curvature": self.curvature_at(path_evolution),
        }
        if physics_factors is not None:
            for name, factor in physics_factors.items():
                parameters[name] = parameters[name] * jnp.asarray(factor)
        return parameters

    def step_one(
        self,
        state,
        planned_controls,
        path_knots,
        physics_factors: Mapping[str, Any] | None = None,
    ):
        """Advance one vehicle using its spatial control plan."""
        state = jnp.asarray(state)
        planned_controls = jnp.asarray(planned_controls)
        path_knots = jnp.asarray(path_knots)

        def state_dot(current_state):
            control = interpolate_control(
                current_state[PATH_EVOLUTION],
                path_knots,
                planned_controls,
            )
            return self.model.state_dot(
                current_state,
                control,
                self.parameters_at(current_state[PATH_EVOLUTION], physics_factors),
            )

        return fixed_step_tsit5(state_dot, state, self.dt, self.substeps)

    def step(
        self,
        states,
        planned_controls,
        path_knots,
        physics_factors: Mapping[str, Any] | None = None,
    ):
        """Advance a state or batch of states."""
        self.validate_physics_factors(physics_factors)
        states = jnp.asarray(states)
        planned_controls = jnp.asarray(planned_controls)
        path_knots = jnp.asarray(path_knots)
        if states.shape[-1] != NX:
            raise ValueError(f"plant state must end in {NX} values")
        if planned_controls.shape[:-2] != states.shape[:-1]:
            raise ValueError(
                "control plan batch shape must match plant state batch shape"
            )
        if planned_controls.shape[-1] != NU:
            raise ValueError(f"control plan must end in {NU} values")
        if path_knots.shape != planned_controls.shape[:-1]:
            raise ValueError("path knots must match the control plan node shape")

        if states.ndim == 1:
            return self.step_one(states, planned_controls, path_knots, physics_factors)

        batch_shape = states.shape[:-1]
        flat_factors = None
        if physics_factors is not None:
            flat_factors = {
                name: jnp.broadcast_to(jnp.asarray(value), batch_shape).reshape(-1)
                for name, value in physics_factors.items()
            }
        next_states = jax.vmap(self.step_one)(
            states.reshape(-1, NX),
            planned_controls.reshape(-1, *planned_controls.shape[-2:]),
            path_knots.reshape(-1, path_knots.shape[-1]),
            flat_factors,
        )
        return next_states.reshape(batch_shape + (NX,))
