"""Steering PD and switched speed PID control.

This code uses synthetic data and parameters, and does not represent any manufacturer's product.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real

import torch

_STATE_DIM, _CONTROL_DIM, _MEMORY_DIM = 7, 3, 3
_SPEED, _LATERAL, _HEADING = 1, 4, 5


def _nonnegative(name: str, value: Real, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < (0.0 if not positive else 1.0e-300)
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {qualifier} scalar")
    return float(value)


@dataclass(frozen=True)
class PIDParameters:
    """Parameters for one speed PID branch."""

    kp: float
    ki: float
    kd: float
    integral_limit: float
    command_limit_nm: float

    def __post_init__(self) -> None:
        for name in ("kp", "ki", "kd"):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))
        for name in ("integral_limit", "command_limit_nm"):
            object.__setattr__(
                self,
                name,
                _nonnegative(name, getattr(self, name), positive=True),
            )


@dataclass(frozen=True)
class ReferenceControllerOutput:
    control: torch.Tensor
    memory: torch.Tensor
    speed_error: torch.Tensor
    longitudinal_mode: torch.Tensor
    throttle_torque: torch.Tensor
    brake_torque: torch.Tensor


def _control_kernel(
    plant_state: torch.Tensor,
    reference_state: torch.Tensor,
    reference_control: torch.Tensor,
    memory: torch.Tensor,
    lateral_kp: float,
    lateral_kd: float,
    dt: float,
    throttle_deadband: float,
    brake_deadband: float,
    throttle_kp: float,
    throttle_ki: float,
    throttle_kd: float,
    throttle_integral_limit: float,
    throttle_torque_limit: float,
    brake_rear_fraction: float,
    brake_kp: float,
    brake_ki: float,
    brake_kd: float,
    brake_integral_limit: float,
    brake_torque_limit: float,
    longitudinal_enabled: bool,
) -> tuple[torch.Tensor, ...]:
    """Compute the control and updated PID memory."""
    lateral_error = reference_state[..., _LATERAL] - plant_state[..., _LATERAL]
    lateral_rate_error = reference_state[..., _SPEED] * torch.sin(
        reference_state[..., _HEADING]
    ) - plant_state[..., _SPEED] * torch.sin(plant_state[..., _HEADING])
    steering = (
        reference_control[..., 0]
        + lateral_kp * lateral_error
        + lateral_kd * lateral_rate_error
    )

    speed_error = reference_state[..., _SPEED] - plant_state[..., _SPEED]
    derivative = (speed_error - memory[..., 2]) / dt
    if longitudinal_enabled:
        throttle_active = speed_error > throttle_deadband
        brake_active = speed_error < -brake_deadband
    else:
        throttle_active = torch.zeros_like(speed_error, dtype=torch.bool)
        brake_active = torch.zeros_like(speed_error, dtype=torch.bool)
    throttle_integral = torch.where(
        throttle_active,
        torch.clamp(
            memory[..., 0] + dt * speed_error,
            0.0,
            throttle_integral_limit,
        ),
        memory[..., 0],
    )
    brake_integral = torch.where(
        brake_active,
        torch.clamp(
            memory[..., 1] + dt * speed_error,
            -brake_integral_limit,
            0.0,
        ),
        memory[..., 1],
    )
    if not longitudinal_enabled:
        throttle_integral = torch.zeros_like(throttle_integral)
        brake_integral = torch.zeros_like(brake_integral)
    throttle_correction = torch.where(
        throttle_active,
        torch.clamp(
            throttle_kp * speed_error
            + throttle_ki * throttle_integral
            + throttle_kd * derivative,
            0.0,
            throttle_torque_limit,
        ),
        torch.zeros_like(speed_error),
    )
    brake_correction = torch.where(
        brake_active,
        torch.clamp(
            brake_kp * speed_error + brake_ki * brake_integral + brake_kd * derivative,
            -brake_torque_limit,
            0.0,
        ),
        torch.zeros_like(speed_error),
    )
    mode = throttle_active.to(torch.int8) - brake_active.to(torch.int8)
    memory_speed_error = (
        speed_error if longitudinal_enabled else torch.zeros_like(speed_error)
    )
    next_memory = torch.stack(
        (throttle_integral, brake_integral, memory_speed_error), dim=-1
    )
    control = torch.stack(
        (
            steering,
            reference_control[..., 1]
            + throttle_correction
            + brake_rear_fraction * brake_correction,
            reference_control[..., 2] + (1.0 - brake_rear_fraction) * brake_correction,
        ),
        dim=-1,
    )
    return (
        control,
        next_memory,
        speed_error,
        mode,
        throttle_correction,
        brake_correction,
    )


_COMPILED_CONTROL_KERNEL = torch.compile(_control_kernel, fullgraph=True, dynamic=True)


@dataclass(frozen=True)
class ReferencePDController:
    """Track a racing reference with steering PD and speed PID."""

    lateral_kp: float
    lateral_kd: float
    dt: float
    throttle_deadband: float
    brake_deadband: float
    throttle: PIDParameters
    brake: PIDParameters
    brake_rear_fraction: float = 0.5
    longitudinal_enabled: bool = True
    compile_kernel: bool = True
    _kernel: Callable[..., tuple[torch.Tensor, ...]] = field(
        init=False, repr=False, compare=False
    )
    _parameters: tuple[float | bool, ...] = field(init=False, repr=False, compare=False)

    memory_dim = _MEMORY_DIM

    def __post_init__(self) -> None:
        for name in (
            "lateral_kp",
            "lateral_kd",
            "throttle_deadband",
            "brake_deadband",
        ):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))
        object.__setattr__(self, "dt", _nonnegative("dt", self.dt, positive=True))
        if not isinstance(self.throttle, PIDParameters):
            raise TypeError("throttle must be PIDParameters")
        if not isinstance(self.brake, PIDParameters):
            raise TypeError("brake must be PIDParameters")
        brake_rear_fraction = _nonnegative(
            "brake_rear_fraction", self.brake_rear_fraction
        )
        if brake_rear_fraction > 1.0:
            raise ValueError("brake_rear_fraction must not exceed 1")
        object.__setattr__(self, "brake_rear_fraction", brake_rear_fraction)
        if not isinstance(self.longitudinal_enabled, bool):
            raise TypeError("longitudinal_enabled must be bool")
        if not isinstance(self.compile_kernel, bool):
            raise TypeError("compile_kernel must be bool")
        parameters = (
            self.lateral_kp,
            self.lateral_kd,
            self.dt,
            self.throttle_deadband,
            self.brake_deadband,
            self.throttle.kp,
            self.throttle.ki,
            self.throttle.kd,
            self.throttle.integral_limit,
            self.throttle.command_limit_nm,
            self.brake_rear_fraction,
            self.brake.kp,
            self.brake.ki,
            self.brake.kd,
            self.brake.integral_limit,
            self.brake.command_limit_nm,
            self.longitudinal_enabled,
        )
        object.__setattr__(self, "_parameters", parameters)
        object.__setattr__(
            self,
            "_kernel",
            _COMPILED_CONTROL_KERNEL if self.compile_kernel else _control_kernel,
        )

    def initial_memory(self, plant_state: torch.Tensor) -> torch.Tensor:
        """Initialize throttle/brake integrals and the prior speed error."""
        return plant_state.new_zeros((*plant_state.shape[:-1], self.memory_dim))

    def memory_scale(self, speed_scale: float) -> tuple[float, float, float]:
        return (
            self.throttle.integral_limit,
            self.brake.integral_limit,
            _nonnegative("speed_scale", speed_scale, positive=True),
        )

    def __call__(
        self,
        plant_state: torch.Tensor,
        reference_state: torch.Tensor,
        reference_control: torch.Tensor,
        memory: torch.Tensor,
    ) -> ReferenceControllerOutput:
        """Return the unclipped control and updated PID memory."""
        for name, value, width in (
            ("plant_state", plant_state, _STATE_DIM),
            ("reference_state", reference_state, _STATE_DIM),
            ("reference_control", reference_control, _CONTROL_DIM),
            ("memory", memory, self.memory_dim),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim < 1 or value.shape[-1] != width:
                raise ValueError(f"{name} must have shape (..., {width})")
        batch_shape = plant_state.shape[:-1]
        if any(
            value.shape[:-1] != batch_shape
            for value in (reference_state, reference_control, memory)
        ):
            raise ValueError("plant, reference, control, and memory batches must match")

        tensors = (plant_state, reference_state, reference_control, memory)
        single_row = self.compile_kernel and plant_state.numel() == _STATE_DIM
        if single_row:
            tensors = tuple(torch.cat((value, value), dim=0) for value in tensors)
        outputs = self._kernel(*tensors, *self._parameters)
        if single_row:
            outputs = tuple(value[:1] for value in outputs)
        return ReferenceControllerOutput(*outputs)


__all__ = [
    "PIDParameters",
    "ReferenceControllerOutput",
    "ReferencePDController",
]
