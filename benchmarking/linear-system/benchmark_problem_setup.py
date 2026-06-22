"""Shared constrained benchmark setup helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from turbompc.dynamics.linear_dynamics import (
    LinearDynamics,
    default_state_dot_parameters,
)
from utils import N_CTRL, N_STATE

DIMENSIONS = N_STATE + N_CTRL


def control_bounds_np(umax: float) -> tuple[np.ndarray, np.ndarray]:
    lower = -umax * np.ones((N_CTRL,), dtype=np.float64)
    upper = umax * np.ones((N_CTRL,), dtype=np.float64)
    return lower, upper


def build_turbompc_linear_problem(
    horizon: int,
    umax: float,
    n_state: int | None = None,
    n_ctrl: int | None = None,
) -> tuple[LinearDynamics, dict]:
    nx = n_state if n_state is not None else N_STATE
    nu = n_ctrl if n_ctrl is not None else N_CTRL

    dynamics = LinearDynamics(
        {
            "verbose": False,
            "num_states": nx,
            "num_controls": nu,
            "names_states": [f"x{i}" for i in range(nx)],
            "names_controls": [f"u{i}" for i in range(nu)],
        }
    )

    params = {
        "horizon": horizon,
        "discretization_resolution": 1.0,
        "discretization_scheme": 0,
        "initial_state": jnp.zeros((nx,)),
        "initial_guess_final_state": jnp.zeros((nx,)),
        "reference_state_trajectory": jnp.zeros((horizon + 1, nx)),
        "reference_control_trajectory": jnp.zeros((horizon + 1, nu)),
        "penalize_control_reference": False,
        "rescale_optimization_variables": False,
        "constrain_initial_control": False,
        "initial_control": jnp.zeros((nu,)),
        "state_rescaling_min": -np.linspace(0.1, 5.0, nx),
        "state_rescaling_max": jnp.ones((nx,)),
        "control_rescaling_min": -np.linspace(0.2, 3.0, nu),
        "control_rescaling_max": jnp.ones((nu,)),
        "weights_penalization_reference_state_trajectory": jnp.ones((nx,)),
        "weights_penalization_final_state": jnp.zeros((nx,)),
        "weights_penalization_control_squared": jnp.ones((nu,)),
        "weights_penalization_control_rate": jnp.zeros((nu,)),
        "state_min_bounds": -jnp.ones((nx,)) * 1.0e7,
        "state_max_bounds": jnp.ones((nx,)) * 1.0e7,
        "control_min_bounds": -jnp.ones((nu,)) * umax,
        "control_max_bounds": jnp.ones((nu,)) * umax,
        "dynamics_state_dot_params": {
            "A": default_state_dot_parameters["A"],
            "B": default_state_dot_parameters["B"],
            "b": default_state_dot_parameters["b"],
        },
    }
    return dynamics, params
