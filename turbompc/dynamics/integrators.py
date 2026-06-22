"""Numerical integration functions."""

import enum
from typing import Dict

import jax.numpy as jnp
from jax.lax import scan
from turbompc.dynamics.base_dynamics import Dynamics
from turbompc.utils.jax_utils import value_and_jacfwd


class DiscretizationScheme(enum.IntEnum):
    """Choice of discretization scheme."""

    EULER = 0
    MIDPOINT = 1
    RUNGEKUTTA4 = 2
    IMPLICIT = 10


def predict_next_state(
    dynamics: Dynamics,
    dt: float,
    discretization_scheme: DiscretizationScheme,
    dynamics_state_dot_params: Dict[str, jnp.array],
    state: jnp.array,
    control: jnp.array,
    control_next: jnp.array = None,
) -> jnp.array:
    """
    Predicts the next state from the current state in a single integration step.

    Args:
        dynamics: dynamics class
            Dynamics class
        dt: discretization resolution
            (float)
        discretization_scheme: choice of discretization scheme
            (DiscretizationScheme)
        dynamics_state_dot_params: parameters for the state_dot function of the dynamics
            (key=string, value=jnp.array(parameter_size))
            where each key is an argument of the function state_dot of the dynamics
        state: state variables
            (_num_state_variables, ) array
        control: control input variable
            (_num_control_variables, ) array
        control_next: next control input variable (used for implicit schemes)
            (_num_control_variables, ) array

    Returns:
        next_state: prediction for the next state
            (_num_state_variables, ) array
    """
    if control_next is None:
        control_next = control
    # if discretization_scheme == DiscretizationScheme.EULER:
    state_next = state + dt * dynamics.state_dot(
        state, control, dynamics_state_dot_params
    )
    if discretization_scheme == DiscretizationScheme.MIDPOINT:
        state_mid = state + 0.5 * dt * dynamics.state_dot(
            state, control, dynamics_state_dot_params
        )
        state_next = state + dt * dynamics.state_dot(
            state_mid, control, dynamics_state_dot_params
        )
    elif discretization_scheme == DiscretizationScheme.RUNGEKUTTA4:
        k1 = dynamics.state_dot(state, control, dynamics_state_dot_params)
        k2 = dynamics.state_dot(
            state + 0.5 * dt * k1, control, dynamics_state_dot_params
        )
        k3 = dynamics.state_dot(
            state + 0.5 * dt * k2, control, dynamics_state_dot_params
        )
        k4 = dynamics.state_dot(state + dt * k3, control, dynamics_state_dot_params)
        state_next = state + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) * dt
    elif discretization_scheme == DiscretizationScheme.IMPLICIT:

        def dynamics_constraint(state_next):
            state_dot_now = dynamics.state_dot(
                state, control, dynamics_state_dot_params
            )
            state_dot_next = dynamics.state_dot(
                state_next, control_next, dynamics_state_dot_params
            )
            return state + 0.5 * dt * (state_dot_now + state_dot_next) - state_next

        state_next = state
        for _ in range(10):
            error, error_grad = value_and_jacfwd(dynamics_constraint, state_next)
            state_next = state_next - jnp.linalg.solve(error_grad, error)
    return state_next


def get_state_trajectory(
    dynamics: Dynamics,
    dt: float,
    discretization_scheme: DiscretizationScheme,
    dynamics_state_dot_params: Dict[str, jnp.array],
    initial_state: jnp.array,
    control_matrix: jnp.array,
):
    """
    Predicts the next state from the current state using multiple integration steps

    Args:
        dynamics: dynamics class
            Dynamics class
        dt: discretization resolution
            (float)
        discretization_scheme: choice of discretization scheme
            (DiscretizationScheme)
        dynamics_state_dot_params: parameters for the state_dot function of the dynamics
            (key=string, value=jnp.array(...))
            values may be constant or time-varying with leading dimension
            (horizon) or (horizon+1); time-varying values are sliced per step
        initial state: initial state variables
            (_num_state_variables, ) array
        control_matrix: control input trajectory
            (horizon, _num_control_variables, ) array

    Returns:
        state_matrix: prediction for the state trajectory
            (horizon+1, _num_state_variables, ) array

    Notes:
        If dynamics_state_dot_params includes time-varying values, pass them as
        arrays with leading dimension (horizon) for explicit schemes or
        (horizon+1) for implicit schemes so both endpoints of each interval
        are available. One-dimensional vectors are treated as constants.
    """

    num_steps = control_matrix.shape[0] - 1

    def _expand_dynamics_params(params, steps):
        expanded = {}
        for key, value in params.items():
            if isinstance(value, jnp.ndarray):
                if value.ndim == 0:
                    expanded[key] = jnp.broadcast_to(value, (steps,))
                elif value.ndim == 1:
                    # Treat 1D vectors as constants; time-varying vectors should be 2D.
                    expanded[key] = jnp.broadcast_to(value, (steps,) + value.shape)
                elif value.shape[0] == steps + 1:
                    expanded[key] = value[:steps]
                elif value.shape[0] == steps:
                    expanded[key] = value
                else:
                    expanded[key] = jnp.broadcast_to(value, (steps,) + value.shape)
            else:
                value_arr = jnp.asarray(value)
                if value_arr.ndim == 0:
                    expanded[key] = jnp.broadcast_to(value_arr, (steps,))
                else:
                    expanded[key] = jnp.broadcast_to(
                        value_arr, (steps,) + value_arr.shape
                    )
        return expanded

    params_seq = _expand_dynamics_params(dynamics_state_dot_params, num_steps)

    def next_state_scan(state, controls_step):
        control, control_next, params_step = controls_step
        state_next = predict_next_state(
            dynamics,
            dt,
            discretization_scheme,
            params_step,
            state,
            control,
            control_next,
        )
        return state_next, state_next

    controls = (control_matrix[:-1], control_matrix[1:], params_seq)
    _, state_matrix = scan(next_state_scan, initial_state, controls)
    state_matrix = jnp.concatenate(
        [initial_state[jnp.newaxis, :], state_matrix], axis=0
    )
    return state_matrix
