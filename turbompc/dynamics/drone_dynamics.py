"""Drone dynamics class."""

from typing import Any, Dict

import jax.numpy as jnp
from turbompc.dynamics.base_dynamics import Dynamics

drone_parameters: Dict[str, Any] = {
    "verbose": False,
    "num_states": 6,
    "num_controls": 3,
    "names_states": ["p_x", "p_y", "p_z", "v_x", "v_y", "v_z"],
    "names_controls": ["F_x", "F_y", "F_z"],
}
drone_state_dot_parameters = {
    "mass": 32.0 * jnp.ones(1),
    "drag_coefficient": 0.2 * jnp.ones(1),
}


class DroneDynamics(Dynamics):
    """Drone dynamics class."""

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initializes the class.

        Args:
            parameters:  parameters of the class.
                (str, Any) dictionary
        """
        super().__init__(parameters)

    def parameters_dictionary_is_valid_or_raise_error(
        self,
        params_to_check: Dict[str, Any],
        params_valid: Dict[str, Any] = drone_parameters,
    ) -> bool:
        """
        Returns True if params_to_check is a valid dictionary of parameters (in
        particular, it contains all keys in params_default_valid).
        Raises an error otherwise.

        Args:
            params_to_check:  parameters to check.
                (str, Any) dictionary
            params_valid:  (default) valid parameters for this class.
                (str, Any) dictionary
        """
        success = super().parameters_dictionary_is_valid_or_raise_error(
            params_to_check, params_valid
        )
        return success

    def state_dot(
        self,
        state: jnp.array,
        control: jnp.array,
        params: Dict[str, Any] = drone_state_dot_parameters,
    ) -> jnp.array:
        """
        Computes the time derivative of the state of the system.

        Returns x_dot = f(x, u) where f describes the dynamics of the system.

        Args:
            state: state of the system (see names_states)
                (_num_states, ) array
            control: control input applied to the system (see names_controls)
                (_num_controls, ) array
            params: parameters of the state_dot function of the dynamics.
                (str, Any) dictionary

        Returns:
            state_dot: time derivative of the state
                (_num_states, ) array
        """
        speed = state[..., 3:]

        drag_coeff_param = params["drag_coefficient"]
        mass_param = params["mass"]

        # If shape is (N, 1) due to vmap, extract just the first element's value
        if drag_coeff_param.ndim > 1:
            drag_coefficient = drag_coeff_param[0]
        else:
            drag_coefficient = drag_coeff_param

        if mass_param.ndim > 1:
            mass = mass_param[0]
        else:
            mass = mass_param

        drag_coefficient = drag_coefficient.reshape(())
        mass = mass.reshape(())

        drag_force = drag_coefficient * jnp.abs(speed) * speed
        acceleration = (control - drag_force) / mass

        state_dot = jnp.concatenate([speed, acceleration], axis=-1)
        return state_dot
