"""Obstacle-avoidance optimal control problem variants."""

from typing import Any, Dict, Tuple

import jax.numpy as jnp
from jax import vmap
from turbompc.problems.optimal_control_problem import (
    OptimalControlProblem,
    make_slack_problem,
)


class OptimalControlProblemObstacle(OptimalControlProblem):
    """Quadratic tracking OCP with circular obstacle avoidance constraints."""

    def step_inequality_constraints(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Returns one-step box and obstacle constraints."""
        g, g_l, g_u = super().step_inequality_constraints(state, control, params)
        position = state[: self.params["obstacles_dimension"]]
        centers = jnp.asarray(params["obstacles_centers"])
        radii = jnp.asarray(params["obstacles_radii"])

        def obstacle_constraint(position, obs_center, obs_radius):
            # inequality constraint is:
            # (position - obs_center)**2 >= obs_radius**2
            # <=> 1 - (position - obs_center)**2 / obs_radius**2 <= 0
            constraint = 1.0 - jnp.linalg.norm(position - obs_center) / (
                obs_radius + 0.001
            )
            return constraint

        g_obs = vmap(obstacle_constraint, in_axes=(None, 0, 0))(
            position, centers, radii
        )
        g_obs_l = -1e9 * jnp.ones_like(g_obs)
        g_obs_u = jnp.zeros_like(g_obs)
        if self.rescale_optimization_variables:
            _, _, _, _, state_diff, _ = self._get_rescaling_params(params)
            row_scale = 1.0 / jnp.mean(state_diff[: self.params["obstacles_dimension"]])
            g_obs = g_obs * row_scale
            g_obs_l = g_obs_l * row_scale
            g_obs_u = g_obs_u * row_scale

        g = jnp.concatenate([g, g_obs])
        g_l = jnp.concatenate([g_l, g_obs_l])
        g_u = jnp.concatenate([g_u, g_obs_u])
        return (g, g_l, g_u)


OptimalControlProblemObstacleSlack = make_slack_problem(OptimalControlProblemObstacle)


__all__ = [
    "OptimalControlProblemObstacle",
    "OptimalControlProblemObstacleSlack",
]
