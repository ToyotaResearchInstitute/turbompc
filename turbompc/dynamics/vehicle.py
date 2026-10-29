"""Single-track Fiala bicycle model (approximate model)."""

import jax.numpy as jnp

from turbompc.dynamics.base_dynamics import Dynamics


class TimeVehicleDynamics(Dynamics):
    """Time-parameterized Fiala bicycle model."""

    @staticmethod
    def _parameters(last_state_name):
        return {
            "num_states": 7,
            "num_controls": 3,
            "names_states": [
                "yaw_rate",
                "lin_vel",
                "sideslip",
                "rear_wheel_speed",
                "lateral_deviation",
                "relative_heading",
                last_state_name,
            ],
            "names_controls": [
                "steering_angle",
                "motor_plus_rear_brake_torques",
                "front_brake_torque",
            ],
        }

    def __init__(self):
        super().__init__(self._parameters("path_evolution"))

    @staticmethod
    def _front_tire_forces(
        slip_angle, longitudinal_force, cornering_stiffness, friction_limit
    ):
        """Fiala lateral force with friction-circle coupling."""
        longitudinal_force = jnp.clip(
            longitudinal_force, -0.999 * friction_limit, 0.999 * friction_limit
        )
        lateral_limit = jnp.sqrt(friction_limit**2 - longitudinal_force**2)
        tangent = jnp.tan(slip_angle)
        lateral_force = (
            -cornering_stiffness * tangent
            + cornering_stiffness**2
            * tangent
            * jnp.abs(tangent)
            / (3.0 * lateral_limit)
            - cornering_stiffness**3 * tangent**3 / (27.0 * lateral_limit**2)
        )
        lateral_force = jnp.where(
            jnp.abs(slip_angle) < jnp.arctan(3.0 * lateral_limit / cornering_stiffness),
            lateral_force,
            -jnp.sign(slip_angle) * lateral_limit,
        )
        return longitudinal_force, lateral_force

    @staticmethod
    def _rear_tire_forces(slip_angle, slip_ratio, cornering_stiffness, friction_limit):
        """Isotropic coupled-slip Fiala force."""
        tangent = jnp.tan(slip_angle)
        total_slip = jnp.sqrt(tangent**2 + slip_ratio**2 + 1e-12)
        total_force = (
            cornering_stiffness * total_slip
            - cornering_stiffness**2 * total_slip**2 / (3.0 * friction_limit)
            + cornering_stiffness**3 * total_slip**3 / (27.0 * friction_limit**2)
        )
        total_force = jnp.where(
            total_slip < jnp.arctan(3.0 * friction_limit / cornering_stiffness),
            total_force,
            friction_limit,
        )
        return (
            total_force * slip_ratio / total_slip,
            -total_force * tangent / total_slip,
        )

    def state_dot(self, state, control, params):
        """Computes the time derivative of the state of the system."""
        (
            yaw_rate,
            lin_vel,
            sideslip,
            rear_wheel_speed,
            lateral_deviation,
            relative_heading,
        ) = state[:6]
        (
            steering_angle,
            motor_plus_rear_brake_torques,
            front_brake_torque,
        ) = control

        curvature = params["curvature"]
        mass = params["mass"]
        yaw_moment_of_inertia = params["yaw_moment_of_inertia"]
        front_axle_distance = params["front_axle_distance"]
        rear_axle_distance = params["rear_axle_distance"]
        wheel_radius = params["wheel_radius"]
        rear_wheel_moment_of_inertia = params["rear_wheel_moment_of_inertia"]
        front_cornering_stiffness = params["front_cornering_stiffness"]
        rear_cornering_stiffness = params["rear_cornering_stiffness"]
        front_friction_coefficient = params["front_friction_coefficient"]
        rear_friction_coefficient = params["rear_friction_coefficient"]
        gear_ratio = params["gear_ratio"]
        axle_distance = front_axle_distance + rear_axle_distance
        front_normal_force = mass * 9.81 * rear_axle_distance / axle_distance
        rear_normal_force = mass * 9.81 * front_axle_distance / axle_distance

        lin_vel = jnp.maximum(lin_vel, 1.0)
        longitudinal_speed = lin_vel * jnp.cos(sideslip)
        lateral_speed = lin_vel * jnp.sin(sideslip)
        front_slip_angle = (
            jnp.arctan2(
                lateral_speed + front_axle_distance * yaw_rate,
                longitudinal_speed,
            )
            - steering_angle
        )
        rear_slip_angle = jnp.arctan2(
            lateral_speed - rear_axle_distance * yaw_rate,
            longitudinal_speed,
        )
        rear_slip_ratio = (rear_wheel_speed - longitudinal_speed) / longitudinal_speed

        front_longitudinal_force, front_lateral_force = self._front_tire_forces(
            front_slip_angle,
            front_brake_torque / wheel_radius,
            front_cornering_stiffness,
            front_friction_coefficient * front_normal_force,
        )
        rear_longitudinal_force, rear_lateral_force = self._rear_tire_forces(
            rear_slip_angle,
            rear_slip_ratio,
            rear_cornering_stiffness,
            rear_friction_coefficient * rear_normal_force,
        )
        yaw_acceleration = (
            front_axle_distance
            * (
                front_lateral_force * jnp.cos(steering_angle)
                + front_longitudinal_force * jnp.sin(steering_angle)
            )
            - rear_axle_distance * rear_lateral_force
        ) / yaw_moment_of_inertia
        lin_vel_rate = (
            front_longitudinal_force * jnp.cos(steering_angle - sideslip)
            - front_lateral_force * jnp.sin(steering_angle - sideslip)
            + rear_longitudinal_force * jnp.cos(sideslip)
            + rear_lateral_force * jnp.sin(sideslip)
        ) / mass
        sideslip_rate = -yaw_rate + (
            front_longitudinal_force * jnp.sin(steering_angle - sideslip)
            + front_lateral_force * jnp.cos(steering_angle - sideslip)
            - rear_longitudinal_force * jnp.sin(sideslip)
            + rear_lateral_force * jnp.cos(sideslip)
        ) / (mass * lin_vel)
        rear_wheel_speed_rate = (
            wheel_radius
            * (
                jnp.maximum(motor_plus_rear_brake_torques, 0.0) * gear_ratio
                + jnp.minimum(motor_plus_rear_brake_torques, 0.0)
                - wheel_radius * rear_longitudinal_force
            )
            / rear_wheel_moment_of_inertia
        )
        path_rate = (
            lin_vel * jnp.cos(relative_heading) / (1.0 - curvature * lateral_deviation)
        )
        lateral_deviation_rate = lin_vel * jnp.sin(relative_heading)
        relative_heading_rate = sideslip_rate + yaw_rate - curvature * path_rate
        return jnp.array(
            [
                yaw_acceleration,
                lin_vel_rate,
                sideslip_rate,
                rear_wheel_speed_rate,
                lateral_deviation_rate,
                relative_heading_rate,
                path_rate,
            ]
        )


class PathVehicleDynamics(TimeVehicleDynamics):
    """Path-parameterized Fiala bicycle model. Its last state is time evolution."""

    def __init__(self):
        Dynamics.__init__(self, self._parameters("time_evolution"))

    def state_dot(self, state, control, params):
        """Return state derivatives with respect to path evolution."""
        time_derivative = super().state_dot(state, control, params)
        path_rate = jnp.maximum(time_derivative[-1], 1.0)
        return jnp.append(time_derivative[:-1] / path_rate, 1.0 / path_rate)
