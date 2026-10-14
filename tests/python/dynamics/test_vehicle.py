import numpy as np

from turbompc.dynamics.vehicle import PathVehicleDynamics, TimeVehicleDynamics
from turbompc.utils.load_params import load_problem_params


def vehicle_params():
    return load_problem_params("racing.yaml")["dynamics_state_dot_params"]


def test_vehicle_state_and_control_names():
    time_model = TimeVehicleDynamics()
    path_model = PathVehicleDynamics()

    assert time_model.names_states == [
        "yaw_rate",
        "lin_vel",
        "sideslip",
        "rear_wheel_speed",
        "lateral_deviation",
        "relative_heading",
        "path_evolution",
    ]
    assert path_model.names_states[-1] == "time_evolution"
    assert time_model.names_controls == [
        "steering_angle",
        "motor_plus_rear_brake_torques",
        "front_brake_torque",
    ]


def test_vehicle_state_dot():
    time_model = TimeVehicleDynamics()
    path_model = PathVehicleDynamics()
    state = np.array([0.1, 10.0, 0.01, 10.0, 0.1, 0.02, 0.0])
    control = np.array([0.03, 50.0, -10.0])
    params = vehicle_params()

    time_derivative = time_model.state_dot(state, control, params)
    path_derivative = path_model.state_dot(state, control, params)
    expected = np.append(
        time_derivative[:-1] / time_derivative[-1], 1.0 / time_derivative[-1]
    )

    np.testing.assert_allclose(path_derivative, expected)
