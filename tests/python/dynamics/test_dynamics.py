"""Checks the dynamics functions."""
import copy

import jax.numpy as jnp
import numpy as np
from turbompc.dynamics.linear_dynamics import (
    LinearDynamics,
    default_parameters,
    default_state_dot_parameters,
)
from turbompc.dynamics.quadrotor_dynamics import (
    QuadrotorDynamics,
    S,
    get_rotation,
    q_conj,
    q_left,
    quadrotor_parameters,
    quadrotor_state_dot_parameters,
)
from turbompc.dynamics.spacecraft_dynamics import (
    SpacecraftDynamics,
    spacecraft_parameters,
    spacecraft_state_dot_parameters,
)

RTOL = 1e-5
ATOL = 1e-8


def _unit_quaternion(rng):
    q_np = rng.standard_normal(4)
    return jnp.array(q_np / np.linalg.norm(q_np))


def test_linear_state_dot_dimensions():
    model = LinearDynamics(default_parameters)
    x = jnp.ones(model.num_states)
    u = jnp.ones(model.num_controls)

    dynamics_state_dot_params = copy.deepcopy(default_state_dot_parameters)
    x_dot = model.state_dot(x, u, dynamics_state_dot_params)
    assert len(x_dot) == model.num_states


def test_spacecraft_state_dot_dimensions():
    model = SpacecraftDynamics(spacecraft_parameters)
    x = jnp.ones(model.num_states)
    u = jnp.ones(model.num_controls)

    dynamics_state_dot_params = copy.deepcopy(spacecraft_state_dot_parameters)
    x_dot = model.state_dot(x, u, dynamics_state_dot_params)
    assert len(x_dot) == model.num_states


def test_skew():
    rng = np.random.default_rng(0)
    a = jnp.array(rng.standard_normal(3))
    b = jnp.array(rng.standard_normal(3))
    assert jnp.allclose(
        jnp.cross(a, b), S(a) @ b, rtol=RTOL, atol=ATOL
    ), "Skew result mismatch"


def test_rotation_orthogonality():
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = _unit_quaternion(rng)
        R = get_rotation(q)
        assert jnp.allclose(
            jnp.eye(3), R.T @ R, rtol=RTOL, atol=ATOL
        ), "Orthogonality result error"


def test_rotation_determinant():
    rng = np.random.default_rng(2)
    for _ in range(10):
        q = _unit_quaternion(rng)
        R = get_rotation(q)
        assert jnp.allclose(
            1.0, np.linalg.det(np.array(R)), rtol=RTOL, atol=ATOL
        ), "Determinant result error"


def test_q_conj():
    rng = np.random.default_rng(3)
    q = _unit_quaternion(rng)
    qi = q_conj(q)
    assert jnp.allclose(q[0], +qi[0], rtol=RTOL, atol=ATOL)
    assert jnp.allclose(q[1], -qi[1], rtol=RTOL, atol=ATOL)
    assert jnp.allclose(q[2], -qi[2], rtol=RTOL, atol=ATOL)
    assert jnp.allclose(q[3], -qi[3], rtol=RTOL, atol=ATOL)


def test_rotation_v_qL():
    rng = np.random.default_rng(4)
    for _ in range(10):
        u = rng.standard_normal(3)
        q = _unit_quaternion(rng)
        R = get_rotation(q)

        a = R @ u
        b = q_left(q_left(q) @ jnp.concatenate((jnp.array([0]), u))) @ q_conj(q)
        assert jnp.allclose(a, b[1:], rtol=RTOL, atol=ATOL), "Rotation map result error"


def test_quadrotor_state_dot_dimensions():
    model = QuadrotorDynamics(quadrotor_parameters)
    x = jnp.ones(model.num_states)
    u = jnp.ones(model.num_controls)

    dynamics_state_dot_params = copy.deepcopy(quadrotor_state_dot_parameters)
    x_dot = model.state_dot(x, u, dynamics_state_dot_params)
    assert len(x_dot) == model.num_states
