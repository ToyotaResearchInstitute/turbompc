import jax.numpy as jnp
from tests.helpers.problem_fixtures import tile_spacecraft_inertia
from turbompc.dynamics.linear_dynamics import LinearDynamics
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.utils.load_params import load_problem_params


def test_default_box_bounds_unconstrained():
    params = load_problem_params("spacecraft.yaml")
    tile_spacecraft_inertia(params)
    dynamics = LinearDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=params)

    x_lo, x_hi, u_lo, u_hi = problem.get_box_bounds()

    assert x_lo.shape == x_hi.shape
    assert u_lo.shape == u_hi.shape

    assert jnp.all(jnp.isneginf(x_lo))
    assert jnp.all(jnp.isposinf(x_hi))
    assert jnp.all(jnp.isneginf(u_lo))
    assert jnp.all(jnp.isposinf(u_hi))


def test_user_provided_box_bounds_shapes():
    params = load_problem_params("spacecraft.yaml")
    tile_spacecraft_inertia(params)
    dynamics = LinearDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=params)

    N = problem.horizon
    nx = problem.num_state_variables
    nu = problem.num_control_variables

    params = dict(params)
    params["state_min_bounds"] = -jnp.ones((N + 1, nx))
    params["state_max_bounds"] = jnp.ones((N + 1, nx))
    params["control_min_bounds"] = -2.0 * jnp.ones((N + 1, nu))
    params["control_max_bounds"] = 2.0 * jnp.ones((N + 1, nu))

    x_lo, x_hi, u_lo, u_hi = problem.get_box_bounds(params)

    assert x_lo.shape == (N + 1, nx)
    assert u_lo.shape == (N + 1, nu)
    assert jnp.all(x_lo <= x_hi)
    assert jnp.all(u_lo <= u_hi)


def test_constant_bounds_are_repeated_over_horizon():
    params = load_problem_params("spacecraft_constrained.yaml")
    tile_spacecraft_inertia(params)
    dynamics = SpacecraftDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=params)

    x_min, x_max, u_min, u_max = problem.get_box_bounds(params)

    N = problem.horizon
    nx = problem.num_state_variables
    nu = problem.num_control_variables

    assert x_min.shape == (N + 1, nx)
    assert x_max.shape == (N + 1, nx)
    assert u_min.shape == (N + 1, nu)
    assert u_max.shape == (N + 1, nu)

    # Verify repetition: all timesteps equal the first
    assert jnp.all(x_min == x_min[0])
    assert jnp.all(x_max == x_max[0])
    assert jnp.all(u_min == u_min[0])
    assert jnp.all(u_max == u_max[0])
