import jax.numpy as jnp
import numpy as np
from tests.helpers.problem_fixtures import make_linear_params, tile_spacecraft_inertia
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.utils.load_params import load_problem_params


def make_linear_ocp(
    horizon: int = 6,
    *,
    implicit: bool = False,
    bounded: bool = False,
    rescale: bool = False,
    rescaling: str = "none",
):
    dynamics, params = make_linear_params(
        horizon,
        implicit=implicit,
        bounded=bounded,
        rescale=rescale,
        rescaling=rescaling,
        initial_state=jnp.array([0.2, -0.1, 0.05, 5.0]),
    )
    return OptimalControlProblem(dynamics=dynamics, params=params), params


def make_spacecraft_ocp(horizon: int = 8, *, bounded: bool = True, seed: int = 42):
    np.random.seed(seed)
    params = load_problem_params("spacecraft.yaml")
    turbompc_horizon = horizon - 1
    params["initial_state"] = -0.1 + np.random.rand(3) * 0.2
    params["horizon"] = turbompc_horizon
    tile_spacecraft_inertia(params, horizon=turbompc_horizon)
    params["reference_state_trajectory"] = jnp.zeros((turbompc_horizon + 1, 3))
    params["reference_control_trajectory"] = jnp.zeros((turbompc_horizon + 1, 3))
    params["weights_penalization_final_state"] = jnp.zeros(3)
    if not bounded:
        big = 1.0e3
        params["state_min_bounds"] = -jnp.ones(3) * big
        params["state_max_bounds"] = jnp.ones(3) * big
        params["control_min_bounds"] = -jnp.ones(3) * big
        params["control_max_bounds"] = jnp.ones(3) * big
    ocp = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    return ocp, params


def make_spacecraft_ocp_implicit(horizon: int = 6):
    dynamics = SpacecraftDynamics()
    nx = dynamics.num_states
    nu = dynamics.num_controls
    bound = 0.5
    params = {
        "horizon": horizon,
        "discretization_resolution": 0.1,
        "discretization_scheme": 10,
        "initial_state": jnp.array([0.1, -0.15, 0.08]),
        "initial_guess_final_state": jnp.zeros((nx,)),
        "reference_state_trajectory": jnp.zeros((horizon + 1, nx)),
        "reference_control_trajectory": jnp.zeros((horizon + 1, nu)),
        "penalize_control_reference": False,
        "rescale_optimization_variables": False,
        "constrain_initial_control": False,
        "initial_control": jnp.zeros((nu,)),
        "state_rescaling_min": -jnp.ones((nx,)),
        "state_rescaling_max": jnp.ones((nx,)),
        "control_rescaling_min": -jnp.ones((nu,)),
        "control_rescaling_max": jnp.ones((nu,)),
        "weights_penalization_reference_state_trajectory": jnp.ones((nx,)),
        "weights_penalization_final_state": jnp.zeros((nx,)),
        "weights_penalization_control_squared": jnp.ones((nu,)),
        "weights_penalization_control_rate": jnp.zeros((nu,)),
        "state_min_bounds": -jnp.ones((nx,)) * bound,
        "state_max_bounds": jnp.ones((nx,)) * bound,
        "control_min_bounds": -jnp.ones((nu,)) * bound,
        "control_max_bounds": jnp.ones((nu,)) * bound,
        "dynamics_state_dot_params": {"inertia_vector": jnp.array([5.0, 2.0, 1.0])},
    }
    tile_spacecraft_inertia(params, horizon=horizon)
    ocp = OptimalControlProblem(dynamics=dynamics, params=params)
    return ocp, params
