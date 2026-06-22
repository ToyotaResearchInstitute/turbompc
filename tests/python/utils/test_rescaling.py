import copy

import jax.numpy as jnp
import numpy as np
from turbompc.dynamics.linear_dynamics import (
    LinearDynamics,
    default_state_dot_parameters,
)
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.sqp_diffmpc import SQPDiffMPCSolver
from turbompc.solvers.sqp_osqp import SQPOSQPSolver
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)


def _make_params(N, nx, nu):
    state_min = -0.5 - 0.5 * jnp.arange(nx)
    state_max = 0.8 + 0.2 * jnp.arange(nx)
    control_min = -0.7 - 0.3 * jnp.arange(nu)
    control_max = 0.9 + 0.1 * jnp.arange(nu)
    params = {
        "horizon": N,
        "discretization_resolution": 0.1,
        "discretization_scheme": 0,
        "initial_state": jnp.zeros((nx,)),
        "initial_guess_final_state": jnp.zeros((nx,)),
        "reference_state_trajectory": jnp.zeros((N + 1, nx)),
        "reference_control_trajectory": jnp.zeros((N + 1, nu)),
        "penalize_control_reference": False,
        "rescale_optimization_variables": True,
        "constrain_initial_control": False,
        "initial_control": jnp.zeros((nu,)),
        "state_rescaling_min": state_min,
        "state_rescaling_max": state_max,
        "control_rescaling_min": control_min,
        "control_rescaling_max": control_max,
        "weights_penalization_reference_state_trajectory": jnp.ones((nx,)),
        "weights_penalization_final_state": jnp.zeros((nx,)),
        "weights_penalization_control_squared": jnp.ones((nu,)),
        "weights_penalization_control_rate": jnp.ones((nu,)) * 0.5,
        "state_min_bounds": 1.2 * state_min,
        "state_max_bounds": 1.2 * state_max,
        "control_min_bounds": 1.1 * control_min,
        "control_max_bounds": 1.1 * control_max,
        "dynamics_state_dot_params": {
            "A": jnp.repeat(
                default_state_dot_parameters["A"][None, :, :], repeats=N + 1, axis=0
            ),
            "B": jnp.repeat(
                default_state_dot_parameters["B"][None, :, :], repeats=N + 1, axis=0
            ),
            "b": jnp.repeat(
                default_state_dot_parameters["b"][None, :], repeats=N + 1, axis=0
            ),
        },
    }
    return params


def test_rescaling_roundtrip_and_bounds():
    np.random.seed(0)
    dynamics = LinearDynamics()
    N = 3
    nx = dynamics.num_states
    nu = dynamics.num_controls

    params = _make_params(N, nx, nu)
    problem = OptimalControlProblem(dynamics=dynamics, params=copy.deepcopy(params))

    states = jnp.array(np.random.randn(N + 1, nx))
    controls = jnp.array(np.random.randn(N + 1, nu))

    states_scaled, controls_scaled = problem.scale_states_controls(
        states, controls, params
    )
    states_unscaled, controls_unscaled = problem.unscale_states_controls(
        states_scaled, controls_scaled, params
    )

    np.testing.assert_allclose(states_unscaled, states, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(controls_unscaled, controls, rtol=1e-6, atol=1e-6)

    x_min, x_max, u_min, u_max = problem.get_box_bounds(params)
    np.testing.assert_allclose(
        x_min[0], params["state_min_bounds"], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        x_max[0], params["state_max_bounds"], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        u_min[0], params["control_min_bounds"], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        u_max[0], params["control_max_bounds"], rtol=1e-6, atol=1e-6
    )


def test_control_rate_weights_are_normalized_to_matrices():
    dynamics = LinearDynamics()
    N = 2
    nx = dynamics.num_states
    nu = dynamics.num_controls

    params = _make_params(N, nx, nu)
    problem = OptimalControlProblem(dynamics=dynamics, params=copy.deepcopy(params))

    rd = problem._get_control_rate_weights(params)
    expected = jnp.repeat(
        jnp.diag(params["weights_penalization_control_rate"])[None],
        repeats=N,
        axis=0,
    )

    np.testing.assert_allclose(rd, expected, rtol=1e-6, atol=1e-6)


def test_rescaling_solution_invariance_osqp():
    np.random.seed(0)
    dynamics = LinearDynamics()
    N = 4
    nx = dynamics.num_states
    nu = dynamics.num_controls

    base_params = _make_params(N, nx, nu)
    base_params["rescale_optimization_variables"] = False
    scaled_params = dict(base_params)
    scaled_params["rescale_optimization_variables"] = True

    problem = OptimalControlProblem(
        dynamics=dynamics, params=copy.deepcopy(base_params)
    )
    solver = SQPOSQPSolver(program=problem)

    sol_base = solver.solve(problem_params=base_params)
    sol_scaled = solver.solve(problem_params=scaled_params)

    np.testing.assert_allclose(sol_base.states, sol_scaled.states, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        sol_base.controls, sol_scaled.controls, rtol=1e-5, atol=1e-5
    )


def test_rescaling_solution_invariance_admm():
    np.random.seed(0)
    dynamics = LinearDynamics()
    N = 4
    nx = dynamics.num_states
    nu = dynamics.num_controls

    base_params = _make_params(N, nx, nu)
    base_params["rescale_optimization_variables"] = False
    scaled_params = dict(base_params)
    scaled_params["rescale_optimization_variables"] = True

    problem = OptimalControlProblem(
        dynamics=dynamics, params=copy.deepcopy(base_params)
    )
    solver = TurboMPCSolver(
        program=problem,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
    )

    sol_base = solver.solve(
        solver.initial_guess(base_params), problem_params=base_params
    )
    sol_scaled = solver.solve(
        solver.initial_guess(scaled_params), problem_params=scaled_params
    )

    np.testing.assert_allclose(sol_base.states, sol_scaled.states, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        sol_base.controls, sol_scaled.controls, rtol=1e-4, atol=1e-4
    )


def test_rescaling_solution_invariance_sqp():
    np.random.seed(0)
    dynamics = LinearDynamics()
    N = 4
    nx = dynamics.num_states
    nu = dynamics.num_controls

    base_params = _make_params(N, nx, nu)
    base_params["weights_penalization_control_rate"] = jnp.zeros((nu,))
    base_params["rescale_optimization_variables"] = False
    scaled_params = dict(base_params)
    scaled_params["rescale_optimization_variables"] = True

    problem = OptimalControlProblem(
        dynamics=dynamics, params=copy.deepcopy(base_params)
    )
    solver = SQPDiffMPCSolver(program=problem)

    sol_base = solver.solve(solver.initial_guess(), problem_params=base_params)
    sol_scaled = solver.solve(solver.initial_guess(), problem_params=scaled_params)

    np.testing.assert_allclose(sol_base.states, sol_scaled.states, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        sol_base.controls, sol_scaled.controls, rtol=1e-4, atol=1e-4
    )
