"""Tests for implicit integrator linearization."""

import jax.numpy as jnp
import numpy as np
from turbompc.dynamics.integrators import get_state_trajectory
from turbompc.dynamics.linear_dynamics import LinearDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.utils.load_params import load_problem_params


def test_implicit_linearization_matches_constraint():
    params = load_problem_params("linear.yaml")
    params["discretization_scheme"] = 10  # implicit trapezoidal
    dynamics = LinearDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=params)

    rng = np.random.default_rng(0)
    controls = jnp.array(
        rng.standard_normal((problem.horizon + 1, dynamics.num_controls))
    )
    states = get_state_trajectory(
        dynamics,
        float(params["discretization_resolution"]),
        problem.discretization_scheme,
        params["dynamics_state_dot_params"],
        params["initial_state"],
        controls,
    )

    As_next, Bs_next, As, Bs, Cs = problem.get_dynamics_linearized_matrices(
        states, controls, params
    )

    residuals = []
    for t in range(problem.horizon):
        residuals.append(
            As_next[t] @ states[t + 1]
            + As[t] @ states[t]
            + Bs_next[t] @ controls[t + 1]
            + Bs[t] @ controls[t]
            - Cs[t + 1]
        )
    residuals = jnp.stack(residuals, axis=0)

    np.testing.assert_allclose(
        residuals, jnp.zeros_like(residuals), rtol=1e-6, atol=1e-6
    )
    assert jnp.any(jnp.abs(Bs_next) > 1.0e-8)


def test_time_varying_dynamics_params_respected():
    params = load_problem_params("linear.yaml")
    params = dict(params)
    params["horizon"] = 4
    params["discretization_scheme"] = 0  # euler
    dynamics = LinearDynamics()
    problem = OptimalControlProblem(dynamics=dynamics, params=params)

    rng = np.random.default_rng(0)
    controls = jnp.array(
        rng.standard_normal((problem.horizon + 1, dynamics.num_controls))
    )
    initial_state = jnp.array(rng.standard_normal((dynamics.num_states,)))

    A_seq = jnp.array(
        rng.standard_normal((problem.horizon, dynamics.num_states, dynamics.num_states))
    )
    B_seq = jnp.array(
        rng.standard_normal(
            (problem.horizon, dynamics.num_states, dynamics.num_controls)
        )
    )
    b_seq = jnp.array(rng.standard_normal((problem.horizon, dynamics.num_states)))
    dynamics_params = {"A": A_seq, "B": B_seq, "b": b_seq}

    states = get_state_trajectory(
        dynamics,
        float(params["discretization_resolution"]),
        problem.discretization_scheme,
        dynamics_params,
        initial_state,
        controls,
    )

    params["initial_state"] = initial_state
    params["dynamics_state_dot_params"] = dynamics_params
    constraints = problem.equality_constraints(states, controls, params)
    np.testing.assert_allclose(constraints, jnp.zeros_like(constraints), atol=1e-6)
