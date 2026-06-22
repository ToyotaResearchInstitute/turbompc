from __future__ import annotations

import jax.numpy as jnp


def assert_solution_nontrivial(sol, *, atol: float = 1e-8):
    assert sol.status == 0
    assert jnp.isfinite(sol.states).all()
    assert jnp.isfinite(sol.controls).all()
    assert float(jnp.max(jnp.abs(sol.states))) > atol


def assert_equality_residual(problem, sol, params, *, tol: float = 1e-3):
    eq = problem.equality_constraints(sol.states, sol.controls, params)
    assert float(jnp.linalg.norm(eq, ord=jnp.inf)) < tol


def assert_box_bounds(problem, sol, params, *, tol: float = 1e-6):
    x_min, x_max, u_min, u_max = problem.get_box_bounds(params)
    assert jnp.all(sol.states >= x_min - tol)
    assert jnp.all(sol.states <= x_max + tol)
    assert jnp.all(sol.controls >= u_min - tol)
    assert jnp.all(sol.controls <= u_max + tol)


def assert_solution_changes(sol_a, sol_b, *, atol: float = 1e-8):
    state_delta = jnp.max(jnp.abs(sol_a.states - sol_b.states))
    control_delta = jnp.max(jnp.abs(sol_a.controls - sol_b.controls))
    assert float(jnp.maximum(state_delta, control_delta)) > atol


def assert_slack_active(sol, *, atol: float = 1e-8):
    assert hasattr(sol, "slack")
    assert float(jnp.max(jnp.abs(sol.slack))) > atol
