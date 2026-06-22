"""Backtracking linesearch utilities."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from turbompc.problems.optimal_control_problem import (
    OptimalControlProblem,
    SlackProblemAdapter,
)
from turbompc.utils.jax_utils import value_and_jacrev


def evaluate_constraints_with_bounds(
    program: OptimalControlProblem,
    states: jnp.ndarray,
    controls: jnp.ndarray,
    slacks: jnp.ndarray,
    problem_params: Dict[str, any],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return equality norm, inequality norm, and inequality data.

    If use_slack_variables=True, constraints are evaluated on g(x,u)+slack.
    """
    eq_constraints = program.equality_constraints(states, controls, problem_params)
    eq_constraints_l1_norm = jnp.sum(jnp.abs(eq_constraints))
    use_slack_variables = program.use_slack_variables
    if use_slack_variables and not isinstance(program, SlackProblemAdapter):
        raise ValueError("Slack-enabled problems require SlackProblemAdapter.")
    if use_slack_variables:
        ineq_values, ineq_lower, ineq_upper = program.inequality_constraints_with_slack(
            states, controls, slacks, problem_params
        )
    else:
        ineq_values, ineq_lower, ineq_upper = program.inequality_constraints(
            states, controls, problem_params
        )
    ineq_values = ineq_values.reshape((states.shape[0], -1))
    ineq_lower = ineq_lower.reshape((states.shape[0], -1))
    ineq_upper = ineq_upper.reshape((states.shape[0], -1))
    ineq_violation = jnp.maximum(
        0.0, ineq_lower.reshape(-1) - ineq_values.reshape(-1)
    ) + jnp.maximum(0.0, ineq_values.reshape(-1) - ineq_upper.reshape(-1))
    ineq_constraints_l1_norm = jnp.sum(ineq_violation)
    return (
        eq_constraints_l1_norm,
        ineq_constraints_l1_norm,
        ineq_values,
        ineq_lower,
        ineq_upper,
    )


def backtracking_linesearch(
    program: OptimalControlProblem,
    problem_params: Dict[str, any],
    solver_params: Dict[str, any],
    states: jnp.ndarray,
    controls: jnp.ndarray,
    slacks: jnp.ndarray,
    states_new: jnp.ndarray,
    controls_new: jnp.ndarray,
    slacks_new: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Backtracking linesearch for both SQP solvers."""
    horizon = program.horizon
    nx = program.num_state_variables
    nu = program.num_control_variables
    eta = solver_params["linesearch_eta"]
    alpha_candidates = jnp.array(solver_params["linesearch_alphas"])
    use_slack_variables = program.use_slack_variables
    if use_slack_variables and not isinstance(program, SlackProblemAdapter):
        raise ValueError("Slack-enabled problems require SlackProblemAdapter.")
    states_delta = states_new - states
    controls_delta = controls_new - controls
    slacks_delta = slacks_new - slacks
    z = jnp.concatenate([states, controls, slacks], axis=-1).flatten()
    z_delta = jnp.concatenate(
        [states_delta, controls_delta, slacks_delta], axis=-1
    ).flatten()

    def cost_function(z_flat):
        z_mat = jnp.reshape(z_flat, (horizon + 1, -1))
        if use_slack_variables:
            cost_value = program.cost_with_slack(
                z_mat[:, :nx],
                z_mat[:, nx : (nx + nu)],
                z_mat[:, (nx + nu) :],
                problem_params,
            )
        else:
            cost_value = program.cost(
                z_mat[:, :nx], z_mat[:, nx : (nx + nu)], problem_params
            )
        return cost_value * jnp.ones(1)

    cost, cost_dz = value_and_jacrev(cost_function, z)
    cost, cost_dz = cost[0], cost_dz[0]
    cost_dz_times_delta_z = jnp.dot(cost_dz, z_delta)

    (
        eq_constraints_l1_norm,
        ineq_constraints_l1_norm,
        *_,
    ) = evaluate_constraints_with_bounds(
        program, states, controls, slacks, problem_params
    )
    constraints_l1_norm = eq_constraints_l1_norm + ineq_constraints_l1_norm
    mu_min = float(solver_params.get("linesearch_mu_min", 0.1))
    mu_max = float(solver_params.get("linesearch_mu_max", 1.0e6))
    constraint_floor = float(solver_params.get("linesearch_constraint_floor", 1.0e-8))
    mu_raw = cost_dz_times_delta_z / (
        0.5 * jnp.maximum(constraints_l1_norm, constraint_floor)
    )
    linesearch_mu = jnp.clip(
        jnp.nan_to_num(mu_raw, nan=mu_min, posinf=mu_max, neginf=mu_min),
        mu_min,
        mu_max,
    )
    merit_value = cost + linesearch_mu * constraints_l1_norm
    merit_derivative = cost_dz_times_delta_z - linesearch_mu * constraints_l1_norm

    def candidate_merit_function(alpha):
        candidate_states = states + alpha * states_delta
        candidate_controls = controls + alpha * controls_delta
        candidate_slacks = slacks + alpha * slacks_delta
        if use_slack_variables:
            candidate_cost = program.cost_with_slack(
                candidate_states, candidate_controls, candidate_slacks, problem_params
            )
        else:
            candidate_cost = program.cost(
                candidate_states, candidate_controls, problem_params
            )
        (
            eq_constraints_l1_norm,
            ineq_constraints_l1_norm,
            *_,
        ) = evaluate_constraints_with_bounds(
            program,
            candidate_states,
            candidate_controls,
            candidate_slacks,
            problem_params,
        )
        merit_value_candidate = candidate_cost + linesearch_mu * (
            eq_constraints_l1_norm + ineq_constraints_l1_norm
        )
        return merit_value_candidate

    candidate_merits = jax.vmap(candidate_merit_function)(alpha_candidates)
    decrease_values = candidate_merits - (
        merit_value + eta * alpha_candidates * merit_derivative
    )

    # Find largest α such that the decrease condition holds.
    # If Armijo rejects every finite candidate, take the finite candidate with
    # the best actual merit value. Falling back to the smallest alpha can stall
    # SQP indefinitely when the local merit derivative is too optimistic.
    finite_mask = (
        jnp.isfinite(alpha_candidates)
        & jnp.isfinite(candidate_merits)
        & jnp.isfinite(decrease_values)
    )
    valid_mask = finite_mask & (decrease_values <= 0)
    valid_index = jnp.max(jnp.where(valid_mask, jnp.arange(len(alpha_candidates)), -1))
    best_finite_index = jnp.argmin(jnp.where(finite_mask, candidate_merits, jnp.inf))
    fallback_index = jnp.where(jnp.any(finite_mask), best_finite_index, 0)
    # Prefer the largest Armijo-accepted step.  The lowest-merit candidate is
    # only a fallback when Armijo rejects every finite candidate; otherwise it
    # can select an unnecessarily tiny alpha and stall SQP progress.
    index = jnp.where(valid_index >= 0, valid_index, fallback_index)
    alpha = alpha_candidates[index]

    updated_states = states + alpha * states_delta
    updated_controls = controls + alpha * controls_delta
    updated_slacks = slacks + alpha * slacks_delta
    return (updated_states, updated_controls, updated_slacks), alpha
