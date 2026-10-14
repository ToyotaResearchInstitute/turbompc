"""Linesearch methods."""

import enum
from typing import Any, Callable, Dict, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp

from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.admm.inequality_log_barrier import (
    barrier_value,
    fraction_to_boundary_step,
)
from turbompc.utils.jax_utils import value_and_jacrev


class LinesearchMethod(enum.IntEnum):
    """Line search method"""

    NONE = 0
    MERIT = 1
    FILTER = 2


_NAME_TO_LINESEARCH_METHOD: dict[str, LinesearchMethod] = {
    "none": LinesearchMethod.NONE,
    "merit": LinesearchMethod.MERIT,
    "filter": LinesearchMethod.FILTER,
}
LINESEARCH_METHOD_CHOICES: tuple[str, ...] = tuple(sorted(_NAME_TO_LINESEARCH_METHOD))


def parse_linesearch_method(
    value: Union[str, LinesearchMethod],
) -> LinesearchMethod:
    """Parse a line-search method from its public name or enum value."""
    if isinstance(value, LinesearchMethod):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Expected str or LinesearchMethod, got {type(value)}")
    try:
        return _NAME_TO_LINESEARCH_METHOD[value.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown linesearch method: {value!r}. Options:"
            f" {LINESEARCH_METHOD_CHOICES}"
        ) from exc


def evaluate_constraints_with_bounds(
    program: OptimalControlProblem,
    states: jnp.ndarray,
    controls: jnp.ndarray,
    slacks: jnp.ndarray,
    problem_params: Dict[str, Any],
    *,
    norm: Union[int, float],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return equality norm, inequality norm, and inequality data.

    If the program uses slack variables, constraints are evaluated on ``g(x, u) + slack``.
    ``norm`` must be either ``1`` or ``jnp.inf``.
    """
    if norm not in (1, float("inf")):
        raise ValueError(f"Constraint norm must be 1 or inf, got {norm!r}.")

    def constraint_norm(values):
        if values.size == 0:
            return jnp.array(0.0, dtype=states.dtype)
        return jnp.linalg.norm(values.reshape(-1), ord=norm)

    eq_constraints = program.equality_constraints(states, controls, problem_params)
    eq_constraints_norm = constraint_norm(eq_constraints)
    if program.use_slack_variables:
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
    ineq_constraints_norm = constraint_norm(ineq_violation)
    return (
        eq_constraints_norm,
        ineq_constraints_norm,
        ineq_values,
        ineq_lower,
        ineq_upper,
    )


def backtracking_merit_step(
    objective_value: jnp.ndarray,
    objective_derivative: jnp.ndarray,
    constraint_l1_norm: jnp.ndarray,
    candidate_merit: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    linesearch_params: Dict[str, Any],
) -> jnp.ndarray:
    """Return the largest Armijo-accepted candidate step."""
    eta = linesearch_params["armijo_coefficient"]
    alpha_candidates = jnp.asarray(linesearch_params["alphas"])
    mu_min = float(linesearch_params.get("penalty_min", 0.1))
    mu_max = float(linesearch_params.get("penalty_max", 1.0e6))
    constraint_floor = float(linesearch_params.get("constraint_norm_floor", 1.0e-8))
    mu_raw = objective_derivative / (
        0.5 * jnp.maximum(constraint_l1_norm, constraint_floor)
    )
    linesearch_mu = jnp.clip(
        jnp.nan_to_num(mu_raw, nan=mu_min, posinf=mu_max, neginf=mu_min),
        mu_min,
        mu_max,
    )
    merit_value = objective_value + linesearch_mu * constraint_l1_norm
    merit_derivative = objective_derivative - linesearch_mu * constraint_l1_norm
    candidate_merits = jax.vmap(candidate_merit, in_axes=(0, None))(
        alpha_candidates, linesearch_mu
    )
    decrease_values = candidate_merits - (
        merit_value + eta * alpha_candidates * merit_derivative
    )

    # Use the largest Armijo step, or the lowest finite merit if none pass.
    finite = (
        jnp.isfinite(alpha_candidates)
        & jnp.isfinite(candidate_merits)
        & jnp.isfinite(decrease_values)
    )
    accepted = finite & (decrease_values <= 0)
    accepted_index = jnp.max(jnp.where(accepted, jnp.arange(len(alpha_candidates)), -1))
    restoration_index = jnp.argmin(jnp.where(finite, candidate_merits, jnp.inf))
    fallback_index = jnp.where(jnp.any(finite), restoration_index, 0)
    index = jnp.where(accepted_index >= 0, accepted_index, fallback_index)
    return alpha_candidates[index]


def filter_step(
    alpha_candidates: jnp.ndarray,
    theta0: jnp.ndarray,
    phi0: jnp.ndarray,
    thetas: jnp.ndarray,
    phis: jnp.ndarray,
    linesearch_params: Dict[str, Any],
) -> jnp.ndarray:
    """Return the largest filter-accepted candidate step."""
    constraint_reduction_factor = linesearch_params.get(
        "constraint_reduction_factor", 1.0e-5
    )
    objective_reduction_factor = linesearch_params.get(
        "objective_reduction_factor", 1.0e-5
    )
    finite = jnp.isfinite(alpha_candidates) & jnp.isfinite(thetas) & jnp.isfinite(phis)
    accepted = finite & (
        (thetas <= (1.0 - constraint_reduction_factor) * theta0)
        | (phis <= phi0 - objective_reduction_factor * theta0)
    )

    accepted_index = jnp.argmax(jnp.where(accepted, alpha_candidates, -jnp.inf))
    restoration_index = jnp.argmin(jnp.where(finite, thetas, jnp.inf))
    smallest_index = jnp.argmin(
        jnp.where(jnp.isfinite(alpha_candidates), alpha_candidates, jnp.inf)
    )
    fallback_index = jnp.where(jnp.any(finite), restoration_index, smallest_index)
    index = jnp.where(jnp.any(accepted), accepted_index, fallback_index)
    return alpha_candidates[index]


def _objective_value(
    program: OptimalControlProblem,
    states: jnp.ndarray,
    controls: jnp.ndarray,
    slacks: jnp.ndarray,
    problem_params: Dict[str, Any],
) -> jnp.ndarray:
    if program.use_slack_variables:
        return program.cost_with_slack(states, controls, slacks, problem_params)
    return program.cost(states, controls, problem_params)


def filter_linesearch(
    program: OptimalControlProblem,
    problem_params: Dict[str, Any],
    linesearch_params: Dict[str, Any],
    states: jnp.ndarray,
    controls: jnp.ndarray,
    slacks: jnp.ndarray,
    states_new: jnp.ndarray,
    controls_new: jnp.ndarray,
    slacks_new: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Choose the largest step that improves constraints or objective.

    If no step qualifies, choose the finite step with the least constraint
    violation.
    """
    alpha_candidates = jnp.asarray(linesearch_params["alphas"], dtype=states.dtype)
    states_delta = states_new - states
    controls_delta = controls_new - controls
    slacks_delta = slacks_new - slacks

    def theta_phi(alpha):
        candidate_states = states + alpha * states_delta
        candidate_controls = controls + alpha * controls_delta
        candidate_slacks = slacks + alpha * slacks_delta
        eq_l1, ineq_l1, *_ = evaluate_constraints_with_bounds(
            program,
            candidate_states,
            candidate_controls,
            candidate_slacks,
            problem_params,
            norm=1,
        )
        phi = _objective_value(
            program,
            candidate_states,
            candidate_controls,
            candidate_slacks,
            problem_params,
        )
        return eq_l1 + ineq_l1, phi

    theta0, phi0 = theta_phi(jnp.asarray(0.0, dtype=states.dtype))
    thetas, phis = jax.vmap(theta_phi)(alpha_candidates)
    alpha = filter_step(alpha_candidates, theta0, phi0, thetas, phis, linesearch_params)
    return (
        states + alpha * states_delta,
        controls + alpha * controls_delta,
        slacks + alpha * slacks_delta,
    ), alpha


def _merit_linesearch(
    program: OptimalControlProblem,
    problem_params: Dict[str, Any],
    linesearch_params: Dict[str, Any],
    states: jnp.ndarray,
    controls: jnp.ndarray,
    slacks: jnp.ndarray,
    states_new: jnp.ndarray,
    controls_new: jnp.ndarray,
    slacks_new: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Choose an SQP step with an Armijo merit test."""
    horizon = program.horizon
    nx = program.num_state_variables
    nu = program.num_control_variables
    states_delta = states_new - states
    controls_delta = controls_new - controls
    slacks_delta = slacks_new - slacks
    z = jnp.concatenate([states, controls, slacks], axis=-1).flatten()
    z_delta = jnp.concatenate(
        [states_delta, controls_delta, slacks_delta], axis=-1
    ).flatten()

    def cost_function(z_flat):
        z_mat = jnp.reshape(z_flat, (horizon + 1, -1))
        cost_value = _objective_value(
            program,
            z_mat[:, :nx],
            z_mat[:, nx : (nx + nu)],
            z_mat[:, (nx + nu) :],
            problem_params,
        )
        return cost_value * jnp.ones(1)

    cost, cost_dz = value_and_jacrev(cost_function, z)
    cost, cost_dz = cost[0], cost_dz[0]
    cost_dz_times_delta_z = jnp.dot(cost_dz, z_delta)
    eq_l1, ineq_l1, *_ = evaluate_constraints_with_bounds(
        program, states, controls, slacks, problem_params, norm=1
    )
    constraints_l1_norm = eq_l1 + ineq_l1

    def candidate_merit(alpha, linesearch_mu):
        candidate_states = states + alpha * states_delta
        candidate_controls = controls + alpha * controls_delta
        candidate_slacks = slacks + alpha * slacks_delta
        candidate_cost = _objective_value(
            program,
            candidate_states,
            candidate_controls,
            candidate_slacks,
            problem_params,
        )
        eq_l1, ineq_l1, *_ = evaluate_constraints_with_bounds(
            program,
            candidate_states,
            candidate_controls,
            candidate_slacks,
            problem_params,
            norm=1,
        )
        return candidate_cost + linesearch_mu * (eq_l1 + ineq_l1)

    alpha = backtracking_merit_step(
        cost,
        cost_dz_times_delta_z,
        constraints_l1_norm,
        candidate_merit,
        linesearch_params,
    )
    return (
        states + alpha * states_delta,
        controls + alpha * controls_delta,
        slacks + alpha * slacks_delta,
    ), alpha


def backtracking_linesearch(
    program: OptimalControlProblem,
    problem_params: Dict[str, Any],
    solver_params: Dict[str, Any],
    states: jnp.ndarray,
    controls: jnp.ndarray,
    slacks: jnp.ndarray,
    states_new: jnp.ndarray,
    controls_new: jnp.ndarray,
    slacks_new: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Compatibility wrapper for the legacy merit linesearch parameters."""
    linesearch_params = {
        "alphas": solver_params["linesearch_alphas"],
        "armijo_coefficient": solver_params["linesearch_eta"],
        "penalty_min": solver_params.get("linesearch_mu_min", 0.1),
        "penalty_max": solver_params.get("linesearch_mu_max", 1.0e6),
        "constraint_norm_floor": solver_params.get(
            "linesearch_constraint_floor", 1.0e-8
        ),
    }
    return _merit_linesearch(
        program,
        problem_params,
        linesearch_params,
        states,
        controls,
        slacks,
        states_new,
        controls_new,
        slacks_new,
    )


def linesearch(
    method: LinesearchMethod,
    program: OptimalControlProblem,
    problem_params: Dict[str, Any],
    linesearch_params: Dict[str, Any],
    states: jnp.ndarray,
    controls: jnp.ndarray,
    slacks: jnp.ndarray,
    states_new: jnp.ndarray,
    controls_new: jnp.ndarray,
    slacks_new: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Apply line search."""
    if not isinstance(method, LinesearchMethod):
        raise TypeError(
            "linesearch method must be a statically configured LinesearchMethod."
        )
    if method == LinesearchMethod.NONE:
        return (
            states_new,
            controls_new,
            slacks_new,
        ), jnp.asarray(1.0, dtype=states.dtype)
    if method == LinesearchMethod.MERIT:
        implementation = _merit_linesearch
        method_params = linesearch_params["merit"]
    elif method == LinesearchMethod.FILTER:
        implementation = filter_linesearch
        method_params = linesearch_params["filter"]
    else:
        raise ValueError(f"Unsupported linesearch method: {method!r}")
    method_params = {**method_params, "alphas": linesearch_params["alphas"]}
    return implementation(
        program,
        problem_params,
        method_params,
        states,
        controls,
        slacks,
        states_new,
        controls_new,
        slacks_new,
    )


class BarrierLinesearchTerms(NamedTuple):
    """Barrier quantities the log-barrier line search interpolates alongside x.

    ``s_g`` is the outer barrier image and ``s_g_qp`` the inner-QP result
    mapped back to the original constraint coordinates. Both are strictly
    inside the finite bounds.
    """

    s_g: jnp.ndarray  # (N + 1, m)
    s_g_qp: jnp.ndarray  # (N + 1, m)
    lower: jnp.ndarray  # (N + 1, m)
    upper: jnp.ndarray  # (N + 1, m)
    log_barrier_kappa: jnp.ndarray  # scalar
    fraction_to_boundary: float


def _barrier_theta_phi(
    program: OptimalControlProblem,
    problem_params: Dict[str, Any],
    states: jnp.ndarray,
    controls: jnp.ndarray,
    slacks: jnp.ndarray,
    s_g: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    log_barrier_kappa: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return the constraint violation and barrier objective at one iterate.

    ``theta`` measures inequality consistency against ``s_g`` instead of a
    bound violation; ``phi`` carries the two-sided barrier on ``s_g``.
    """
    equality = program.equality_constraints(states, controls, problem_params)
    values, _, _ = program.inequality_constraints(states, controls, problem_params)
    values = values.reshape(s_g.shape)
    if program.use_slack_variables:
        values = values + slacks
    theta = jnp.sum(jnp.abs(equality.reshape(-1))) + jnp.sum(
        jnp.abs((values - s_g).reshape(-1))
    )
    phi = _objective_value(program, states, controls, slacks, problem_params)
    return theta, phi + barrier_value(s_g, lower, upper, log_barrier_kappa)


def _barrier_step_functions(
    program: OptimalControlProblem,
    problem_params: Dict[str, Any],
    states: jnp.ndarray,
    controls: jnp.ndarray,
    slacks: jnp.ndarray,
    states_new: jnp.ndarray,
    controls_new: jnp.ndarray,
    slacks_new: jnp.ndarray,
    barrier: BarrierLinesearchTerms,
    linesearch_params: Dict[str, Any],
) -> Tuple[Callable, jnp.ndarray, Callable]:
    """Return ``(candidate, alphas, theta_phi)`` shared by both barrier rules.

    ``alphas`` is the configured grid scaled by the two-sided
    fraction-to-boundary maximum, so every candidate ``s_g`` stays strictly
    inside the finite bounds.
    """
    states_delta = states_new - states
    controls_delta = controls_new - controls
    slacks_delta = slacks_new - slacks
    s_g_delta = barrier.s_g_qp - barrier.s_g
    max_step = fraction_to_boundary_step(
        barrier.s_g,
        s_g_delta,
        barrier.lower,
        barrier.upper,
        barrier.fraction_to_boundary,
    )

    def candidate(alpha):
        return (
            states + alpha * states_delta,
            controls + alpha * controls_delta,
            slacks + alpha * slacks_delta,
            barrier.s_g + alpha * s_g_delta,
        )

    def theta_phi(alpha):
        return _barrier_theta_phi(
            program,
            problem_params,
            *candidate(alpha),
            barrier.lower,
            barrier.upper,
            barrier.log_barrier_kappa,
        )

    alphas = jnp.asarray(linesearch_params["alphas"], dtype=states.dtype)
    return candidate, max_step * alphas, theta_phi


def _barrier_merit_linesearch(
    program: OptimalControlProblem,
    problem_params: Dict[str, Any],
    linesearch_params: Dict[str, Any],
    states: jnp.ndarray,
    controls: jnp.ndarray,
    slacks: jnp.ndarray,
    states_new: jnp.ndarray,
    controls_new: jnp.ndarray,
    slacks_new: jnp.ndarray,
    barrier: BarrierLinesearchTerms,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Choose a log-barrier SQP step with an Armijo merit test.

    The objective carries the two-sided barrier and the constraint norm
    measures inequality consistency against ``s_g`` instead of a bound
    violation.
    """
    candidate, alphas, theta_phi = _barrier_step_functions(
        program,
        problem_params,
        states,
        controls,
        slacks,
        states_new,
        controls_new,
        slacks_new,
        barrier,
        linesearch_params,
    )

    def candidate_merit(alpha, penalty):
        theta, phi = theta_phi(alpha)
        return phi + penalty * theta

    zero = jnp.asarray(0.0, dtype=states.dtype)
    theta0, _ = theta_phi(zero)
    phi0, phi_derivative = jax.value_and_grad(lambda alpha: theta_phi(alpha)[1])(zero)
    alpha = backtracking_merit_step(
        phi0,
        phi_derivative,
        theta0,
        candidate_merit,
        {**linesearch_params, "alphas": alphas},
    )
    return candidate(alpha), alpha


def _barrier_filter_linesearch(
    program: OptimalControlProblem,
    problem_params: Dict[str, Any],
    linesearch_params: Dict[str, Any],
    states: jnp.ndarray,
    controls: jnp.ndarray,
    slacks: jnp.ndarray,
    states_new: jnp.ndarray,
    controls_new: jnp.ndarray,
    slacks_new: jnp.ndarray,
    barrier: BarrierLinesearchTerms,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Choose a log-barrier step with the shared stateless filter rule."""
    candidate, alphas, theta_phi = _barrier_step_functions(
        program,
        problem_params,
        states,
        controls,
        slacks,
        states_new,
        controls_new,
        slacks_new,
        barrier,
        linesearch_params,
    )

    zero = jnp.asarray(0.0, dtype=states.dtype)
    theta0, phi0 = theta_phi(zero)
    thetas, phis = jax.vmap(theta_phi)(alphas)
    alpha = filter_step(alphas, theta0, phi0, thetas, phis, linesearch_params)
    return candidate(alpha), alpha


def barrier_linesearch(
    method: LinesearchMethod,
    program: OptimalControlProblem,
    problem_params: Dict[str, Any],
    linesearch_params: Dict[str, Any],
    states: jnp.ndarray,
    controls: jnp.ndarray,
    slacks: jnp.ndarray,
    states_new: jnp.ndarray,
    controls_new: jnp.ndarray,
    slacks_new: jnp.ndarray,
    barrier: BarrierLinesearchTerms,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Apply the log-barrier line search, interpolating ``s_g`` alongside x.

    Returns ``((states, controls, slacks, s_g), alpha)``. Every step is limited
    by the two-sided fraction-to-boundary maximum, so ``s_g`` stays inside the
    finite bounds.
    """
    if not isinstance(method, LinesearchMethod):
        raise TypeError(
            "linesearch method must be a statically configured LinesearchMethod."
        )
    if method == LinesearchMethod.NONE:
        s_g_delta = barrier.s_g_qp - barrier.s_g
        alpha = fraction_to_boundary_step(
            barrier.s_g,
            s_g_delta,
            barrier.lower,
            barrier.upper,
            barrier.fraction_to_boundary,
        )
        return (
            states + alpha * (states_new - states),
            controls + alpha * (controls_new - controls),
            slacks + alpha * (slacks_new - slacks),
            barrier.s_g + alpha * s_g_delta,
        ), alpha
    if method == LinesearchMethod.MERIT:
        implementation = _barrier_merit_linesearch
        method_params = linesearch_params["merit"]
    elif method == LinesearchMethod.FILTER:
        implementation = _barrier_filter_linesearch
        method_params = linesearch_params["filter"]
    else:
        raise ValueError(f"Unsupported log-barrier linesearch method: {method!r}")
    return implementation(
        program,
        problem_params,
        {**method_params, "alphas": linesearch_params["alphas"]},
        states,
        controls,
        slacks,
        states_new,
        controls_new,
        slacks_new,
        barrier,
    )
