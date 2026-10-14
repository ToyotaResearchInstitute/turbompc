"""Finite differences vs automatic differentiation tests.

Uses DIRECT_JAX_DENSE backward so it runs on CPU. The CUDSS direct backward path
uses the same backward formulas with a different inner KKT solver.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.constraint_handling_fixtures import (
    handlings,
    linear_problem,
    obstacle_problem,
    with_handling,
)
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.status import SQPStatus
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    HessianApproximation,
    TurboMPCSolver,
)
from turbompc.utils.load_params import load_solver_params


def _base_spacecraft_params(
    *,
    horizon: int = 6,
    use_slack: bool = False,
    bounded: bool = True,
    rescale: bool = False,
) -> Dict[str, Any]:
    nx, nu = 3, 3
    # Use a single vector inertia so we can perturb it in FD tests.
    inertia_vec = jnp.array([5.0, 2.0, 1.0], dtype=jnp.float64)
    inertia_tiled = jnp.broadcast_to(inertia_vec[None, :], (horizon + 1, 3))
    bound = 0.5
    params: Dict[str, Any] = {
        "horizon": horizon,
        "discretization_resolution": 0.1,
        "discretization_scheme": 0,  # EULER
        "initial_state": jnp.array([0.1, -0.15, 0.08], dtype=jnp.float64),
        "initial_guess_final_state": jnp.zeros(nx, dtype=jnp.float64),
        "reference_state_trajectory": jnp.zeros((horizon + 1, nx), dtype=jnp.float64),
        "reference_control_trajectory": jnp.zeros((horizon + 1, nu), dtype=jnp.float64),
        "penalize_control_reference": False,
        "rescale_optimization_variables": rescale,
        "state_rescaling_min": jnp.array([-2.0, -4.0, -8.0], dtype=jnp.float64),
        "state_rescaling_max": jnp.array([2.0, 4.0, 8.0], dtype=jnp.float64),
        "control_rescaling_min": jnp.array([-0.5, -1.0, -2.0], dtype=jnp.float64),
        "control_rescaling_max": jnp.array([0.5, 1.0, 2.0], dtype=jnp.float64),
        "constrain_initial_control": False,
        "initial_control": jnp.zeros(nu, dtype=jnp.float64),
        "weights_penalization_reference_state_trajectory": jnp.ones(
            nx, dtype=jnp.float64
        ),
        "weights_penalization_final_state": jnp.zeros(nx, dtype=jnp.float64),
        "weights_penalization_control_squared": jnp.full(nu, 0.1, dtype=jnp.float64),
        "weights_penalization_control_rate": jnp.zeros(nu, dtype=jnp.float64),
        "weights_linear_penalization_final_state": jnp.zeros(nx, dtype=jnp.float64),
        "dynamics_state_dot_params": {"inertia_vector": inertia_tiled},
    }
    if bounded:
        params["state_min_bounds"] = jnp.full(nx, -bound, dtype=jnp.float64)
        params["state_max_bounds"] = jnp.full(nx, bound, dtype=jnp.float64)
        params["control_min_bounds"] = jnp.full(nu, -bound, dtype=jnp.float64)
        params["control_max_bounds"] = jnp.full(nu, bound, dtype=jnp.float64)
    if use_slack:
        params["use_slack_variables"] = True
        params["slack_penalization_weight"] = jnp.asarray(100.0, dtype=jnp.float64)
    return params


def _tight_solver_params(
    log_barrier_kappa: float = 0.0, linesearch: str = "none"
) -> Dict[str, Any]:
    # The dense Schur solve lets ADMM reach residuals of ~1e-11. The SQP error
    # floor scales with the multipliers (~3e-10 with large slacks), so tighter
    # SQP tolerances only exhaust the iteration limit.
    sp = load_solver_params("turbompc.yaml")
    sp["num_sqp_iteration_max"] = 60
    sp["tol_convergence"] = 1e-9
    sp["linesearch"]["method"] = linesearch
    sp["warm_start_backward"] = True
    sp["admm"]["max_iter"] = 5000
    sp["admm"]["eps_abs"] = 1e-10
    sp["admm"]["eps_rel"] = 1e-10
    sp["admm"]["check_termination_every"] = 1
    return with_handling(sp, log_barrier_kappa)


def _turbompc_solver(
    problem,
    *,
    use_full_hessian: bool,
    log_barrier_kappa: float = 0.0,
    linesearch: str = "none",
) -> TurboMPCSolver:
    # The problems here are small, so a dense Schur solve is exact and fast.
    return TurboMPCSolver(
        program=problem,
        params=_tight_solver_params(log_barrier_kappa, linesearch),
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE,
        backward_hessian=(
            HessianApproximation.FULL_HESSIAN
            if use_full_hessian
            else HessianApproximation.GAUSS_NEWTON
        ),
    )


def _make_solver(
    *,
    use_slack: bool,
    bounded: bool = True,
    horizon: int = 6,
    rescale: bool = False,
):
    # The spacecraft dynamics are nonlinear, so only the full-Hessian backward
    # pass is the exact gradient that finite differences can check tightly.
    params = _base_spacecraft_params(
        horizon=horizon, use_slack=use_slack, bounded=bounded, rescale=rescale
    )
    problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    return _turbompc_solver(problem, use_full_hessian=True), params


def _fd_grad_scalar_at_leaf(
    objective_fn: Callable[[Dict[str, Any]], jnp.ndarray],
    weights: Dict[str, Any],
    key_path: tuple,
    flat_idx: int,
    eps: float,
) -> float:
    """Central-difference gradient of `objective_fn(weights)` w.r.t. the
    scalar at `weights[key_path[0]][key_path[1]]...[key_path[-1]].ravel()[flat_idx]`.

    Works with arbitrarily-nested dict weights.
    """

    def _get(tree, path):
        for k in path:
            tree = tree[k]
        return tree

    def _set(tree, path, val):
        if not path:
            return val
        out = dict(tree)
        out[path[0]] = _set(tree[path[0]], path[1:], val)
        return out

    leaf = jnp.asarray(_get(weights, key_path))
    flat = leaf.ravel()

    def _perturbed(delta):
        new_flat = flat.at[flat_idx].add(delta)
        new_leaf = new_flat.reshape(leaf.shape)
        new_weights = _set(weights, key_path, new_leaf)
        return float(objective_fn(new_weights).sum())

    v_plus = _perturbed(eps)
    v_minus = _perturbed(-eps)
    return (v_plus - v_minus) / (2.0 * eps)


def _compare_grad(
    objective_fn,
    weights: Dict[str, Any],
    grad_ad: Dict[str, Any],
    eps: float = 1e-4,
    rel_tol: float = 1e-5,
    abs_tol: float = 1e-9,
    include_paths: list[tuple] | None = None,
    flat_indices_by_path: Dict[tuple, list[int]] | None = None,
) -> Dict[str, float]:
    """Compare AD grad to FD for each leaf at tree depth 1 or 2.

    Per-element check uses numpy-style hybrid tolerance:
        |fd_i - ad_i| <= abs_tol + rel_tol * max(|fd_i|, |ad_i|)
    `abs_tol` absorbs the FD/SQP noise floor on small-magnitude components;
    `rel_tol` catches multiplicative errors on large components. The reported
    "rel_err" per path is the max over indices of
        |fd_i - ad_i| / (abs_tol + rel_tol * max(|fd_i|, |ad_i|))
    so the path passes iff the reported value is < 1.0.

    If include_paths is given, only those paths are checked (tuples of keys).
    Otherwise every top-level key in weights is checked.
    """
    results: Dict[str, float] = {}

    if include_paths is None:
        include_paths = [(k,) for k in weights.keys()]

    for path in include_paths:
        try:
            ad_leaf = grad_ad
            for k in path:
                ad_leaf = ad_leaf[k]
            ad_leaf = jnp.asarray(ad_leaf)
        except KeyError:
            results[".".join(str(p) for p in path)] = float("inf")
            continue

        leaf_flat = ad_leaf.ravel()
        n_total = leaf_flat.size
        indices = (
            flat_indices_by_path.get(path) if flat_indices_by_path is not None else None
        )
        if indices is None:
            indices = list(range(n_total))
        worst_score = 0.0
        fd_by_idx: Dict[int, float] = {}
        for i in indices:
            fd_i = _fd_grad_scalar_at_leaf(objective_fn, weights, path, i, eps)
            fd_by_idx[i] = fd_i
            ad_i = float(leaf_flat[i])
            denom = abs_tol + rel_tol * max(abs(fd_i), abs(ad_i))
            score = abs(fd_i - ad_i) / denom
            if score > worst_score:
                worst_score = score

        results[".".join(str(p) for p in path)] = worst_score
    return results


def _objective_from_solver(solver, params):
    guess = solver.initial_guess(params)
    # Gradients of an unconverged solve are not the implicit-function gradients.
    assert int(solver.solve(guess, params).status) == int(SQPStatus.SOLVED)

    def objective(weights):
        sol = solver.solve(guess, params, weights)
        # Scalar objective: sum of states-squared + controls-squared
        return jnp.sum(sol.states**2) + jnp.sum(sol.controls**2)

    return objective


@pytest.mark.extended
def test_slack_weight_grad_direct():
    # Tight bounds => slack active.
    horizon = 5
    params = _base_spacecraft_params(horizon=horizon, use_slack=True, bounded=True)
    bound = 0.05
    params["state_min_bounds"] = jnp.full(3, -bound, dtype=jnp.float64)
    params["state_max_bounds"] = jnp.full(3, bound, dtype=jnp.float64)
    params["initial_state"] = jnp.array([0.1, -0.15, 0.08], dtype=jnp.float64)

    problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    solver = _turbompc_solver(problem, use_full_hessian=True)
    objective = _objective_from_solver(solver, params)

    weights = {
        "slack_penalization_weight": jnp.asarray(100.0, dtype=jnp.float64),
        "weights_penalization_reference_state_trajectory": jnp.asarray(
            params["weights_penalization_reference_state_trajectory"]
        ),
    }

    grad_ad = jax.grad(objective)(weights)
    results = _compare_grad(objective, weights, grad_ad)
    assert results["slack_penalization_weight"] < 1.0, results


# Each test forces a known active-set composition (all-lower, all-upper, or
# mixed) and checks `dL/dγ` from the slack-eliminated backward IFT against FD.


def _make_slack_solver(
    *,
    initial_state,
    state_bound: float,
    horizon: int = 5,
    control_bound: float = 1e3,
):
    """Build a spacecraft+slack solver with a state-bound that the initial
    state explicitly violates, so slack is forced active on the state side.

    `initial_state` sits outside [-state_bound, state_bound] on every
    component whose sign you care about (positive → upper-active,
    negative → lower-active). `control_bound` is left wide by default so
    the slack-active set is determined by the state-bound violation only,
    not by control saturation."""
    params = _base_spacecraft_params(horizon=horizon, use_slack=True, bounded=True)
    params["state_min_bounds"] = jnp.full(3, -state_bound, dtype=jnp.float64)
    params["state_max_bounds"] = jnp.full(3, state_bound, dtype=jnp.float64)
    params["control_min_bounds"] = jnp.full(3, -control_bound, dtype=jnp.float64)
    params["control_max_bounds"] = jnp.full(3, control_bound, dtype=jnp.float64)
    params["initial_state"] = jnp.asarray(initial_state, dtype=jnp.float64)
    problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    return _turbompc_solver(problem, use_full_hessian=True), params


def _classify_active_set(sol) -> tuple[int, int]:
    """Returns (n_lower, n_upper) — counts of active inequalities at the SQP
    solution. Used to assert each test setup actually exercises the intended
    active-set composition."""
    assert sol.kkt_state is not None, "solver did not return kkt_state"
    n_lower = int(jnp.asarray(sol.kkt_state.ineq_active_lower_idx).sum())
    n_upper = int(jnp.asarray(sol.kkt_state.ineq_active_upper_idx).sum())
    return n_lower, n_upper


def _slack_grad_check(
    solver, params, expect_lower: bool, expect_upper: bool, label: str
):
    # Sanity: solve once to confirm setup actually exercises the intended
    # active-set composition before doing the (expensive) FD vs AD comparison.
    sol = solver.solve(solver.initial_guess(params), params, {})
    n_lower, n_upper = _classify_active_set(sol)
    if expect_lower:
        assert n_lower > 0, f"{label}: setup produced no lower-active constraints"
    else:
        assert (
            n_lower == 0
        ), f"{label}: setup produced unexpected lower-active (n={n_lower})"
    if expect_upper:
        assert n_upper > 0, f"{label}: setup produced no upper-active constraints"
    else:
        assert (
            n_upper == 0
        ), f"{label}: setup produced unexpected upper-active (n={n_upper})"

    objective = _objective_from_solver(solver, params)
    weights = {
        "slack_penalization_weight": jnp.asarray(100.0, dtype=jnp.float64),
        # Pair with a second weight to dodge jacfwd({}) bug when slack is alone.
        "weights_penalization_reference_state_trajectory": jnp.asarray(
            params["weights_penalization_reference_state_trajectory"]
        ),
    }
    grad_ad = jax.grad(objective)(weights)
    results = _compare_grad(
        objective,
        weights,
        grad_ad,
        include_paths=[("slack_penalization_weight",)],
    )
    assert results["slack_penalization_weight"] < 1.0, results


def test_slack_grad_lower_active_only():
    solver, params = _make_slack_solver(
        initial_state=(-0.15, -0.15, -0.15), state_bound=0.1
    )
    _slack_grad_check(
        solver, params, expect_lower=True, expect_upper=False, label="slack lower-only"
    )


def test_slack_grad_upper_active_only():
    solver, params = _make_slack_solver(
        initial_state=(0.15, 0.15, 0.15), state_bound=0.1
    )
    _slack_grad_check(
        solver, params, expect_lower=False, expect_upper=True, label="slack upper-only"
    )


def test_slack_grad_mixed_active():
    solver, params = _make_slack_solver(
        initial_state=(0.15, -0.15, 0.15), state_bound=0.1
    )
    _slack_grad_check(
        solver, params, expect_lower=True, expect_upper=True, label="slack mixed"
    )


def test_linear_terminal_cost_grad_full_hessian_rescaled():
    solver, params = _make_solver(
        use_slack=False, bounded=False, horizon=5, rescale=True
    )
    objective = _objective_from_solver(solver, params)

    weights = {
        "weights_linear_penalization_final_state": jnp.array(
            [0.1, 0.2, 0.3], dtype=jnp.float64
        ),
    }

    grad_ad = jax.grad(objective)(weights)
    results = _compare_grad(objective, weights, grad_ad)
    assert results["weights_linear_penalization_final_state"] < 1.0, results


def test_dynamics_state_dot_params_grad_full_hessian_rescaled():
    solver, params = _make_solver(
        use_slack=False, bounded=False, horizon=5, rescale=True
    )
    objective = _objective_from_solver(solver, params)

    # Nested dict weight with tiled (H+1, 3) inertia matching what the solver expects.
    inertia_init = params["dynamics_state_dot_params"]["inertia_vector"]
    weights = {
        "dynamics_state_dot_params": {
            "inertia_vector": jnp.asarray(inertia_init),
        },
    }

    grad_ad = jax.grad(objective)(weights)
    results = _compare_grad(
        objective,
        weights,
        grad_ad,
        include_paths=[("dynamics_state_dot_params", "inertia_vector")],
        flat_indices_by_path={
            ("dynamics_state_dot_params", "inertia_vector"): [
                0,
                4,
                inertia_init.size - 1,
            ],
        },
    )
    assert results["dynamics_state_dot_params.inertia_vector"] < 1.0, results


@pytest.mark.extended
def test_combined_slack_full_hessian_multi_weight():
    """Exercise slack + full_hessian + multiple array weights together.

    The slack is inactive here, so ``slack_penalization_weight`` has a zero
    gradient; the slack_grad_* tests cover it with active slacks.
    """
    solver, params = _make_solver(use_slack=True, bounded=True, horizon=5)
    objective = _objective_from_solver(solver, params)

    weights = {
        "weights_penalization_reference_state_trajectory": jnp.asarray(
            params["weights_penalization_reference_state_trajectory"]
        ),
        "weights_penalization_control_squared": jnp.asarray(
            params["weights_penalization_control_squared"]
        ),
        "weights_linear_penalization_final_state": jnp.array(
            [0.1, 0.2, 0.3], dtype=jnp.float64
        ),
    }

    grad_ad = jax.grad(objective)(weights)
    results = _compare_grad(objective, weights, grad_ad)
    worst = max(results.values())
    assert worst < 1.0, f"worst score = {worst:.3e}, results={results}"


# Parameters that enter the inequality rows, for projection and log-barrier
# handling. The linear problem's bounds and the obstacle are active at the
# solution, so every checked gradient flows through the inequality terms.


def _assert_gradient_matches_finite_differences(
    solver, params, key, indices, *, eps, rtol
):
    """Compare ``d loss / d params[key]`` at ``indices`` with central differences."""
    guess = solver.initial_guess(params)
    value = jnp.asarray(params[key])
    assert int(solver.solve(guess, params).status) == int(SQPStatus.SOLVED)

    def objective(v):
        solution = solver.solve(guess, params, weights={key: v})
        return (
            jnp.sum(solution.states**2)
            + jnp.sum(solution.controls**2)
            + jnp.sum(solution.slack**2)
        )

    gradient = np.asarray(jax.grad(objective)(value)).ravel()
    for index in indices:
        step = np.zeros(value.size)
        step[index] = eps
        step = jnp.asarray(step.reshape(value.shape))
        expected = float(objective(value + step) - objective(value - step)) / (2 * eps)
        scale = max(abs(expected), abs(gradient[index]), 1.0e-30)
        assert abs(expected - gradient[index]) / scale < rtol, (
            f"{key}[{index}]: analytic {gradient[index]:.8e} vs finite difference"
            f" {expected:.8e}"
        )


@pytest.mark.parametrize(
    ("constraint_handling", "log_barrier_kappa"), handlings(1.0e-2)
)
@pytest.mark.parametrize("use_slack", [False, True])
@pytest.mark.parametrize(
    ("key", "indices"),
    [
        ("initial_state", [0, 2]),
        # The active lower velocity and control bounds; see ``linear_problem``.
        ("state_min_bounds", [2]),
        ("control_min_bounds", [0]),
    ],
)
def test_bound_and_initial_state_gradients(
    constraint_handling, log_barrier_kappa, use_slack, key, indices
):
    # A wrong or missing backward term shows up at the percent level; 1e-4 is
    # above the finite-difference noise of the 1e-9 SQP tolerance.
    problem, params = linear_problem(use_slack=use_slack)
    _assert_gradient_matches_finite_differences(
        _turbompc_solver(
            problem, use_full_hessian=False, log_barrier_kappa=log_barrier_kappa
        ),
        params,
        key,
        indices,
        eps=1.0e-6,
        rtol=1.0e-4,
    )


@pytest.mark.parametrize(
    ("constraint_handling", "log_barrier_kappa"), handlings(1.0e-3)
)
@pytest.mark.parametrize(
    ("key", "indices"),
    [
        ("obstacles_radii", [0]),
        # x and y of the terminal-stage center, the only stage touching the disk.
        ("obstacles_centers", [10, 11]),
    ],
)
def test_nonlinear_inequality_parameter_gradients(
    constraint_handling, log_barrier_kappa, key, indices
):
    """A nonlinear inequality needs the exact Lagrangian Hessian.

    ``obstacles_centers`` enters ``G`` itself, so it exercises the mixed second
    derivative that Gauss-Newton drops; ``obstacles_radii`` only shifts ``g``.
    """
    problem, params = obstacle_problem()
    solver = _turbompc_solver(
        problem,
        use_full_hessian=True,
        log_barrier_kappa=log_barrier_kappa,
        linesearch="filter",
    )
    _assert_gradient_matches_finite_differences(
        solver, params, key, indices, eps=1.0e-4, rtol=1.0e-5
    )
