import copy

import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

from tests.helpers.problem_fixtures import make_linear_params
from tests.helpers.solver_fixtures import turbompc_solver_params
from turbompc.dynamics.base_dynamics import Dynamics
from turbompc.dynamics.integrators import DiscretizationScheme
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers import linesearch as linesearch_module
from turbompc.solvers.linesearch import LinesearchMethod, parse_linesearch_method
from turbompc.solvers.status import SQPStatus
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    HessianApproximation,
    TurboMPCSolver,
)
from turbompc.utils.load_params import load_solver_params

config.update("jax_enable_x64", True)


class _QuadraticImplicitDynamics(Dynamics):
    def __init__(self):
        super().__init__(
            {
                "num_states": 1,
                "num_controls": 1,
                "names_states": ["x"],
                "names_controls": ["u"],
            }
        )

    def state_dot(self, state, control, params):
        del control, params
        return jnp.array([state[0] ** 2 - 1.0])


def _linear_problem():
    dynamics, params = make_linear_params(
        horizon=3,
        implicit=False,
        bounded=True,
        initial_state=jnp.ones((4,), dtype=jnp.float32),
    )
    return dynamics, params


def _make_jax_solver(problem, solver_params):
    return TurboMPCSolver(
        program=problem,
        params=solver_params,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
    )


def _hard_rescaled_linesearch_fixture(linesearch_method):
    problem_params = {
        "horizon": 1,
        "discretization_resolution": 1.0,
        "discretization_scheme": int(DiscretizationScheme.IMPLICIT),
        "initial_state": jnp.array([0.0]),
        "initial_guess_final_state": jnp.array([0.9]),
        "constrain_initial_control": False,
        "initial_control": jnp.zeros((1,)),
        "reference_state_trajectory": jnp.zeros((2, 1)),
        "reference_control_trajectory": jnp.zeros((2, 1)),
        "penalize_control_reference": False,
        "weights_penalization_reference_state_trajectory": jnp.ones((1,)),
        "weights_penalization_final_state": jnp.ones((1,)),
        "weights_penalization_control_squared": jnp.ones((1,)),
        "weights_penalization_control_rate": jnp.zeros((1,)),
        "state_min_bounds": jnp.array([-100.0]),
        "state_max_bounds": jnp.array([100.0]),
        "control_min_bounds": jnp.array([-1000.0]),
        "control_max_bounds": jnp.array([1000.0]),
        "rescale_optimization_variables": True,
        "state_rescaling_min": jnp.array([-10.0]),
        "state_rescaling_max": jnp.array([10.0]),
        "control_rescaling_min": jnp.array([-100.0]),
        "control_rescaling_max": jnp.array([100.0]),
        "dynamics_state_dot_params": {},
    }
    solver_params = dict(load_solver_params("turbompc.yaml"))
    solver_params["linesearch"]["method"] = linesearch_method
    solver_params["num_sqp_iteration_max"] = 6
    solver_params["tol_convergence"] = 1.0e-6
    problem = OptimalControlProblem(
        dynamics=_QuadraticImplicitDynamics(),
        params=problem_params,
    )
    solver = TurboMPCSolver(
        program=problem,
        params=solver_params,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE,
        backward_hessian=HessianApproximation.GAUSS_NEWTON,
    )
    return solver, problem_params


def test_linesearch_module_exposes_hard_projection_strategies():
    assert (
        linesearch_module.filter_linesearch.__module__ == "turbompc.solvers.linesearch"
    )
    assert linesearch_module.linesearch.__module__ == "turbompc.solvers.linesearch"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("none", LinesearchMethod.NONE),
        ("merit", LinesearchMethod.MERIT),
        ("FILTER", LinesearchMethod.FILTER),
        (LinesearchMethod.FILTER, LinesearchMethod.FILTER),
    ],
)
def test_parse_linesearch_method(value, expected):
    assert parse_linesearch_method(value) == expected


def test_parse_linesearch_method_rejects_invalid_values():
    with pytest.raises(TypeError, match="LinesearchMethod"):
        parse_linesearch_method(False)
    with pytest.raises(ValueError, match="Unknown linesearch method"):
        parse_linesearch_method("trust-region")


class _ConstraintNormProgram:
    use_slack_variables = False

    def equality_constraints(self, states, controls, problem_params):
        del states, controls, problem_params
        return jnp.array([3.0, -4.0])

    def inequality_constraints(self, states, controls, problem_params):
        del states, controls, problem_params
        return (
            jnp.array([[-2.0, 5.0]]),
            jnp.array([[0.0, -10.0]]),
            jnp.array([[10.0, 2.0]]),
        )


@pytest.mark.parametrize(
    ("norm", "expected_eq", "expected_ineq"),
    [(1, 7.0, 5.0), (jnp.inf, 4.0, 3.0)],
)
def test_evaluate_constraints_with_bounds_uses_requested_norm(
    norm, expected_eq, expected_ineq
):
    eq, ineq, *_ = linesearch_module.evaluate_constraints_with_bounds(
        _ConstraintNormProgram(),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 0)),
        {},
        norm=norm,
    )

    np.testing.assert_allclose(eq, expected_eq)
    np.testing.assert_allclose(ineq, expected_ineq)


def test_no_linesearch_returns_the_trial_step_and_scalar_alpha():
    states = jnp.zeros((2, 1), dtype=jnp.float32)
    controls = jnp.zeros((2, 1), dtype=jnp.float32)
    slacks = jnp.zeros((2, 0), dtype=jnp.float32)
    trial_states = states + 1.0
    trial_controls = controls + 2.0

    (accepted_states, accepted_controls, accepted_slacks), alpha = (
        linesearch_module.linesearch(
            LinesearchMethod.NONE,
            None,
            {},
            {},
            states,
            controls,
            slacks,
            trial_states,
            trial_controls,
            slacks,
        )
    )

    np.testing.assert_array_equal(accepted_states, trial_states)
    np.testing.assert_array_equal(accepted_controls, trial_controls)
    np.testing.assert_array_equal(accepted_slacks, slacks)
    assert alpha.shape == ()
    assert alpha.dtype == states.dtype
    assert float(alpha) == 1.0


def test_filter_step_selects_the_largest_improving_candidate():
    alpha = linesearch_module.filter_step(
        jnp.asarray([0.25, 0.5, 1.0]),
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        jnp.asarray([0.9, 1.0, 1.0]),
        jnp.asarray([1.0, 0.9, 1.0]),
        {},
    )

    assert float(alpha) == 0.5


def test_filter_step_restores_with_the_least_constraint_violation():
    alpha = linesearch_module.filter_step(
        jnp.asarray([0.25, 0.5, 1.0]),
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        jnp.asarray([2.0, 1.5, 1.8]),
        jnp.asarray([2.0, 2.0, jnp.nan]),
        {},
    )

    assert float(alpha) == 0.5


def test_default_linesearch_config_has_shared_step_grid():
    solver_params = load_solver_params("turbompc.yaml")
    assert solver_params["linesearch"]["method"] == "none"
    np.testing.assert_array_equal(
        np.asarray(solver_params["linesearch"]["alphas"]),
        np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    )


def test_default_constraint_handling_config_is_hard_projection():
    solver_params = load_solver_params("turbompc.yaml")
    dynamics, problem_params = _linear_problem()
    problem = OptimalControlProblem(dynamics, problem_params)

    assert solver_params["constraint_handling"] == "projection"
    assert solver_params["log_barrier_kappa"] == 0.0
    assert problem.use_slack_variables is False
    assert problem.slack_penalization_weight == 0.0


@pytest.mark.parametrize(
    ("solver_updates", "problem_updates", "error", "message"),
    [
        ({"constraint_handling": "invalid"}, {}, ValueError, "constraint_handling"),
        ({"constraint_handling": False}, {}, TypeError, "constraint_handling"),
        ({"log_barrier_kappa": 1.0}, {}, ValueError, "log_barrier_kappa must be zero"),
        (
            {},
            {"use_slack_variables": True},
            ValueError,
            "slack_penalization_weight must be positive",
        ),
        (
            {},
            {"slack_penalization_weight": 1.0},
            ValueError,
            "slack_penalization_weight must be zero",
        ),
    ],
)
def test_constraint_handling_config_validation(
    solver_updates, problem_updates, error, message
):
    dynamics, problem_params = _linear_problem()
    problem_params.update(problem_updates)
    solver_params = turbompc_solver_params()
    solver_params.update(solver_updates)

    with pytest.raises(error, match=message):
        problem = OptimalControlProblem(dynamics, problem_params)
        _make_jax_solver(problem, solver_params)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, jnp.inf, jnp.nan])
def test_log_barrier_rejects_invalid_fraction_to_boundary(fraction):
    dynamics, problem_params = _linear_problem()
    problem = OptimalControlProblem(dynamics, problem_params)
    solver_params = turbompc_solver_params()
    solver_params["constraint_handling"] = "log_barrier"
    solver_params["log_barrier_kappa"] = 1e-4
    solver_params["linesearch"]["fraction_to_boundary"] = fraction

    with pytest.raises(ValueError, match="fraction_to_boundary"):
        _make_jax_solver(problem, solver_params)


def test_turbompc_solver_single_iteration_does_not_relinearize():
    dynamics, problem_params = _linear_problem()
    problem = OptimalControlProblem(
        dynamics=dynamics, params=copy.deepcopy(problem_params)
    )
    solver_params = turbompc_solver_params(tol=1e-6, sqp_iters=1, admm_max=30)
    solver_params["convergence_criterion"] = "first_order"
    solver = _make_jax_solver(problem, solver_params)

    build_count = {"count": 0}
    original_build = solver._build_qp

    def counted_build(*args, **kwargs):
        build_count["count"] += 1
        return original_build(*args, **kwargs)

    solver._build_qp = counted_build
    guess = solver.initial_guess(problem_params)
    assert guess.status == SQPStatus.NOT_RUN
    sol = solver.solve(guess, problem_params)

    assert sol.num_iter == 1
    assert sol.status == SQPStatus.MAX_ITER
    assert jnp.isfinite(sol.convergence_error)
    assert build_count["count"] == 1


def test_turbompc_solver_verbose_single_iteration_does_not_relinearize():
    dynamics, problem_params = _linear_problem()
    problem = OptimalControlProblem(
        dynamics=dynamics, params=copy.deepcopy(problem_params)
    )
    solver_params = turbompc_solver_params(tol=1e-6, sqp_iters=1, admm_max=30)
    solver_params["convergence_criterion"] = "first_order"
    solver_params["verbose"] = True
    solver = _make_jax_solver(problem, solver_params)

    build_count = {"count": 0}
    original_build = solver._build_qp

    def counted_build(*args, **kwargs):
        build_count["count"] += 1
        return original_build(*args, **kwargs)

    solver._build_qp = counted_build
    sol = solver.solve(solver.initial_guess(problem_params), problem_params)

    assert sol.num_iter == 1
    assert build_count["count"] == 1
    assert jnp.isfinite(sol.stats.equality_violations[0])
    assert jnp.isfinite(sol.stats.inequality_violations[0])


def test_turbompc_solver_step_convergence_criterion_is_available():
    dynamics, problem_params = _linear_problem()
    problem = OptimalControlProblem(
        dynamics=dynamics, params=copy.deepcopy(problem_params)
    )
    solver_params = turbompc_solver_params(tol=1e-6, sqp_iters=3, admm_max=30)
    solver_params["convergence_criterion"] = "step"
    solver = _make_jax_solver(problem, solver_params)

    sol = solver.solve(solver.initial_guess(problem_params), problem_params)

    assert sol.status == 0
    assert jnp.isfinite(sol.convergence_error)


def test_hard_filter_linesearch_improves_rescaled_sqp_convergence():
    def solve(linesearch_method):
        solver, problem_params = _hard_rescaled_linesearch_fixture(linesearch_method)
        solution = solver.solve(solver.initial_guess(problem_params), problem_params)
        return solver, problem_params, solution

    _, _, full_step = solve("none")
    solver, _, filtered = solve("filter")

    assert solver.linesearch_method == LinesearchMethod.FILTER
    np.testing.assert_allclose(float(filtered.linesearch_alphas[0]), 0.1)
    assert filtered.num_iter < full_step.num_iter
    assert filtered.convergence_error < 1.0e-6
    assert full_step.convergence_error > 1.0
    assert filtered.convergence_error < 1.0e-3 * full_step.convergence_error
    np.testing.assert_allclose(
        np.asarray(filtered.states)[-1, 0],
        1.0 - np.sqrt(3.0),
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def test_hard_filter_keeps_full_qp_admm_warm_start():
    filter_solver, problem_params = _hard_rescaled_linesearch_fixture("filter")
    filter_solver.params["num_sqp_iteration_max"] = 1
    filtered = filter_solver.solve(
        filter_solver.initial_guess(problem_params), problem_params
    )

    full_step_solver, _ = _hard_rescaled_linesearch_fixture("none")
    full_step_solver.params["num_sqp_iteration_max"] = 1
    full_step = full_step_solver.solve(
        full_step_solver.initial_guess(problem_params), problem_params
    )

    assert filtered.status == SQPStatus.MAX_ITER
    assert full_step.status == SQPStatus.MAX_ITER
    np.testing.assert_allclose(float(filtered.linesearch_alphas[0]), 0.1)
    assert not np.allclose(filtered.states, full_step.states)
    for field in filtered.admm_state._fields:
        np.testing.assert_allclose(
            np.asarray(getattr(filtered.admm_state, field)),
            np.asarray(getattr(full_step.admm_state, field)),
        )
    assert not hasattr(filtered.kkt_state, "slack")


def test_hard_filter_kkt_state_uses_accepted_partial_slack():
    base_solver, problem_params = _hard_rescaled_linesearch_fixture("filter")
    problem_params = copy.deepcopy(problem_params)
    problem_params.update(
        {
            "state_min_bounds": jnp.array([-0.1]),
            "state_max_bounds": jnp.array([-0.05]),
            "use_slack_variables": True,
            "slack_penalization_weight": 10.0,
        }
    )
    solver_params = copy.deepcopy(base_solver.params)
    solver_params["num_sqp_iteration_max"] = 1
    solver = TurboMPCSolver(
        program=OptimalControlProblem(_QuadraticImplicitDynamics(), problem_params),
        params=solver_params,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE,
        backward_hessian=HessianApproximation.GAUSS_NEWTON,
    )
    filtered = solver.solve(solver.initial_guess(problem_params), problem_params)

    np.testing.assert_allclose(float(filtered.linesearch_alphas[0]), 0.1)
    assert not np.allclose(filtered.slack, filtered.admm_state.xi_g)
    assert not hasattr(filtered.kkt_state, "slack")


def test_turbompc_solver_verbose_false_is_silent(capsys):
    dynamics, problem_params = _linear_problem()
    problem = OptimalControlProblem(
        dynamics=dynamics, params=copy.deepcopy(problem_params)
    )
    solver_params = turbompc_solver_params(tol=1e-6, sqp_iters=1, admm_max=3)
    solver_params["verbose"] = False

    capsys.readouterr()
    solver = _make_jax_solver(problem, solver_params)
    sol = solver.solve(solver.initial_guess(problem_params), problem_params)
    sol.states.block_until_ready()

    captured = capsys.readouterr()
    assert captured.out == ""


def test_turbompc_solver_verbose_prints_sqp_lines_without_alpha_when_no_linesearch(
    capsys,
):
    dynamics, problem_params = _linear_problem()
    problem = OptimalControlProblem(
        dynamics=dynamics, params=copy.deepcopy(problem_params)
    )
    solver_params = turbompc_solver_params(tol=1e-6, sqp_iters=1, admm_max=3)
    solver_params["verbose"] = True
    solver_params["linesearch"]["method"] = "none"

    capsys.readouterr()
    solver = _make_jax_solver(problem, solver_params)
    sol = solver.solve(solver.initial_guess(problem_params), problem_params)
    sol.states.block_until_ready()

    out = capsys.readouterr().out
    assert "[turbompc] init" in out
    assert "[turbompc] sqp it=00" in out
    assert "[turbompc] done status=1" in out
    assert "linesearch=none" in out
    assert "alpha=" not in out
    assert "[turbompc.admm]" not in out


def test_turbompc_solver_rejects_invalid_verbose():
    dynamics, problem_params = _linear_problem()
    problem = OptimalControlProblem(
        dynamics=dynamics, params=copy.deepcopy(problem_params)
    )
    solver_params = turbompc_solver_params(tol=1e-6, sqp_iters=1, admm_max=3)
    solver_params["verbose"] = 2

    with pytest.raises(ValueError, match="verbose"):
        _make_jax_solver(problem, solver_params)
