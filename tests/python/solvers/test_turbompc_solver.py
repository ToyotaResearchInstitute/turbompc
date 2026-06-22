import copy

import jax.numpy as jnp
from jax import config
from tests.helpers.problem_fixtures import make_linear_params
from tests.helpers.solver_fixtures import turbompc_solver_params
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)

config.update("jax_enable_x64", True)


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


def test_turbompc_solver_rti_first_order_does_not_stage_post_step_relinearization():
    dynamics, problem_params = _linear_problem()
    problem = OptimalControlProblem(
        dynamics=dynamics, params=copy.deepcopy(problem_params)
    )
    solver_params = turbompc_solver_params(tol=1e-6, sqp_iters=1, admm_max=30)
    solver_params["convergence_criterion"] = "first_order"
    solver = _make_jax_solver(problem, solver_params)

    build_count = {"count": 0}
    original_build = solver._build_qp_data

    def counted_build(*args, **kwargs):
        build_count["count"] += 1
        return original_build(*args, **kwargs)

    solver._build_qp_data = counted_build
    sol = solver.solve(solver.initial_guess(problem_params), problem_params)

    assert sol.num_iter == 1
    assert jnp.isfinite(sol.convergence_error)
    assert build_count["count"] == 1


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
