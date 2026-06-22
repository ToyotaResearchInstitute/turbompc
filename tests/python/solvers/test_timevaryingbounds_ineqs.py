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


def test_time_varying_bounds_should_be_enforced():
    import jax.numpy as jnp

    horizon = 3
    dynamics, params = make_linear_params(
        horizon=horizon,
        implicit=False,
        bounded=False,
        initial_state=jnp.ones((4,), dtype=jnp.float32),
    )
    umin = jnp.array(
        [[-0.1, -0.1]] + [[-jnp.inf, -jnp.inf]] * horizon, dtype=jnp.float32
    )
    umax = jnp.array([[0.1, 0.1]] + [[jnp.inf, jnp.inf]] * horizon, dtype=jnp.float32)
    params["control_min_bounds"] = umin
    params["control_max_bounds"] = umax

    problem = OptimalControlProblem(dynamics=dynamics, params=params)
    assert problem.num_inequality_constraints >= 2


def test_turbompc_solver_without_inequalities_should_not_crash():
    import jax
    import jax.numpy as jnp

    dynamics, problem_params = make_linear_params(
        horizon=3,
        implicit=False,
        bounded=False,
        initial_state=jnp.ones((4,), dtype=jnp.float32),
    )
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    solver_params = turbompc_solver_params(tol=1e-6, sqp_iters=4, admm_max=30)
    solver_params["linesearch"] = False
    solver_params["verbose"] = False
    solver_params["admm"]["check_termination_every"] = 10
    solver_params["admm"]["adapt_rho_every"] = 0
    solver = TurboMPCSolver(
        program=problem,
        params=solver_params,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
    )
    init = solver.initial_guess(problem_params)

    solve_jit = jax.jit(lambda: solver.solve(init, problem_params, {}))
    sol = solve_jit()
    assert jnp.all(jnp.isfinite(sol.controls))
