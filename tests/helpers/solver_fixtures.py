import jax.numpy as jnp

from turbompc.solvers.qp_data import (
    QPCostBlocks,
    QPData,
    QPEqualityBlocks,
    QPInequalityBlocks,
)
from turbompc.utils.load_params import load_solver_params


def make_tiny_qp(*, q, initial_rhs=0.0, hessian=None):
    """Build a two-stage scalar QP for solver-status tests."""
    dtype = q.dtype
    if hessian is None:
        hessian = jnp.zeros((2, 1, 1), dtype=dtype)
    return QPData(
        cost=QPCostBlocks(
            D=hessian,
            E=jnp.zeros((1, 1, 1), dtype=dtype),
            q=q,
        ),
        eq=QPEqualityBlocks(
            A0=jnp.zeros((1, 1), dtype=dtype),
            A_minus=jnp.zeros((1, 1, 1), dtype=dtype),
            A_plus=jnp.zeros((1, 1, 1), dtype=dtype),
            c0=jnp.asarray([initial_rhs], dtype=dtype),
            c=jnp.zeros((1, 1), dtype=dtype),
        ),
        ineq=QPInequalityBlocks(
            G=jnp.zeros((2, 1, 1), dtype=dtype),
            l=jnp.full((2, 1), -jnp.inf, dtype=dtype),
            u=jnp.full((2, 1), jnp.inf, dtype=dtype),
            slack_penalization_weight=jnp.asarray(0.0, dtype=dtype),
        ),
    )


def build_gauss_newton_qp(solver, states, controls, problem_params):
    """Build an ordinary forward QP through TurboMPC's shared builder."""
    from turbompc.solvers.turbompc_solver import HessianApproximation

    slacks = jnp.zeros(
        (states.shape[0], solver.program.num_inequality_constraints),
        dtype=states.dtype,
    )
    return solver._build_qp(
        states,
        controls,
        slacks,
        problem_params,
        None,
        hessian=HessianApproximation.GAUSS_NEWTON,
        scale_variables=True,
    )


def sqp_params(
    *,
    tol: float = 1e-6,
    sqp_iters: int = 20,
    linesearch: bool = False,
    warm_start_backward: bool = False,
    pcg_eps: float = 1e-12,
    linesearch_alphas=None,
):
    sp = load_solver_params("sqp.yaml")
    sp["tol_convergence"] = tol
    sp["num_sqp_iteration_max"] = sqp_iters
    sp["pcg"]["tol_epsilon"] = pcg_eps
    sp["linesearch"] = linesearch
    sp["warm_start_backward"] = warm_start_backward
    if linesearch_alphas is not None:
        sp["linesearch_alphas"] = list(linesearch_alphas)
    return sp


def sqp_osqp_params(
    *,
    tol: float = 1e-6,
    sqp_iters: int = 20,
    linesearch: bool = False,
    osqp_eps: float = 1e-8,
    linesearch_alphas=None,
):
    sp = load_solver_params("sqp_osqp.yaml")
    sp["tol_convergence"] = tol
    sp["num_sqp_iteration_max"] = sqp_iters
    sp["linesearch"] = linesearch
    sp["osqp"]["eps_abs"] = osqp_eps
    sp["osqp"]["eps_rel"] = osqp_eps
    if linesearch_alphas is not None:
        sp["linesearch_alphas"] = list(linesearch_alphas)
    return sp


def turbompc_solver_params(
    *, tol: float = 1e-10, sqp_iters: int = 30, admm_max: int = 500
):
    sp = load_solver_params("turbompc.yaml")
    sp["tol_convergence"] = tol
    sp["num_sqp_iteration_max"] = sqp_iters
    sp["admm"]["eps_abs"] = tol
    sp["admm"]["eps_rel"] = tol
    sp["admm"]["max_iter"] = admm_max
    return sp


def load_drone_solver_params():
    solver_params = load_solver_params("sqp_osqp.yaml")
    solver_params["num_sqp_iteration_max"] = 8
    solver_params["tol_convergence"] = 1e-4
    solver_params["linesearch"] = False
    solver_params["linesearch_alphas"] = [0.1, 0.3, 0.7, 1.0]

    solver_params["osqp"]["max_iter"] = 10000
    solver_params["osqp"]["eps_abs"] = 1.0e-7
    solver_params["osqp"]["eps_rel"] = 1.0e-7
    solver_params["osqp"]["check_termination_every"] = 25
    solver_params["osqp"]["verbose"] = False

    solver2_params = load_solver_params("turbompc.yaml")
    solver2_params["num_sqp_iteration_max"] = solver_params["num_sqp_iteration_max"]
    solver2_params["tol_convergence"] = solver_params["tol_convergence"]
    solver2_params["linesearch"]["method"] = (
        "merit" if solver_params["linesearch"] else "none"
    )
    solver2_params["linesearch"]["alphas"] = solver_params["linesearch_alphas"]

    solver2_params["admm"]["max_iter"] = solver_params["osqp"]["max_iter"]
    solver2_params["admm"]["eps_abs"] = solver_params["osqp"]["eps_abs"]
    solver2_params["admm"]["eps_rel"] = solver_params["osqp"]["eps_rel"]
    solver2_params["admm"]["check_termination_every"] = solver_params["osqp"][
        "check_termination_every"
    ]
    solver2_params["admm"]["pcg"]["tol_epsilon"] = 1e-15

    return solver_params, solver2_params
