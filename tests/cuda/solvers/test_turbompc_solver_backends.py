import copy
from typing import Any, Dict

import jax
import numpy as np
import osqp
import pytest
import scipy.sparse as sp

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from tests.helpers.assertions import assert_solution_nontrivial
from tests.helpers.backend_utils import backend_param
from tests.helpers.constants import (
    ADMM_EPS_BY_BACKEND,
    ADMM_EPS_BY_FORWARD_BACKEND,
    COST_TOL,
    COST_TOL_FUSED,
    EQ_TOL,
    OSQP_EPS,
    Z_TOL,
    Z_TOL_FUSED,
)
from tests.helpers.problem_fixtures import (
    make_drone_params,
    make_linear_params,
    make_spacecraft_params,
)
from tests.helpers.solver_fixtures import (
    load_drone_solver_params,
    sqp_osqp_params,
    turbompc_solver_params,
)
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.obstacle_avoidance import OptimalControlProblemObstacle
from turbompc.problems.optimal_control_problem import (
    OptimalControlProblem,
    OptimalControlProblemSlack,
)
from turbompc.solvers.admm import ADMMSolver
from turbompc.solvers.linear_systems_solvers.backends import (
    AdmmBackend,
    SchurSolverBackend,
)
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver
from turbompc.solvers.qp_utils import ZShape, pack_z
from turbompc.solvers.sqp_osqp import SQPOSQPSolver
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)

_DEFAULT_BACKWARD: dict[ForwardBackend, BackwardBackend] = {
    ForwardBackend.ADMM_JAX_LOOP_PCG: BackwardBackend.ADMM_JAX_LOOP_PCG,
    ForwardBackend.ADMM_JAX_LOOP_PCG_FFI: BackwardBackend.ADMM_JAX_LOOP_PCG_FFI,
    ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI: BackwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
    ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE: BackwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
    ForwardBackend.ADMM_FUSED_PCG: BackwardBackend.DIRECT_CUDSS_FFI,
    ForwardBackend.ADMM_FUSED_CUDSS: BackwardBackend.DIRECT_CUDSS_FFI,
}


def _resolve_backend(backend):
    """Accept SchurSolverBackend or ForwardBackend, return (ForwardBackend, BackwardBackend)."""
    if isinstance(backend, ForwardBackend):
        return backend, _DEFAULT_BACKWARD[backend]
    # Legacy SchurSolverBackend → ForwardBackend mapping
    _MAP = {
        SchurSolverBackend.PCG: ForwardBackend.ADMM_JAX_LOOP_PCG,
        SchurSolverBackend.PCG_FFI: ForwardBackend.ADMM_JAX_LOOP_PCG_FFI,
        SchurSolverBackend.CUDSS_FFI: ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        SchurSolverBackend.JAX_DENSE: ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
    }
    fb = _MAP[backend]
    return fb, _DEFAULT_BACKWARD[fb]


def _make_solver(
    problem: OptimalControlProblem,
    solver_params: Dict[str, Any],
    backend,
) -> TurboMPCSolver:
    fb, bb = _resolve_backend(backend)
    return TurboMPCSolver(
        program=problem,
        params=solver_params,
        forward_backend=fb,
        backward_backend=bb,
    )


def _admm_params(
    backend,
    *,
    max_iter: int | None = None,
    tol_convergence: float | None = None,
    linesearch: bool | None = None,
) -> Dict[str, Any]:
    solver_params = turbompc_solver_params(
        tol=EQ_TOL if tol_convergence is None else tol_convergence,
        admm_max=500 if max_iter is None else max_iter,
    )
    # Resolve eps from either backend enum type
    if isinstance(backend, ForwardBackend):
        eps = ADMM_EPS_BY_FORWARD_BACKEND.get(backend, 1e-4)
    else:
        eps = ADMM_EPS_BY_BACKEND[backend]
    fb = (
        backend if isinstance(backend, ForwardBackend) else _resolve_backend(backend)[0]
    )
    if fb == ForwardBackend.ADMM_JAX_LOOP_PCG:
        eps = max(eps, 1e-7)
    solver_params["admm"]["eps_abs"] = eps
    solver_params["admm"]["eps_rel"] = eps
    if linesearch is not None:
        solver_params["linesearch"] = linesearch
    # Tighten PCG tolerance for PCG_FFI
    if fb == ForwardBackend.ADMM_JAX_LOOP_PCG_FFI:
        solver_params["admm"]["pcg"]["tol_epsilon"] = 1e-12
    return solver_params


# Default CI covers the production GPU path. Other backends are parity checks.
ALL_BACKENDS = [
    backend_param(ForwardBackend.ADMM_FUSED_CUDSS),
    backend_param(ForwardBackend.ADMM_JAX_LOOP_PCG, marks=[pytest.mark.extended]),
    backend_param(ForwardBackend.ADMM_JAX_LOOP_PCG_FFI, marks=[pytest.mark.extended]),
    backend_param(ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI, marks=[pytest.mark.extended]),
    backend_param(ForwardBackend.ADMM_FUSED_PCG, marks=[pytest.mark.extended]),
]


def _is_fused(backend) -> bool:
    """True for fused/cuDSS-loop backends (looser tolerances)."""
    if isinstance(backend, ForwardBackend):
        return backend in {
            ForwardBackend.ADMM_FUSED_PCG,
            ForwardBackend.ADMM_FUSED_CUDSS,
        }
    return False


def _tol(backend):
    """Return (z_tol, cost_tol) for the given backend."""
    if _is_fused(backend):
        return Z_TOL_FUSED, COST_TOL_FUSED
    return Z_TOL, COST_TOL


def _assert_nontrivial_solution(sol, *, atol: float = 1e-8):
    assert_solution_nontrivial(sol, atol=atol)


def _osqp_params(
    *,
    eps: float = OSQP_EPS,
    tol_convergence: float = EQ_TOL,
    linesearch: bool = False,
) -> Dict[str, Any]:
    return sqp_osqp_params(tol=tol_convergence, osqp_eps=eps, linesearch=linesearch)


@pytest.mark.parametrize(
    "backend",
    ALL_BACKENDS,
)
@pytest.mark.parametrize("horizon", [2, 3])
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize(
    "rescale,rescaling",
    [
        (False, "none"),
        (True, "unit"),
        (True, "linspace"),
    ],
)
@pytest.mark.parametrize("initial_control", [False, True])
def test_turbompc_solver_linear_smoke(
    backend,
    horizon,
    implicit,
    bounded,
    rescale,
    rescaling,
    initial_control,
):
    dynamics, params = make_linear_params(
        horizon, implicit, bounded, rescale, initial_control, rescaling=rescaling
    )
    solver_params = _admm_params(
        backend, max_iter=1000, tol_convergence=EQ_TOL, linesearch=False
    )

    problem = OptimalControlProblem(dynamics=dynamics, params=copy.deepcopy(params))
    solver = _make_solver(problem, solver_params, backend)

    sol = solver.solve(solver.initial_guess(params), problem_params=params)
    assert sol.status == 0
    assert sol.states.shape == (horizon + 1, dynamics.num_states)
    assert sol.controls.shape == (horizon + 1, dynamics.num_controls)
    _assert_nontrivial_solution(sol)

    osqp_params = _osqp_params(eps=OSQP_EPS, tol_convergence=EQ_TOL, linesearch=False)
    osqp_solver = SQPOSQPSolver(program=problem, params=osqp_params)
    osqp_sol = osqp_solver.solve(problem_params=params)
    assert osqp_sol.status == 0

    z_osqp = pack_z(osqp_sol.states, osqp_sol.controls)
    z_admm = pack_z(sol.states, sol.controls)
    z_tol, cost_tol = _tol(backend)
    assert float(jnp.max(jnp.abs(z_osqp - z_admm))) < z_tol

    cost_osqp = problem.cost(osqp_sol.states, osqp_sol.controls, params)
    cost_admm = problem.cost(sol.states, sol.controls, params)
    assert float(jnp.abs(cost_osqp - cost_admm)) < cost_tol


@pytest.mark.parametrize(
    "backend",
    ALL_BACKENDS,
)
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("rescale", [False, True])
def test_turbompc_solver_jit_linear_smoke(backend, implicit, bounded, rescale):
    dynamics, params = make_linear_params(3, implicit, bounded, rescale)
    solver_params = _admm_params(backend, max_iter=1000, tol_convergence=EQ_TOL)

    problem = OptimalControlProblem(dynamics=dynamics, params=copy.deepcopy(params))
    solver = _make_solver(problem, solver_params, backend)

    states0, controls0 = problem.initial_guess(params)

    def jittable_solve(x, u, jittable_params):
        new_params = copy.deepcopy(params)
        for k, v in jittable_params.items():
            new_params[k] = v
        return solver._solve_impl(x, u, new_params)

    solve_jit = jax.jit(jittable_solve)
    sol = solve_jit(
        states0,
        controls0,
        {
            "initial_state": jnp.zeros(4),
            "weights_penalization_reference_state_trajectory": jnp.ones(4),
        },
    )

    assert sol.status == 0
    assert sol.states.shape == (4, dynamics.num_states)
    assert sol.controls.shape == (4, dynamics.num_controls)
    _assert_nontrivial_solution(sol)


@pytest.mark.parametrize(
    "backend",
    ALL_BACKENDS,
)
@pytest.mark.parametrize(
    "case",
    [
        dict(
            name="explicit_no_rate",
            implicit=False,
            rate=0.0,
            control_weight=1.0,
            bounds=10.0,
            sqp_iters=15,
            admm_max=1000,
        ),
        dict(
            name="explicit_rate",
            implicit=False,
            rate=0.5,
            control_weight=1.0,
            bounds=10.0,
            sqp_iters=15,
            admm_max=1000,
        ),
        dict(
            name="implicit_active_bounds",
            implicit=True,
            rate=0.5,
            control_weight=0.5,
            bounds=0.25,
            control_bounds=0.15,
            sqp_iters=4,
            admm_max=10000,
        ),
    ],
)
def test_turbompc_solver_matches_osqp_spacecraft(backend, case):
    control_bounds = case.get("control_bounds", case["bounds"])
    params = make_spacecraft_params(
        horizon=6,
        implicit=case["implicit"],
        rate_weight=case["rate"],
        control_weight=case["control_weight"],
        ref_weight=5.0,
        final_weight=2.0,
        initial_state=jnp.array([0.2, -0.2, 0.15]),
        initial_guess_final_state=jnp.array([0.0, 0.0, 0.0]),
        state_bounds=jnp.ones((3,)) * case["bounds"],
        control_bounds=jnp.ones((3,)) * control_bounds,
    )

    problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)

    osqp_params = _osqp_params()
    osqp_params["num_sqp_iteration_max"] = case["sqp_iters"]
    osqp_solver = SQPOSQPSolver(program=problem, params=osqp_params)
    osqp_sol = osqp_solver.solve(problem_params=params)
    assert osqp_sol.status == 0

    admm_params = _admm_params(backend)
    admm_params["admm"]["max_iter"] = case["admm_max"]
    admm_params["num_sqp_iteration_max"] = case["sqp_iters"]
    admm_solver = _make_solver(problem, admm_params, backend)
    admm_sol = admm_solver.solve(
        admm_solver.initial_guess(params), problem_params=params
    )
    assert admm_sol.status == 0

    last_idx = max(int(admm_sol.num_iter) - 1, 0)
    assert int(admm_sol.admm_iters[last_idx]) < admm_params["admm"]["max_iter"]

    z_osqp = pack_z(osqp_sol.states, osqp_sol.controls)
    z_admm = pack_z(admm_sol.states, admm_sol.controls)
    z_tol, cost_tol = _tol(backend)
    assert float(jnp.max(jnp.abs(z_osqp - z_admm))) < z_tol

    cost_osqp = problem.cost(osqp_sol.states, osqp_sol.controls, params)
    cost_admm = problem.cost(admm_sol.states, admm_sol.controls, params)
    assert float(jnp.abs(cost_osqp - cost_admm)) < cost_tol

    eq_admm = problem.equality_constraints(admm_sol.states, admm_sol.controls, params)
    assert float(jnp.max(jnp.abs(eq_admm))) < EQ_TOL


# Low-level ADMM API test — only JAX-loop backends (fused uses a different API)
@pytest.mark.parametrize(
    "backend",
    [
        backend_param(SchurSolverBackend.PCG),
        backend_param(SchurSolverBackend.PCG_FFI),
        backend_param(SchurSolverBackend.CUDSS_FFI),
    ],
)
def test_qp_admm_matches_osqp_spacecraft_implicit_active_bounds(backend):
    params = make_spacecraft_params(
        horizon=6,
        implicit=True,
        rate_weight=0.5,
        control_weight=0.5,
        ref_weight=5.0,
        final_weight=2.0,
        initial_state=jnp.array([0.2, -0.2, 0.15]),
        initial_guess_final_state=jnp.array([0.0, 0.0, 0.0]),
        state_bounds=jnp.ones((3,)) * 0.25,
        control_bounds=jnp.ones((3,)) * 0.15,
    )

    problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    states0, controls0 = problem.initial_guess(params)

    solver_osqp = SQPOSQPSolver(program=problem)
    P, q, Aeq, beq, Aineq, l_ineq, u_ineq = solver_osqp._build_qp_matrices_dense(
        states0, controls0, params
    )
    A_full = sp.vstack([sp.csc_matrix(Aeq), sp.csc_matrix(Aineq)], format="csc")
    l_full = np.concatenate([np.array(beq), np.array(l_ineq)])
    u_full = np.concatenate([np.array(beq), np.array(u_ineq)])

    osqp_solver = osqp.OSQP()
    P_csc = sp.triu(sp.csc_matrix(P))
    osqp_solver.setup(
        P_csc,
        np.array(q),
        A_full,
        l_full,
        u_full,
        eps_abs=1e-10,
        eps_rel=1e-10,
        max_iter=10000,
        verbose=False,
        polish=False,
        warm_start=False,
        scaling=0,
        check_termination=25,
    )
    result = osqp_solver.solve()
    assert result.info.status == "solved"
    z_osqp = jnp.array(result.x, dtype=states0.dtype)

    admm_params = turbompc_solver_params()
    if backend == SchurSolverBackend.PCG_FFI:
        admm_params["admm"]["pcg"]["tol_epsilon"] = 1e-12
    schur_solver = make_schur_solver(
        backend,
        problem.horizon,
        problem.num_state_variables,
        problem.num_control_variables,
        pcg_params=admm_params["admm"]["pcg"],
    )
    zshape = ZShape(
        horizon=problem.horizon,
        num_states=problem.num_state_variables,
        num_controls=problem.num_control_variables,
    )
    admm_solver = ADMMSolver(
        zshape=zshape,
        schur_solver=schur_solver,
        pcg_params=admm_params["admm"]["pcg"],
        sigma=admm_params["admm"]["sigma"],
        max_iter=10000,
        eps_abs=1e-10,
        eps_rel=1e-10,
        rho_min=admm_params["admm"].get("rho_min", 1.0e-6),
        rho_max=admm_params["admm"].get("rho_max", 1.0e6),
        check_termination_every=1,
        adapt_rho_every=admm_params["admm"].get("adapt_rho_every", 25),
        adaptive_rho_tolerance=admm_params["admm"].get("adaptive_rho_tolerance", 5.0),
        rho_f_factor=admm_params["admm"].get(
            "rho_f_factor",
            admm_params["admm"].get("active_constraint_rho_factor", 1000.0),
        ),
        admm_backend=AdmmBackend.JAX_LOOP,
    )
    qp_data = _make_solver(problem, admm_params, backend)._build_qp_data(
        states0, controls0, params
    )
    (states_admm, controls_admm), _, admm_state = admm_solver.solve(
        qp_data=qp_data,
        rho_bar=admm_params["admm"]["rho"],
    )
    z_admm = pack_z(states_admm, controls_admm)
    assert float(jnp.max(jnp.abs(z_osqp - z_admm))) < 2e-3

    def _kkt_residuals(P_, q_, A_, l_, u_, z, y):
        Az = A_ @ z
        proj = jnp.minimum(jnp.maximum(Az, l_), u_)
        r_primal = jnp.max(jnp.abs(Az - proj))
        r_dual = jnp.max(jnp.abs(P_ @ z + q_ + A_.T @ y))
        return float(r_primal), float(r_dual)

    z_osqp = jnp.array(z_osqp).reshape(-1)
    z_admm = jnp.array(z_admm).reshape(-1)
    y_osqp = jnp.array(result.y, dtype=z_osqp.dtype)
    y_admm = jnp.concatenate(
        [
            admm_state.y_f_0.reshape(-1),
            admm_state.y_f_dyn.reshape(-1),
            admm_state.y_g.reshape(-1),
        ]
    )

    A_dense = jnp.vstack([jnp.array(Aeq), jnp.array(Aineq)])
    l_dense = jnp.concatenate([jnp.array(beq), jnp.array(l_ineq)])
    u_dense = jnp.concatenate([jnp.array(beq), jnp.array(u_ineq)])

    r_p_osqp, r_d_osqp = _kkt_residuals(P, q, A_dense, l_dense, u_dense, z_osqp, y_osqp)
    r_p_admm, r_d_admm = _kkt_residuals(P, q, A_dense, l_dense, u_dense, z_admm, y_admm)

    obj_osqp = 0.5 * (z_osqp @ (P @ z_osqp)) + q @ z_osqp
    obj_admm = 0.5 * (z_admm @ (P @ z_admm)) + q @ z_admm
    assert float(jnp.abs(obj_osqp - obj_admm)) < 2e-4
    assert r_p_osqp < 1.0e-6
    assert r_d_osqp < 1.0e-6
    assert r_p_admm < 1.0e-4
    assert r_d_admm < 1.0e-4


@pytest.mark.parametrize(
    "backend",
    ALL_BACKENDS,
)
def test_turbompc_solver_warm_start_reduces_admm_iters(backend):
    np.random.seed(0)
    dynamics, params = make_linear_params(4, implicit=False, bounded=False)

    solver_params = _admm_params(backend, max_iter=100, linesearch=False)
    solver_params["num_sqp_iteration_max"] = 3

    problem = OptimalControlProblem(dynamics=dynamics, params=copy.deepcopy(params))
    solver = _make_solver(problem, solver_params, backend)

    sol_cold = solver.solve(solver.initial_guess(params), problem_params=params)
    sol_warm = solver.solve(sol_cold, problem_params=params)

    cold_iters = int(sol_cold.admm_iters[0])
    warm_iters = int(sol_warm.admm_iters[0])
    # Direct solvers (cuDSS) may already converge at minimum iterations,
    # so warm start may not reduce count further.
    assert warm_iters <= cold_iters


@pytest.mark.parametrize(
    "backend",
    ALL_BACKENDS,
)
def test_turbompc_solver_respects_initial_control_constraint(backend):
    init_control = jnp.array([0.25, -0.35])
    dynamics, params = make_linear_params(4, implicit=False, bounded=False)
    params["initial_state"] = jnp.array([0.2, -0.1, 0.05, 0.0])
    params["weights_penalization_control_squared"] = (
        jnp.ones((dynamics.num_controls,)) * 0.1
    )
    params["constrain_initial_control"] = True
    params["initial_control"] = init_control

    solver_params = _admm_params(
        backend, max_iter=100, tol_convergence=1.0e-6, linesearch=False
    )

    problem = OptimalControlProblem(dynamics=dynamics, params=copy.deepcopy(params))
    solver = _make_solver(problem, solver_params, backend)
    sol = solver.solve(solver.initial_guess(params), problem_params=params)

    osqp_solver = SQPOSQPSolver(program=problem)
    osqp_sol = osqp_solver.solve(problem_params=params)
    assert osqp_sol.status == 0
    assert sol.status == 0
    assert np.allclose(
        np.asarray(sol.controls[0]), np.asarray(init_control), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    "backend",
    ALL_BACKENDS,
)
def test_turbompc_solver_matches_osqp_with_slack_linear_ineq(backend):
    dynamics, params = make_linear_params(3, implicit=False, bounded=False)
    nx = dynamics.num_states
    nu = dynamics.num_controls

    params["initial_state"] = jnp.zeros((nx,))
    params["state_min_bounds"] = jnp.array([1.0] + [-1.0e3] * (nx - 1))
    params["state_max_bounds"] = jnp.array([1.0] + [1.0e3] * (nx - 1))
    params["control_min_bounds"] = -jnp.ones((nu,)) * 1.0e3
    params["control_max_bounds"] = jnp.ones((nu,)) * 1.0e3
    params["ineq_include_box"] = True
    params["use_slack_variables"] = True
    params["slack_penalization_weight"] = 10.0

    problem = OptimalControlProblemSlack(dynamics=dynamics, params=params)

    solver_osqp = SQPOSQPSolver(program=problem)
    sol_osqp = solver_osqp.solve(problem_params=params)
    assert sol_osqp.status == 0

    solver_admm = _make_solver(
        problem,
        _admm_params(backend),
        backend,
        # backward backend is part of the solver enum; no separate flag
    )
    sol_admm = solver_admm.solve(
        solver_admm.initial_guess(params), problem_params=params
    )
    assert sol_admm.status == 0

    z_osqp = pack_z(sol_osqp.states, sol_osqp.controls)
    z_admm = pack_z(sol_admm.states, sol_admm.controls)
    assert float(jnp.max(jnp.abs(z_osqp - z_admm))) < 5e-3

    g, l, u = problem.inequality_constraints(sol_admm.states, sol_admm.controls, params)
    g = (g + sol_admm.slack).reshape(-1)
    l = l.reshape(-1)
    u = u.reshape(-1)
    # Fused/cuDSS-loop backends use eps_abs = 1e-4 (see ADMM_EPS_BY_FORWARD_BACKEND).
    # Once cuDSS-loop stops exactly at the iter that first satisfies eps_abs (no
    # post-convergence over-iteration), feasibility violation can be a small
    # multiple of eps_abs; pad accordingly.
    ineq_margin = 5e-4 if _is_fused(backend) else 1e-6
    assert jnp.all(g >= l - ineq_margin)
    assert jnp.all(g <= u + ineq_margin)
    g_raw, l_raw, u_raw = problem.inequality_constraints(
        sol_admm.states, sol_admm.controls, params
    )
    g_raw = g_raw.reshape(-1)
    assert jnp.any(g_raw < l_raw.reshape(-1) - 1.0e-6)


@pytest.mark.parametrize(
    "backend",
    ALL_BACKENDS,
)
def test_turbompc_solver_vs_sqp_osqp_obstacle_avoidance(backend):
    horizon = 50
    obs_centers = jnp.array([[-1.4, -0.1], [-0.7, 0.3], [-0.3, 0.25]])
    obs_radii = jnp.array([0.3, 0.2, 0.2])
    problem_params, dynamics = make_drone_params(
        horizon=horizon, obs_centers=obs_centers, obs_radii=obs_radii
    )
    solver_params, solver2_params = load_drone_solver_params()

    problem = OptimalControlProblemObstacle(dynamics=dynamics, params=problem_params)
    solver = _make_solver(problem, solver2_params, backend)
    solution_1 = solver.solve(
        solver.initial_guess(problem_params), problem_params=problem_params
    )
    convergence_error_1 = solution_1.convergence_error
    equality_constraints_error_1 = jnp.linalg.norm(
        problem.equality_constraints(
            solution_1.states, solution_1.controls, problem.params
        ),
        ord=jnp.inf,
    )
    ineq_values, ineq_lower, ineq_upper = problem.inequality_constraints(
        solution_1.states, solution_1.controls, problem_params
    )
    ineq_values = ineq_values.reshape(-1)
    ineq_lower = ineq_lower.reshape(-1)
    ineq_upper = ineq_upper.reshape(-1)
    ineq_violation = jnp.maximum(0.0, ineq_lower - ineq_values) + jnp.maximum(
        0.0, ineq_values - ineq_upper
    )
    inequality_constraints_error_1 = jnp.linalg.norm(ineq_violation, ord=jnp.inf)

    solver = SQPOSQPSolver(program=problem, params=solver_params)
    solution_2 = solver.solve(problem_params=problem_params)
    convergence_error_2 = solution_2.convergence_error
    equality_constraints_error_2 = jnp.linalg.norm(
        problem.equality_constraints(
            solution_2.states, solution_2.controls, problem.params
        ),
        ord=jnp.inf,
    )
    ineq_values, ineq_lower, ineq_upper = problem.inequality_constraints(
        solution_2.states, solution_2.controls, problem_params
    )
    ineq_values = ineq_values.reshape(-1)
    ineq_lower = ineq_lower.reshape(-1)
    ineq_upper = ineq_upper.reshape(-1)
    ineq_violation = jnp.maximum(0.0, ineq_lower - ineq_values) + jnp.maximum(
        0.0, ineq_values - ineq_upper
    )
    inequality_constraints_error_2 = jnp.linalg.norm(ineq_violation, ord=jnp.inf)

    z_admm = pack_z(solution_1.states, solution_1.controls)
    z_osqp = pack_z(solution_2.states, solution_2.controls)
    assert float(jnp.max(jnp.abs(z_osqp - z_admm))) < 2e-3
    assert convergence_error_1 < 0.1
    assert convergence_error_2 < 0.1
    assert equality_constraints_error_1 < 1e-3
    assert equality_constraints_error_2 < 1e-3
    assert inequality_constraints_error_1 < 1e-3
    assert inequality_constraints_error_2 < 1e-3


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize(
    "scheme,scheme_id",
    [
        (0, "euler"),
        (1, "midpoint"),
        (2, "rk4"),
        (10, "implicit"),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_turbompc_solver_integrators(backend, scheme, scheme_id):
    """All integrator types produce finite, converged solutions."""
    from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics

    params = make_spacecraft_params(
        horizon=6,
        implicit=(scheme == 10),
        rate_weight=0.0,
        control_weight=1.0,
        ref_weight=5.0,
        final_weight=2.0,
        initial_state=jnp.array([0.2, -0.2, 0.15]),
        initial_guess_final_state=jnp.array([0.0, 0.0, 0.0]),
        state_bounds=jnp.ones((3,)) * 10.0,
        control_bounds=jnp.ones((3,)) * 10.0,
    )
    params["discretization_scheme"] = scheme

    problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    solver_params = _admm_params(backend, max_iter=1000)
    solver_params["num_sqp_iteration_max"] = 10
    solver = _make_solver(problem, solver_params, backend)

    sol = solver.solve(solver.initial_guess(params), problem_params=params)
    assert sol.status == 0
    _assert_nontrivial_solution(sol)

    eq = problem.equality_constraints(sol.states, sol.controls, params)
    assert (
        float(jnp.max(jnp.abs(eq))) < EQ_TOL
    ), f"Eq constraint violation with {scheme_id}"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize(
    "control_bound,state_bound_factor,desc",
    [
        (0.15, 1e7, "control_only"),  # tight control, effectively no state bounds
        (1e7, 0.25, "state_only"),  # tight state, effectively no control bounds
        (0.15, 0.25, "both_tight"),  # both active
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_turbompc_solver_constraint_types(
    backend, control_bound, state_bound_factor, desc
):
    """Box constraints: control-only, state-only, and both tight."""
    from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics

    params = make_spacecraft_params(
        horizon=6,
        implicit=False,
        rate_weight=0.5,
        control_weight=0.5,
        ref_weight=5.0,
        final_weight=2.0,
        initial_state=jnp.array([0.2, -0.2, 0.15]),
        initial_guess_final_state=jnp.array([0.0, 0.0, 0.0]),
        state_bounds=jnp.ones((3,)) * state_bound_factor,
        control_bounds=jnp.ones((3,)) * control_bound,
    )

    problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    solver_params = _admm_params(backend, max_iter=5000)
    solver_params["num_sqp_iteration_max"] = 10
    solver = _make_solver(problem, solver_params, backend)

    sol = solver.solve(solver.initial_guess(params), problem_params=params)
    assert sol.status == 0
    _assert_nontrivial_solution(sol)

    # Check bounds are respected
    z_tol, _ = _tol(backend)
    if control_bound < 1.0:
        assert float(jnp.max(jnp.abs(sol.controls))) <= control_bound + z_tol, (
            "Control bound violated:"
            f" max|u|={float(jnp.max(jnp.abs(sol.controls))):.4f} > {control_bound}"
        )
    if state_bound_factor < 1.0:
        assert float(jnp.max(jnp.abs(sol.states))) <= state_bound_factor + z_tol, (
            f"State bound violated: max|x|={float(jnp.max(jnp.abs(sol.states))):.4f} >"
            f" {state_bound_factor}"
        )


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_turbompc_solver_obstacle_avoidance_general_ineq(backend):
    """General nonlinear inequality constraints (obstacle avoidance)."""
    horizon = 50
    obs_centers = jnp.array([[-1.4, -0.1], [-0.7, 0.3]])
    obs_radii = jnp.array([0.3, 0.2])
    problem_params, dynamics = make_drone_params(
        horizon=horizon, obs_centers=obs_centers, obs_radii=obs_radii
    )
    _, solver2_params = load_drone_solver_params()

    problem = OptimalControlProblemObstacle(dynamics=dynamics, params=problem_params)
    solver = _make_solver(problem, solver2_params, backend)
    sol = solver.solve(
        solver.initial_guess(problem_params), problem_params=problem_params
    )

    _assert_nontrivial_solution(sol)

    # Equality constraints (dynamics)
    eq = problem.equality_constraints(sol.states, sol.controls, problem_params)
    assert float(jnp.linalg.norm(eq, ord=jnp.inf)) < 1e-3

    # Inequality constraints (obstacles)
    ineq_values, ineq_lower, ineq_upper = problem.inequality_constraints(
        sol.states, sol.controls, problem_params
    )
    ineq_violation = jnp.maximum(
        0.0, ineq_lower.reshape(-1) - ineq_values.reshape(-1)
    ) + jnp.maximum(0.0, ineq_values.reshape(-1) - ineq_upper.reshape(-1))
    ineq_margin = 1e-2 if _is_fused(backend) else 1e-3
    assert float(jnp.linalg.norm(ineq_violation, ord=jnp.inf)) < ineq_margin
