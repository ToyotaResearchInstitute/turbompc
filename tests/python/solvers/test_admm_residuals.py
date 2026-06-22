from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import osqp
import pytest
import scipy.sparse as sp
from tests.helpers.problem_fixtures import (
    cost_blocks_from_qr,
    make_spacecraft_params,
    tile_spacecraft_inertia,
)
from tests.helpers.solver_fixtures import turbompc_solver_params
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.admm.admm import (
    ADMMSolver,
    ADMMState,
    _compute_dynamics_dual_terms,
    _compute_residuals,
    _dynamics_residual,
    compute_gamma,
    compute_S_Phiinv,
)
from turbompc.solvers.linear_systems_solvers.backends import (
    AdmmBackend,
    SchurSolverBackend,
)
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver
from turbompc.solvers.qp_data import (
    QPCostBlocks,
    QPData,
    QPEqualityBlocks,
    qpdata_from_ocp_blocks,
)
from turbompc.solvers.qp_utils import ZShape, pack_x, pack_z, unpack_z
from turbompc.solvers.sqp_osqp import SQPOSQPSolver
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)
from turbompc.utils.load_params import load_problem_params


def _build_dense_C(eq, nx: int, nu: int) -> jnp.ndarray:
    N = eq.A_minus.shape[0]
    n = nx + nu
    C = jnp.zeros(((N + 1) * nx, (N + 1) * n), dtype=eq.A0.dtype)
    C = C.at[:nx, :n].set(eq.A0)
    for k in range(N):
        row = (k + 1) * nx
        col_xk = k * n
        col_xkp1 = (k + 1) * n
        C = C.at[row : row + nx, col_xk : col_xk + n].set(eq.A_minus[k])
        C = C.at[row : row + nx, col_xkp1 : col_xkp1 + n].set(eq.A_plus[k])
    return C


def _make_jax_turbompc_solver(program, params=None):
    return TurboMPCSolver(
        program=program,
        params=params,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
    )


def _build_dense_A(C: jnp.ndarray, n: int) -> jnp.ndarray:
    m, p = C.shape
    A = jnp.zeros((m + p, p), dtype=C.dtype)
    A = A.at[:m, :].set(C)
    A = A.at[m:, :].set(jnp.eye(p))
    return A


def _make_eq_blocks(
    As_next: jnp.ndarray,
    Bs_next: jnp.ndarray,
    As: jnp.ndarray,
    Bs: jnp.ndarray,
    Cs: jnp.ndarray,
) -> QPEqualityBlocks:
    nx = As.shape[1]
    nu = Bs.shape[2]
    A0 = jnp.concatenate(
        [jnp.eye(nx, dtype=As.dtype), jnp.zeros((nx, nu), dtype=As.dtype)], axis=1
    )
    A_minus = jnp.concatenate([As, Bs], axis=2)
    A_plus = jnp.concatenate([As_next, Bs_next], axis=2)
    return QPEqualityBlocks(
        A0=A0,
        A_minus=A_minus,
        A_plus=A_plus,
        c0=Cs[0],
        c=Cs[1:],
    )


def _build_dense_P_from_cost_blocks(cost: QPCostBlocks) -> jnp.ndarray:
    N = cost.D.shape[0] - 1
    n = cost.D.shape[1]
    P = jnp.zeros(((N + 1) * n, (N + 1) * n), dtype=cost.D.dtype)
    for k in range(N + 1):
        P = P.at[k * n : (k + 1) * n, k * n : (k + 1) * n].set(cost.D[k])
    for k in range(N):
        P = P.at[(k + 1) * n : (k + 2) * n, k * n : (k + 1) * n].set(cost.E[k])
        P = P.at[k * n : (k + 1) * n, (k + 1) * n : (k + 2) * n].set(cost.E[k].T)
    return P


@pytest.mark.parametrize("rate_weight", [0.0, 0.5])
@pytest.mark.parametrize("rescale", [False, True])
def test_cost_blocks_gradient_matches_cost(rate_weight, rescale):
    params = make_spacecraft_params(
        horizon=4,
        implicit=False,
        rate_weight=rate_weight,
        control_weight=0.7,
        ref_weight=1.3,
        final_weight=0.4,
    )
    params["penalize_control_reference"] = True
    dtype = params["initial_state"].dtype
    params["reference_control_trajectory"] = (
        jnp.arange((params["horizon"] + 1) * 3, dtype=dtype).reshape(
            params["horizon"] + 1, 3
        )
        * 0.01
    )
    params["rescale_optimization_variables"] = rescale
    params["state_rescaling_max"] = jnp.array([0.1, 0.07, 0.2])
    params["state_rescaling_min"] = -params["state_rescaling_max"]
    params["control_rescaling_max"] = jnp.array([2.0, 50.0, 0.5])
    params["control_rescaling_min"] = -params["control_rescaling_max"]

    problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    rng = np.random.default_rng(13)
    states, controls = problem.initial_guess(params)
    states = states + jnp.array(rng.standard_normal(states.shape)) * 0.1
    controls = controls + jnp.array(rng.standard_normal(controls.shape)) * 0.1

    z_shape = ZShape(
        params["horizon"], problem.num_state_variables, problem.num_control_variables
    )
    z = pack_z(states, controls)

    def cost_from_z(z_flat):
        states_z, controls_z = unpack_z(z_flat, z_shape)
        return problem.cost(states_z, controls_z, params)

    D, E, q = problem.get_cost_linearized_blocks(states, controls, params)
    qp_gradient = _build_dense_P_from_cost_blocks(QPCostBlocks(D, E, q)) @ z
    qp_gradient = qp_gradient + q.reshape(-1)

    np.testing.assert_allclose(
        qp_gradient, jax.grad(cost_from_z)(z), rtol=1e-6, atol=1e-6
    )


def _assemble_full_S(S_blocks: jnp.ndarray) -> jnp.ndarray:
    """Assemble full banded S from per-time blocks."""
    T, nz, _ = S_blocks.shape
    S_full = jnp.zeros((T * nz, T * nz), dtype=S_blocks.dtype)
    for t in range(T):
        row = slice(t * nz, (t + 1) * nz)
        col_prev = slice((t - 1) * nz, t * nz)
        col = slice(t * nz, (t + 1) * nz)
        col_next = slice((t + 1) * nz, (t + 2) * nz)
        left = S_blocks[t, :, :nz]
        mid = S_blocks[t, :, nz : 2 * nz]
        right = S_blocks[t, :, 2 * nz :]
        if t > 0:
            S_full = S_full.at[row, col_prev].set(left)
        S_full = S_full.at[row, col].set(mid)
        if t < T - 1:
            S_full = S_full.at[row, col_next].set(right)
    return S_full


def _compute_kkt_residuals_dense(
    P: jnp.ndarray,
    q: jnp.ndarray,
    A: jnp.ndarray,
    l: jnp.ndarray,
    u: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> Tuple[float, float]:
    """Return (primal, dual) KKT residuals for l <= A x <= u."""
    Az = A @ x
    proj = jnp.minimum(jnp.maximum(Az, l), u)
    r_primal = jnp.max(jnp.abs(Az - proj))
    r_dual = jnp.max(jnp.abs(P @ x + q + A.T @ y))
    return float(r_primal), float(r_dual)


@pytest.mark.parametrize("N", [2, 3])
def test_dynamics_dual_terms_match_explicit_cty(N):
    rng = np.random.default_rng(0)
    nx = 2
    nu = 1

    As_next = jnp.array(rng.standard_normal((N, nx, nx)))
    As = jnp.array(rng.standard_normal((N, nx, nx)))
    Bs_next = jnp.array(rng.standard_normal((N, nx, nu)))
    Bs = jnp.array(rng.standard_normal((N, nx, nu)))
    y_f = jnp.array(rng.standard_normal((N + 1, nx)))
    y_f0 = y_f[0]
    y_f_dyn = y_f[1:]
    dual_states, dual_controls = _compute_dynamics_dual_terms(
        As_next, As, Bs_next, Bs, y_f_dyn
    )

    n = nx + nu
    eq = _make_eq_blocks(
        As_next=As_next,
        Bs_next=Bs_next,
        As=As,
        Bs=Bs,
        Cs=jnp.zeros((N + 1, nx)),
    )
    C = _build_dense_C(eq, nx, nu)
    # A = _build_dense_A(C, n)
    y_f_stack = jnp.concatenate([jnp.zeros_like(y_f0), y_f_dyn.reshape(-1)], axis=0)
    dual_terms = C.T @ y_f_stack
    dual_terms = dual_terms.reshape(N + 1, n)
    cty_states = dual_terms[:, :nx]
    cty_controls = dual_terms[:, nx:]

    np.testing.assert_allclose(dual_states, cty_states, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dual_controls, cty_controls, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("N", [2, 3])
def test_residuals_match_explicit_dense(N):
    rng = np.random.default_rng(1)
    nx = 2
    nu = 1
    n = nx + nu

    As_next = jnp.array(rng.standard_normal((N, nx, nx)))
    As = jnp.array(rng.standard_normal((N, nx, nx)))
    Bs_next = jnp.array(rng.standard_normal((N, nx, nu)))
    Bs = jnp.array(rng.standard_normal((N, nx, nu)))
    Cs = jnp.array(rng.standard_normal((N + 1, nx)))

    Qmat = jnp.array(rng.standard_normal((N + 1, nx, nx)))
    Qmat = (Qmat + jnp.swapaxes(Qmat, -1, -2)) * 0.5
    Rmat = jnp.array(rng.standard_normal((N + 1, nu, nu)))
    Rmat = (Rmat + jnp.swapaxes(Rmat, -1, -2)) * 0.5
    Rd = jnp.array(rng.standard_normal((N, nu, nu)))
    Rd = (Rd + jnp.swapaxes(Rd, -1, -2)) * 0.5
    qvec = jnp.array(rng.standard_normal((N + 1, nx)))
    rvec = jnp.array(rng.standard_normal((N + 1, nu)))

    m_ineq = 2
    ineq_blocks = jnp.array(rng.standard_normal((N + 1, m_ineq, nx + nu)))
    ineq_l = -jnp.ones((N + 1, m_ineq))
    ineq_u = jnp.ones((N + 1, m_ineq))
    D, E, q = cost_blocks_from_qr(Qmat, Rmat, Rd, qvec, rvec)
    A0 = jnp.concatenate([jnp.eye(nx), jnp.zeros((nx, nu))], axis=1)
    c0 = Cs[0]
    qp = qpdata_from_ocp_blocks(
        D=D,
        E=E,
        q=q,
        A0=A0,
        c0=c0,
        As_next=As_next,
        Bs_next=Bs_next,
        As=As,
        Bs=Bs,
        c_dyn=Cs[1:],
        ineq_blocks=ineq_blocks,
        ineq_l=ineq_l,
        ineq_u=ineq_u,
    )

    states = jnp.array(rng.standard_normal((N + 1, nx)))
    controls = jnp.array(rng.standard_normal((N + 1, nu)))
    x = pack_z(states, controls)

    # multipliers + slack
    y_f = jnp.array(rng.standard_normal((N + 1, nx)))
    y_f0 = y_f[0]
    y_f_dyn = y_f[1:]
    z_g = jnp.array(rng.standard_normal((N + 1, m_ineq)))
    y_g = jnp.array(rng.standard_normal((N + 1, m_ineq)))

    # 1) primal and dual residuals - dense computation
    C = _build_dense_C(qp.eq, nx, nu)
    A = _build_dense_A(C, n)

    c = jnp.concatenate([qp.eq.c0, qp.eq.c.reshape(-1)], axis=0)
    eq_dense = (C @ x) - c
    np.testing.assert_allclose(eq_dense, (A @ x)[: len(c)] - c, rtol=1e-6, atol=1e-6)
    eq = _dynamics_residual(qp.eq, pack_x(states, controls))
    np.testing.assert_allclose(eq_dense, eq, rtol=1e-6, atol=1e-6)

    G_dense = jnp.zeros(((N + 1) * m_ineq, (N + 1) * n), dtype=ineq_blocks.dtype)
    for t in range(N + 1):
        G_dense = G_dense.at[
            t * m_ineq : (t + 1) * m_ineq,
            t * n : (t + 1) * n,
        ].set(ineq_blocks[t])
    Gx_dense = (G_dense @ x).reshape(N + 1, m_ineq)
    # also match vmap version
    x_blocks = pack_x(states, controls)
    from jax import vmap

    Gx = vmap(lambda G, z: G @ z)(ineq_blocks, x_blocks)
    np.testing.assert_allclose(Gx_dense, Gx, rtol=1e-6, atol=1e-6)

    primal_residual = float(jnp.max(jnp.abs(eq)))
    primal_residual = max(primal_residual, float(jnp.max(jnp.abs(Gx - z_g))))

    P = _build_dense_P_from_cost_blocks(qp.cost)
    q = qp.cost.q.reshape(-1)
    y_f_stack = jnp.concatenate([y_f0, y_f_dyn.reshape(-1)], axis=0)
    dual_residual = P @ x + q + C.T @ y_f_stack + G_dense.T @ y_g.reshape(-1)
    dual_residual = np.max(np.abs(dual_residual))

    # primal and dual residuals - optimized computation
    residuals = _compute_residuals(
        qp,
        ADMMState(
            x_blocks=pack_x(states, controls),
            y_g=y_g,
            y_f_0=y_f0,
            y_f_dyn=y_f_dyn,
            z_g=z_g,
            xi_g=jnp.zeros_like(z_g),
            rho_bar=0.1,
        ),
    )

    np.testing.assert_allclose(
        float(residuals.primal_residual), primal_residual, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        float(residuals.dual_residual), dual_residual, rtol=1e-6, atol=1e-6
    )

    gamma = 2.0
    xi_g = jnp.ones_like(z_g) * 0.3
    dual_residual_slack = max(
        dual_residual, float(jnp.max(jnp.abs(gamma * xi_g + y_g)))
    )
    qp_slack = QPData(
        cost=qp.cost,
        eq=qp.eq,
        ineq=qp.ineq.__class__(
            G=qp.ineq.G,
            l=qp.ineq.l,
            u=qp.ineq.u,
            slack_penalization_weight=jnp.array(gamma, dtype=qp.ineq.l.dtype),
            use_slack_variables=True,
        ),
    )
    residuals = _compute_residuals(
        qp_slack,
        ADMMState(
            x_blocks=pack_x(states, controls),
            y_g=y_g,
            y_f_0=y_f0,
            y_f_dyn=y_f_dyn,
            z_g=z_g,
            xi_g=xi_g,
            rho_bar=0.1,
        ),
    )
    np.testing.assert_allclose(
        float(residuals.dual_residual), dual_residual_slack, rtol=1e-6, atol=1e-6
    )


def test_residuals_match_dense_spacecraft_qp():
    params = load_problem_params("spacecraft_constrained.yaml")
    params = dict(params)
    params["horizon"] = 3
    tile_spacecraft_inertia(params)
    params["reference_state_trajectory"] = jnp.zeros((params["horizon"] + 1, 3))
    params["reference_control_trajectory"] = jnp.zeros((params["horizon"] + 1, 3))
    params["weights_penalization_control_rate"] = jnp.ones((3,)) * 0.5

    problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    solver = _make_jax_turbompc_solver(problem)

    rng = np.random.default_rng(2)
    states, controls = problem.initial_guess(params)
    states = jnp.array(rng.standard_normal(states.shape)) * 0.1
    controls = jnp.array(rng.standard_normal(controls.shape)) * 0.1
    qp = solver._build_qp_data(states, controls, params)

    N = params["horizon"]
    nx = problem.num_state_variables
    nu = problem.num_control_variables
    n = nx + nu

    x = pack_z(states, controls).reshape(-1)

    # equality multipliers
    y_f = jnp.array(rng.standard_normal((N + 1, nx)))
    y_f0 = y_f[0]
    y_f_dyn = y_f[1:]

    # inequality constraints
    m_ineq = int(qp.ineq.G.shape[1]) if qp.ineq.G.shape[0] > 0 else 0
    if m_ineq > 0:
        z_g = jnp.array(rng.standard_normal((N + 1, m_ineq)))
        y_g = jnp.array(rng.standard_normal((N + 1, m_ineq)))
    else:
        z_g = jnp.zeros((N + 1, 0), dtype=states.dtype)
        y_g = jnp.zeros((N + 1, 0), dtype=states.dtype)
    if m_ineq > 0:
        x_blocks = pack_x(states, controls)  # (N+1, nx+nu)
        from jax import vmap

        ineq_values = vmap(lambda G, x: G @ x)(qp.ineq.G, x_blocks)  # (N+1, m)
    else:
        ineq_values = jnp.zeros((N + 1, 0), dtype=states.dtype)

    C = _build_dense_C(qp.eq, nx, nu)
    c = jnp.concatenate([qp.eq.c0, qp.eq.c.reshape(-1)], axis=0)
    eq_dense = (C @ x) - c
    eq = _dynamics_residual(qp.eq, pack_x(states, controls))
    np.testing.assert_allclose(eq_dense, eq, rtol=1e-6, atol=1e-6)
    A = _build_dense_A(C, n)
    np.testing.assert_allclose(eq_dense, (A @ x)[: len(c)] - c, rtol=1e-6, atol=1e-6)

    if m_ineq > 0:
        G_dense = jnp.zeros(((N + 1) * m_ineq, (N + 1) * n), dtype=qp.ineq.G.dtype)
        for t in range(N + 1):
            G_dense = G_dense.at[
                t * m_ineq : (t + 1) * m_ineq,
                t * n : (t + 1) * n,
            ].set(qp.ineq.G[t])
        Gx_dense = (G_dense @ x).reshape(N + 1, m_ineq)
        np.testing.assert_allclose(Gx_dense, ineq_values, rtol=1e-6, atol=1e-6)

    else:
        G_dense = jnp.zeros((0, (N + 1) * n), dtype=states.dtype)
        Gx_dense = jnp.zeros((N + 1, 0), dtype=states.dtype)

    primal_residual = float(jnp.max(jnp.abs(eq_dense)))
    if m_ineq > 0:
        primal_residual = max(primal_residual, float(jnp.max(jnp.abs(Gx_dense - z_g))))

    P = _build_dense_P_from_cost_blocks(qp.cost)
    q = qp.cost.q.reshape(-1)
    y_f_stack = jnp.concatenate([y_f0, y_f_dyn.reshape(-1)], axis=0)
    dual_vec = P @ x + q + C.T @ y_f_stack
    if m_ineq > 0:
        dual_vec = dual_vec + G_dense.T @ y_g.reshape(-1)

    dual_residual = float(jnp.max(jnp.abs(dual_vec)))

    eq = _dynamics_residual(qp.eq, pack_x(states, controls))
    residuals = _compute_residuals(
        qp,
        ADMMState(
            x_blocks=pack_x(states, controls),
            y_g=y_g,
            y_f_0=y_f0,
            y_f_dyn=y_f_dyn,
            z_g=z_g,
            xi_g=jnp.zeros_like(z_g),
            rho_bar=jnp.array(0.1, dtype=states.dtype),
        ),
    )

    np.testing.assert_allclose(
        float(residuals.primal_residual), primal_residual, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        float(residuals.dual_residual), dual_residual, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("rescale", [False, True])
def test_qp_assembly_matches_osqp_spacecraft_rescaling_implicit(implicit, rescale):
    params = make_spacecraft_params(
        horizon=3,
        implicit=implicit,
        rate_weight=0.5,
        control_weight=1.0,
        ref_weight=1.0,
        final_weight=0.0,
    )
    params["rescale_optimization_variables"] = rescale
    params["state_rescaling_min"] = jnp.array([-0.1, -0.1, -0.1])
    params["state_rescaling_max"] = jnp.array([0.1, 0.07, 0.2])
    params["control_rescaling_min"] = jnp.array([-1.0, -1.0, -1.0])
    params["control_rescaling_max"] = jnp.array([2.0, 50.0, 0.5])

    problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    solver_admm = _make_jax_turbompc_solver(problem)
    solver_osqp = SQPOSQPSolver(program=problem)

    rng = np.random.default_rng(4 if implicit else 5)
    states, controls = problem.initial_guess(params)
    states = states + jnp.array(rng.standard_normal(states.shape)) * 0.1
    controls = controls + jnp.array(rng.standard_normal(controls.shape)) * 0.1

    qp = solver_admm._build_qp_data(states, controls, params)
    P_admm = _build_dense_P_from_cost_blocks(qp.cost)
    q_admm = qp.cost.q.reshape(-1)
    C_admm = _build_dense_C(
        qp.eq, problem.num_state_variables, problem.num_control_variables
    )
    beq_admm = jnp.concatenate([qp.eq.c0, qp.eq.c.reshape(-1)], axis=0)

    P_osqp, q_osqp, Aeq_osqp, beq_osqp, _, _, _ = solver_osqp._build_qp_matrices_dense(
        states, controls, params
    )

    np.testing.assert_allclose(P_admm, P_osqp, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(q_admm, q_osqp, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(C_admm, Aeq_osqp, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(beq_admm, beq_osqp, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("rescale", [False, True])
def test_qp_assembly_initial_control_row_matches_osqp(rescale):
    params = make_spacecraft_params(
        horizon=3,
        implicit=False,
        rate_weight=0.0,
        control_weight=0.1,
        ref_weight=1.0,
        final_weight=0.0,
    )
    params["constrain_initial_control"] = True
    params["initial_control"] = jnp.array([0.2, -0.3, 0.1])
    params["rescale_optimization_variables"] = rescale
    params["state_rescaling_min"] = jnp.array([-0.1, -0.1, -0.1])
    params["state_rescaling_max"] = jnp.array([0.1, 0.07, 0.2])
    params["control_rescaling_min"] = jnp.array([-1.0, -1.0, -1.0])
    params["control_rescaling_max"] = jnp.array([2.0, 50.0, 0.5])

    problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    solver_osqp = SQPOSQPSolver(program=problem)
    states, controls = problem.initial_guess(params)
    _, _, Aeq_osqp, beq_osqp, _, _, _ = solver_osqp._build_qp_matrices_dense(
        states, controls, params
    )

    nx = problem.num_state_variables
    nu = problem.num_control_variables
    row0 = slice(nx, nx + nu)
    col_u0 = slice(nx, nx + nu)

    expected_row = jnp.zeros_like(Aeq_osqp[row0, :])
    expected_row = expected_row.at[:, col_u0].set(jnp.eye(nu))
    np.testing.assert_allclose(Aeq_osqp[row0, :], expected_row, rtol=1e-6, atol=1e-6)

    if rescale:
        control_diff = (
            params["control_rescaling_max"] - params["control_rescaling_min"]
        ) / 2.0
        expected_b = params["initial_control"] / control_diff
    else:
        expected_b = params["initial_control"]
    np.testing.assert_allclose(beq_osqp[row0], expected_b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("rescale", [False, True])
@pytest.mark.parametrize("final_weight", [0.0, 2.0])
@pytest.mark.parametrize("implicit", [False, True])
def test_kkt_residuals_admm_vs_osqp_spacecraft(rescale, final_weight, implicit):
    params = make_spacecraft_params(
        horizon=6,
        implicit=implicit,
        rate_weight=0.5,
        control_weight=0.5,
        ref_weight=5.0,
        final_weight=final_weight,
        initial_state=jnp.array([0.2, -0.2, 0.15]),
        initial_guess_final_state=jnp.array([0.0, 0.0, 0.0]),
        state_bounds=jnp.ones((3,)) * 0.25,
        control_bounds=jnp.ones((3,)) * 0.15,
    )
    params["rescale_optimization_variables"] = rescale
    params["state_rescaling_min"] = jnp.array([-0.1, -0.1, -0.1])
    params["state_rescaling_max"] = jnp.array([0.1, 0.07, 0.2])
    params["control_rescaling_min"] = jnp.array([-1.0, -1.0, -1.0])
    params["control_rescaling_max"] = jnp.array([2.0, 50.0, 0.5])

    problem = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    solver_osqp = SQPOSQPSolver(program=problem)
    solver_admm = _make_jax_turbompc_solver(problem)

    states0, controls0 = problem.initial_guess(params)
    P, q, Aeq, beq, Aineq, l_ineq, u_ineq = solver_osqp._build_qp_matrices_dense(
        states0, controls0, params
    )
    A = jnp.vstack([jnp.array(Aeq), jnp.array(Aineq)])
    l = jnp.concatenate([jnp.array(beq), jnp.array(l_ineq)])
    u = jnp.concatenate([jnp.array(beq), jnp.array(u_ineq)])

    osqp_solver = osqp.OSQP()
    osqp_solver.setup(
        sp.triu(sp.csc_matrix(P)),
        np.array(q),
        sp.csc_matrix(A),
        np.array(l),
        np.array(u),
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=10000,
        verbose=False,
        polish=False,
        warm_start=False,
        scaling=0,
    )
    result = osqp_solver.solve()
    assert result.info.status == "solved"
    z_osqp = jnp.array(result.x, dtype=P.dtype)
    y_osqp = jnp.array(result.y, dtype=P.dtype)

    qp = solver_admm._build_qp_data(states0, controls0, params)
    admm_params = turbompc_solver_params()
    admm_cfg = admm_params["admm"]
    schur_solver = make_schur_solver(
        SchurSolverBackend.PCG,
        solver_admm.program.horizon,
        solver_admm.program.num_state_variables,
        solver_admm.program.num_control_variables,
        pcg_params=admm_cfg["pcg"],
    )
    zshape = ZShape(
        horizon=solver_admm.program.horizon,
        num_states=solver_admm.program.num_state_variables,
        num_controls=solver_admm.program.num_control_variables,
    )
    admm_solver = ADMMSolver(
        zshape=zshape,
        schur_solver=schur_solver,
        pcg_params=admm_cfg["pcg"],
        sigma=admm_cfg["sigma"],
        max_iter=10000,
        eps_abs=1e-8,
        eps_rel=1e-8,
        rho_min=admm_cfg.get("rho_min", 1.0e-6),
        rho_max=admm_cfg.get("rho_max", 1.0e6),
        check_termination_every=admm_cfg.get("check_termination_every", 1),
        adapt_rho_every=admm_cfg.get("adapt_rho_every", 5),
        adaptive_rho_tolerance=admm_cfg.get("adaptive_rho_tolerance", 5.0),
        rho_f_factor=admm_cfg.get(
            "rho_f_factor", admm_cfg.get("active_constraint_rho_factor", 1000.0)
        ),
        admm_backend=AdmmBackend.JAX_LOOP,
    )
    (states_admm, controls_admm), admm_stats, admm_state = admm_solver.solve(
        qp_data=qp,
        rho_bar=admm_cfg["rho"],
    )
    z_admm = pack_z(states_admm, controls_admm)
    y_admm = jnp.concatenate(
        [
            admm_state.y_f_0.reshape(-1),
            admm_state.y_f_dyn.reshape(-1),
            admm_state.y_g.reshape(-1),
        ]
    )

    r_p_osqp, r_d_osqp = _compute_kkt_residuals_dense(P, q, A, l, u, z_osqp, y_osqp)
    r_p_admm, r_d_admm = _compute_kkt_residuals_dense(P, q, A, l, u, z_admm, y_admm)

    assert r_p_osqp < 1.0e-6
    assert r_d_osqp < 1.0e-6
    assert r_p_admm < 1.0e-4
    assert r_d_admm < 2.0e-3


@pytest.mark.parametrize("N", [2, 3])
def test_admm_linear_system_matches_dense(N):
    rng = np.random.default_rng(3)
    nx = 2
    nu = 1
    n = nx + nu
    rho = 0.3
    rho_factor = 4.0
    rho_f = rho * rho_factor
    sigma = 1.0e-3

    As_next = jnp.array(rng.standard_normal((N, nx, nx)))
    As = jnp.array(rng.standard_normal((N, nx, nx)))
    Bs_next = jnp.array(rng.standard_normal((N, nx, nu)))
    Bs = jnp.array(rng.standard_normal((N, nx, nu)))
    Cs = jnp.array(rng.standard_normal((N + 1, nx)))

    Qmat = jnp.array(rng.standard_normal((N + 1, nx, nx)))
    Qmat = (Qmat + jnp.swapaxes(Qmat, -1, -2)) * 0.5
    Rmat = jnp.array(rng.standard_normal((N + 1, nu, nu)))
    Rmat = (Rmat + jnp.swapaxes(Rmat, -1, -2)) * 0.5
    Rd = jnp.array(rng.standard_normal((N, nu, nu)))
    Rd = (Rd + jnp.swapaxes(Rd, -1, -2)) * 0.5
    qvec = jnp.array(rng.standard_normal((N + 1, nx)))
    rvec = jnp.array(rng.standard_normal((N + 1, nu)))

    ineq_blocks = jnp.zeros((N + 1, 0, nx + nu))
    D, E, q = cost_blocks_from_qr(Qmat, Rmat, Rd, qvec, rvec)
    A0 = jnp.concatenate([jnp.eye(nx), jnp.zeros((nx, nu))], axis=1)
    c0 = Cs[0]
    qp_data = qpdata_from_ocp_blocks(
        D=D,
        E=E,
        q=q,
        A0=A0,
        c0=c0,
        As_next=As_next,
        Bs_next=Bs_next,
        As=As,
        Bs=Bs,
        c_dyn=Cs[1:],
        ineq_blocks=ineq_blocks,
        ineq_l=jnp.zeros((N + 1, 0)),
        ineq_u=jnp.zeros((N + 1, 0)),
    )
    schur = compute_S_Phiinv(
        qp_data=qp_data,
        rho_f=rho_f,
        sigma=sigma,
        rho_ineq=rho,
    )
    S_full = _assemble_full_S(schur.S)

    C = _build_dense_C(qp_data.eq, nx, nu)
    P = _build_dense_P_from_cost_blocks(qp_data.cost)
    S_expected = P + sigma * jnp.eye((N + 1) * n) + rho_f * (C.T @ C)
    np.testing.assert_allclose(S_full, S_expected, rtol=1e-6, atol=1e-6)

    states = jnp.array(rng.standard_normal((N + 1, nx)))
    controls = jnp.array(rng.standard_normal((N + 1, nu)))
    z_guess = pack_z(states, controls).reshape(-1)
    y_f = jnp.array(rng.standard_normal((N + 1, nx)))
    y_f0 = y_f[0]
    y_f_dyn = y_f[1:]
    D, E, q = cost_blocks_from_qr(Qmat, Rmat, Rd, qvec, rvec)
    qp_data = qpdata_from_ocp_blocks(
        D=D,
        E=E,
        q=q,
        A0=A0,
        c0=c0,
        As_next=As_next,
        Bs_next=Bs_next,
        As=As,
        Bs=Bs,
        c_dyn=Cs[1:],
        ineq_blocks=ineq_blocks,
        ineq_l=jnp.zeros((N + 1, 0)),
        ineq_u=jnp.zeros((N + 1, 0)),
    )
    gammas = compute_gamma(
        qp_data=qp_data,
        x_blocks=pack_x(states, controls),
        z_g=jnp.zeros((N + 1, 0)),
        y_g=jnp.zeros((N + 1, 0)),
        y_f_0=y_f0,
        y_f_dyn=y_f_dyn,
        rho_f=rho_f,
        rho_ineq=rho,
        sigma=sigma,
    )
    q = pack_z(qvec, rvec).reshape(-1)

    c_stack = jnp.concatenate([Cs[0], Cs[1:].reshape(-1)], axis=0)
    y_f_stack = jnp.concatenate([y_f0, y_f_dyn.reshape(-1)], axis=0)
    dual_term = C.T @ (rho_f * c_stack - y_f_stack)
    gamma_expected = sigma * z_guess - q + dual_term
    gamma_expected = gamma_expected.reshape((N + 1, n))

    np.testing.assert_allclose(gammas, gamma_expected, rtol=1e-6, atol=1e-6)
