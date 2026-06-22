"""Tests for the PCG solver."""

import jax.numpy as jnp
import numpy as np
import pytest
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.linear_systems_solvers.linear_solve import (
    solve_block_tridi_system,
)
from turbompc.solvers.linear_systems_solvers.pcg_dual import (
    PCGDualOptimalControl,
    QPDynamicsCostPremultipliedMatrices,
)
from turbompc.solvers.qp_data import (
    QPCostBlocks,
    QPData,
    QPEqualityBlocks,
    QPInequalityBlocks,
)


def test_constructor_and_properties():
    """Test solver constructor."""
    solver_params = {"max_iter": 50, "tol_epsilon": 1e-5}
    oc = PCGDualOptimalControl(10, 4, 2, solver_params)

    assert oc.name == "PCGDualOptimalControl"
    assert oc.horizon == 10
    assert oc.num_states == 4
    assert oc.num_controls == 2
    assert oc.params == solver_params


def test_zero_qp_parameters_shapes():
    """Test that dimensions of all QP matrices are correctly defined."""
    oc = PCGDualOptimalControl(5, 3, 2)
    qp_data = oc.zero_qp_data()

    assert qp_data.eq.A_minus.shape == (5, 3, 5)
    assert qp_data.eq.A_plus.shape == (5, 3, 5)
    assert qp_data.eq.c.shape == (5, 3)
    assert qp_data.cost.D.shape == (6, 5, 5)
    assert qp_data.cost.E.shape == (5, 5, 5)
    assert qp_data.cost.q.shape == (6, 5)


def _make_qpdata(As, As_next, Bs, Qs, qs, Rs, rs, Cs) -> QPData:
    N = As.shape[0]
    nx = As.shape[1]
    nu = Bs.shape[2]
    n = nx + nu
    D = jnp.zeros((N + 1, n, n), dtype=Qs.dtype)
    E = jnp.zeros((N, n, n), dtype=Qs.dtype)
    D = D.at[:, :nx, :nx].set(Qs)
    D = D.at[:N, nx:, nx:].set(Rs)
    q = jnp.zeros((N + 1, n), dtype=qs.dtype)
    q = q.at[:, :nx].set(qs)
    q = q.at[:N, nx:].set(rs)
    A0 = jnp.concatenate(
        [jnp.eye(nx, dtype=As.dtype), jnp.zeros((nx, nu), dtype=As.dtype)],
        axis=1,
    )
    A_minus = jnp.concatenate([As, Bs], axis=2)
    A_plus = jnp.concatenate([As_next, jnp.zeros((N, nx, nu), dtype=As.dtype)], axis=2)
    eq = QPEqualityBlocks(A0=A0, A_minus=A_minus, A_plus=A_plus, c0=Cs[0], c=Cs[1:])
    ineq = QPInequalityBlocks(
        G=jnp.zeros((N + 1, 0, n), dtype=As.dtype),
        l=jnp.zeros((N + 1, 0), dtype=As.dtype),
        u=jnp.zeros((N + 1, 0), dtype=As.dtype),
        slack_penalization_weight=jnp.array(0.0, dtype=As.dtype),
        use_slack_variables=False,
    )
    return QPData(cost=QPCostBlocks(D=D, E=E, q=q), eq=eq, ineq=ineq)


def test_zero_schur_and_dynamics_cost_shapes():
    """Test that dimensions of the cost matrices are correctly defined."""
    oc = PCGDualOptimalControl(4, 2, 1)
    dyn_cost = oc.zero_dynamics_cost_premultiplied_matrices()
    schur = oc.zero_schur_complement_matrices()

    assert dyn_cost.As_x_Qinv.shape == (4, 2, 2)
    assert dyn_cost.Asnext_x_Qinvnext.shape == (4, 2, 2)
    assert dyn_cost.Bs_x_Rinv.shape == (4, 2, 1)
    assert schur.S.shape == (5, 2, 6)
    assert schur.preconditioner_Phiinv.shape == (5, 2, 6)


@pytest.mark.parametrize("T", [2, 5])
@pytest.mark.parametrize("nx", [1, 5])
@pytest.mark.parametrize("nu", [1, 5])
def test_compute_S_Phiinv_identity_dynamics(T, nx, nu):
    """Test computation of the preconditioner."""

    oc = PCGDualOptimalControl(T, nx, nu, {"max_iter": 10, "tol_epsilon": 1e-8})

    As = jnp.broadcast_to(jnp.eye(nx), (T, nx, nx))
    As_next = jnp.broadcast_to(jnp.eye(nx), (T, nx, nx))
    Bs = jnp.ones((T, nx, nu))
    Cs = jnp.ones((T + 1, nx))
    Qs = jnp.tile(jnp.eye(nx), (T + 1, 1, 1))
    Rs = jnp.tile(jnp.eye(nu), (T, 1, 1))
    qs = jnp.zeros((T + 1, nx))
    rs = jnp.zeros((T, nu))
    qp_data = _make_qpdata(As, As_next, Bs, Qs, qs, Rs, rs, Cs)

    schur = oc.compute_S_Phiinv(qp_data)
    assert schur.S.shape == (T + 1, nx, nx * 3)
    assert schur.preconditioner_Phiinv.shape[0] == T + 1
    assert not jnp.isnan(schur.S).any()


@pytest.mark.parametrize("T", [2, 5])
@pytest.mark.parametrize("nx", [1, 5])
@pytest.mark.parametrize("nu", [1, 5])
def test_compute_gamma_zeros(T, nx, nu):
    """Test computation of the RHS gamma."""

    oc = PCGDualOptimalControl(T, nx, nu)
    dyn_cost = oc.zero_dynamics_cost_premultiplied_matrices()
    Cs = jnp.zeros((T + 1, nx))
    Q0inv = jnp.eye(nx)
    qs = jnp.zeros((T + 1, nx))
    rs = jnp.zeros((T, nu))

    gamma = oc.compute_gamma(dyn_cost, Cs, Q0inv, qs, rs)
    assert gamma.shape == (T + 1, nx)
    assert jnp.allclose(gamma, 0.0)


@pytest.mark.parametrize("T", [2, 5])
@pytest.mark.parametrize("nx", [1, 5])
@pytest.mark.parametrize("nu", [1, 5])
def test_get_states_controls_from_zero_multipliers(T, nx, nu):
    """Test retrieving the solution."""
    oc = PCGDualOptimalControl(T, nx, nu)
    qp_seed = oc.zero_qp_data()
    As = qp_seed.eq.A_minus[:, :, :nx]
    Bs = qp_seed.eq.A_minus[:, :, nx:]
    As_next = qp_seed.eq.A_plus[:, :, :nx]
    Cs = jnp.concatenate([qp_seed.eq.c0[jnp.newaxis], qp_seed.eq.c], axis=0)
    Qs = jnp.tile(jnp.eye(nx), (T + 1, 1, 1))
    Rs = jnp.tile(jnp.eye(nu), (T, 1, 1))
    qs = jnp.zeros((T + 1, nx))
    rs = jnp.zeros((T, nu))
    qp_data = _make_qpdata(As, As_next, Bs, Qs, qs, Rs, rs, Cs)

    lambdas = jnp.zeros((T + 1, nx))
    sol = oc.get_states_controls_from_kkt_multipliers(lambdas, qp_data)

    assert sol.states.shape == (T + 1, nx)
    assert sol.controls.shape == (T + 1, nu)
    assert jnp.allclose(sol.states, 0.0)
    assert jnp.allclose(sol.controls, 0.0)


@pytest.mark.parametrize("T", [2, 3])
@pytest.mark.parametrize("nx", [1, 2])
@pytest.mark.parametrize("nu", [1, 2])
def test_solve_KKT_Schur_trivial(T, nx, nu):
    """Test solving a trivial KKT system."""

    oc = PCGDualOptimalControl(T, nx, nu, {"max_iter": 10, "tol_epsilon": 1e-8})

    As = jnp.broadcast_to(jnp.eye(nx), (T, nx, nx))
    As_next = jnp.broadcast_to(jnp.eye(nx), (T, nx, nx))
    Bs = jnp.ones((T, nx, nu))
    Cs = jnp.ones((T + 1, nx))
    Qs = jnp.tile(jnp.eye(nx), (T + 1, 1, 1))
    Rs = jnp.tile(jnp.eye(nu), (T, 1, 1))
    qs = jnp.zeros((T + 1, nx))
    rs = jnp.zeros((T, nu))
    qp_data = _make_qpdata(As, As_next, Bs, Qs, qs, Rs, rs, Cs)

    schur = oc.compute_S_Phiinv(qp_data)

    dyn_cost = oc.zero_dynamics_cost_premultiplied_matrices()
    Q0inv = jnp.eye(nx)
    qs = jnp.zeros((T + 1, nx))
    rs = jnp.zeros((T, nu))
    # Cs length must match T+1
    gammas = oc.compute_gamma(dyn_cost, jnp.ones((T + 1, nx)), Q0inv, qs, rs)
    lambdas_guess = jnp.zeros((T + 1, nx))

    sol, debug = oc.solve_KKT_Schur(qp_data, schur, gammas, lambdas_guess)

    assert sol.states.shape == (T + 1, nx)
    assert sol.controls.shape == (T + 1, nu)
    assert debug.num_iterations <= 10
    assert debug.convergence_eta >= 0.0


def test_solve_KKT_Schur_warmstart():
    """Test warm-starting PCG."""
    (T, nx, nu) = (3, 4, 5)

    oc = PCGDualOptimalControl(T, nx, nu, {"max_iter": 20, "tol_epsilon": 1e-13})

    As = jnp.broadcast_to(jnp.diag(np.random.rand(nx)), (T, nx, nx))
    As_next = jnp.broadcast_to(jnp.diag(np.random.rand(nx)), (T, nx, nx))
    Bs = jnp.ones((T, nx, nu))
    Cs = jnp.ones((T + 1, nx))
    Qs = jnp.tile(jnp.diag(np.random.rand(nx)), (T + 1, 1, 1))
    Rs = jnp.tile(jnp.eye(nu), (T, 1, 1))
    qs = jnp.zeros((T + 1, nx))
    rs = jnp.zeros((T, nu))
    qp_data = _make_qpdata(As, As_next, Bs, Qs, qs, Rs, rs, Cs)
    schur = oc.compute_S_Phiinv(qp_data)
    dyn_cost = oc.zero_dynamics_cost_premultiplied_matrices()
    Q0inv = jnp.eye(nx)
    qs = jnp.zeros((T + 1, nx))
    rs = jnp.zeros((T, nu))
    # Cs length must match T+1
    gammas = oc.compute_gamma(dyn_cost, jnp.ones((T + 1, nx)), Q0inv, qs, rs)
    lambdas_guess = jnp.zeros((T + 1, nx))

    sol, debug = oc.solve_KKT_Schur(qp_data, schur, gammas, lambdas_guess)

    assert sol.states.shape == (T + 1, nx)
    assert sol.controls.shape == (T + 1, nu)
    assert debug.num_iterations <= 10
    assert debug.convergence_eta >= 0.0

    sol_warm, debug_warm = oc.solve_KKT_Schur(
        qp_data, schur, gammas, sol.kkt_multipliers
    )
    assert sol_warm.states.shape == (T + 1, nx)
    assert sol_warm.controls.shape == (T + 1, nu)
    assert debug_warm.num_iterations <= 10
    assert debug_warm.convergence_eta >= 0.0
    # Warm-starting changes the PCG trajectory; allow small numerical differences.
    np.testing.assert_allclose(sol.states, sol_warm.states, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sol.controls, sol_warm.controls, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        sol.kkt_multipliers, sol_warm.kkt_multipliers, rtol=1e-6, atol=1e-6
    )
    assert debug_warm.num_iterations <= debug.num_iterations


def test_pcg_dual_matches_block_tridi_scipy():
    np.random.seed(0)
    T = 3
    nx = 2
    nu = 1
    oc = PCGDualOptimalControl(T, nx, nu, {"max_iter": 2000, "tol_epsilon": 1e-12})

    As = jnp.broadcast_to(jnp.eye(nx), (T, nx, nx))
    As_next = jnp.broadcast_to(jnp.eye(nx), (T, nx, nx))
    Bs = 0.1 * jnp.ones((T, nx, nu))

    Cs = jnp.array(np.random.randn(T + 1, nx))
    Qs = jnp.tile(jnp.eye(nx), (T + 1, 1, 1))
    Rs = jnp.tile(jnp.eye(nu), (T, 1, 1))
    qs = jnp.array(np.random.randn(T + 1, nx))
    rs = jnp.array(np.random.randn(T, nu))

    qp_data = _make_qpdata(As, As_next, Bs, Qs, qs, Rs, rs, Cs)
    schur = oc.compute_S_Phiinv(qp_data)

    Qinv = jnp.tile(jnp.eye(nx), (T + 1, 1, 1))
    Rinv = jnp.tile(jnp.eye(nu), (T, 1, 1))
    dyn_cost = QPDynamicsCostPremultipliedMatrices(
        As @ Qinv[:-1], As_next @ Qinv[1:], Bs @ Rinv
    )
    Q0inv = jnp.eye(nx)
    gammas = oc.compute_gamma(dyn_cost, Cs, Q0inv, qs, rs)

    lambdas_guess = jnp.zeros((T + 1, nx))
    sol, _ = oc.solve_KKT_Schur(qp_data, schur, gammas, lambdas_guess)

    sol_ext, _ = oc.solve_KKT_Schur_external(
        qp_data, schur, gammas, backend=SchurSolverBackend.JAX_DENSE
    )
    lambdas_ref = solve_block_tridi_system(
        schur.S, gammas, backend=SchurSolverBackend.JAX_DENSE
    )

    np.testing.assert_allclose(sol.kkt_multipliers, lambdas_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        sol_ext.kkt_multipliers, lambdas_ref, rtol=1e-6, atol=1e-6
    )
