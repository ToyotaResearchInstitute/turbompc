import jax.numpy as jnp
import numpy as np
from tests.helpers.problem_fixtures import cost_blocks_from_qr
from turbompc.solvers.admm import ADMMSolver
from turbompc.solvers.admm.admm import compute_gamma, compute_S_Phiinv
from turbompc.solvers.linear_systems_solvers.backends import (
    AdmmBackend,
    SchurSolverBackend,
)
from turbompc.solvers.linear_systems_solvers.linear_solve import (
    solve_block_tridi_system,
)
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver
from turbompc.solvers.qp_data import qpdata_from_ocp_blocks
from turbompc.solvers.qp_utils import ZShape


def test_admm_runs_on_toy_ocp():
    np.random.seed(0)
    N = 3
    nx = 1
    nu = 1

    As_next = jnp.array(0.1 * np.random.randn(N, nx, nx))
    Bs_next = jnp.array(0.1 * np.random.randn(N, nx, nu))
    As = jnp.array(0.1 * np.random.randn(N, nx, nx))
    Bs = jnp.array(0.1 * np.random.randn(N, nx, nu))
    Cs = jnp.array(0.1 * np.random.randn(N + 1, nx))

    Qmat = jnp.tile(jnp.eye(nx)[None], (N + 1, 1, 1))
    Rmat = jnp.tile(jnp.eye(nu)[None], (N + 1, 1, 1))
    Rd = jnp.tile(jnp.eye(nu)[None], (N, 1, 1)) * 0.05
    qvec = jnp.array(0.1 * np.random.randn(N + 1, nx))
    rvec = jnp.array(0.1 * np.random.randn(N + 1, nu))
    schur_solver = make_schur_solver(
        SchurSolverBackend.PCG,
        N,
        nx,
        nu,
        pcg_params={"max_iter": 200, "tol_epsilon": 1.0e-8},
    )

    x_min = -jnp.ones((N + 1, nx))
    x_max = jnp.ones((N + 1, nx))
    u_min = -jnp.ones((N + 1, nu))
    u_max = jnp.ones((N + 1, nu))
    ineq_blocks = jnp.zeros((N + 1, nx + nu, nx + nu))
    ineq_l = jnp.zeros((N + 1, nx + nu))
    ineq_u = jnp.zeros((N + 1, nx + nu))
    for t in range(N + 1):
        ineq_blocks = ineq_blocks.at[t].set(jnp.eye(nx + nu))
        ineq_l = jnp.concatenate([x_min, u_min], axis=-1)
        ineq_u = jnp.concatenate([x_max, u_max], axis=-1)
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
        ineq_l=ineq_l,
        ineq_u=ineq_u,
    )

    zshape = ZShape(horizon=N, num_states=nx, num_controls=nu)
    solver = ADMMSolver(
        zshape=zshape,
        schur_solver=schur_solver,
        pcg_params={"max_iter": 200, "tol_epsilon": 1.0e-8},
        eps_abs=1e-9,
        eps_rel=1e-5,
        sigma=1.0e-6,
        max_iter=100000,
        admm_backend=AdmmBackend.JAX_LOOP,
    )
    (states, controls), stats, _ = solver.solve(
        qp_data=qp_data,
        rho_bar=0.1,
    )

    assert stats.num_iter > 0
    assert jnp.all(states >= x_min - 1.0e-5)
    assert jnp.all(states <= x_max + 1.0e-5)
    assert jnp.all(controls >= u_min - 1.0e-5)
    assert jnp.all(controls <= u_max + 1.0e-5)

    assert int(stats.num_iter) < solver.max_iter
    last_idx = max(int(stats.num_iter) - 1, 0)
    primal_margin = solver.eps_abs + solver.eps_rel * float(
        stats.primal_residual_norm_terms[last_idx]
    )
    dual_margin = solver.eps_abs + solver.eps_rel * float(
        stats.dual_residual_norm_terms[last_idx]
    )
    assert float(stats.primal_residuals[last_idx]) <= primal_margin
    assert float(stats.dual_residuals[last_idx]) <= dual_margin


def _assemble_full_S(S_blocks: jnp.ndarray) -> jnp.ndarray:
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


def test_pcg_optimal_control_matches_dense():
    np.random.seed(0)
    N = 3
    nx = 2
    nu = 1
    rho = 1.0
    rho_ineq = rho
    rho_f = 1e3 * rho
    sigma = 1.0e-3

    As = jnp.array(0.1 * np.random.randn(N, nx, nx))
    Bs = jnp.array(0.1 * np.random.randn(N, nx, nu))
    As_next = jnp.array(0.1 * np.random.randn(N, nx, nx))
    Bs_next = jnp.array(0.1 * np.random.randn(N, nx, nu))
    Cs = jnp.zeros((N + 1, nx))

    Qmat = jnp.tile(jnp.eye(nx)[None], (N + 1, 1, 1))
    Rmat = jnp.tile(jnp.eye(nu)[None], (N + 1, 1, 1))
    Rd = jnp.tile(jnp.eye(nu)[None], (N, 1, 1)) * 0.1
    qs = jnp.array(0.1 * np.random.randn(N + 1, nx))
    rs = jnp.array(0.1 * np.random.randn(N + 1, nu))

    pcg_params = {"max_iter": 200, "tol_epsilon": 1.0e-10}
    schur_solver_pcg = make_schur_solver(
        SchurSolverBackend.PCG,
        N,
        nx,
        nu,
        pcg_params=pcg_params,
    )
    schur_solver_dense = make_schur_solver(
        SchurSolverBackend.JAX_DENSE,
        N,
        nx,
        nu,
        pcg_params=pcg_params,
    )
    D, E, q = cost_blocks_from_qr(Qmat, Rmat, Rd, qs, rs)
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
        ineq_blocks=jnp.zeros((N + 1, 0, nx + nu)),
        ineq_l=jnp.zeros((N + 1, 0)),
        ineq_u=jnp.zeros((N + 1, 0)),
    )
    schur = compute_S_Phiinv(
        qp_data=qp_data,
        rho_f=rho_f,
        sigma=sigma,
        rho_ineq=rho_ineq,
    )

    zs_guess = jnp.array(0.1 * np.random.randn(N + 1, nx + nu))
    m_ineq = 3
    ineq_blocks = jnp.array(0.2 * np.random.randn(N + 1, m_ineq, nx + nu))
    z_g = jnp.array(0.1 * np.random.randn(N + 1, m_ineq))
    y_g = jnp.array(0.1 * np.random.randn(N + 1, m_ineq))
    y_f = jnp.zeros_like(Cs)
    y_f0 = y_f[0]
    y_f_dyn = y_f[1:]

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
        ineq_l=jnp.zeros((N + 1, m_ineq)),
        ineq_u=jnp.zeros((N + 1, m_ineq)),
    )
    gammas = compute_gamma(
        qp_data=qp_data,
        x_blocks=zs_guess,
        z_g=z_g,
        y_g=y_g,
        y_f_0=y_f0,
        y_f_dyn=y_f_dyn,
        rho_f=rho_f,
        rho_ineq=rho_ineq,
        sigma=sigma,
    )
    z_pcg, _ = schur_solver_pcg.solve(schur, gammas, zs_guess)
    z_ext, _ = schur_solver_dense.solve(schur, gammas, zs_guess)

    S_full = _assemble_full_S(schur.S)
    gamma_full = gammas.reshape((N + 1) * (nx + nu))
    z_ref = jnp.linalg.solve(S_full, gamma_full).reshape((N + 1, nx + nu))
    z_dense = solve_block_tridi_system(
        schur.S, gammas, backend=SchurSolverBackend.JAX_DENSE
    )

    assert np.max(jnp.abs(z_pcg - z_ref)) < 1.0e-5
    assert np.max(jnp.abs(z_pcg - z_dense)) < 1.0e-5
    assert np.max(jnp.abs(z_ext - z_dense)) < 1.0e-5
