"""ADMM solver for box-constrained optimal control QPs."""
from typing import NamedTuple, Optional, Tuple

import jax.numpy as jnp
from jax import lax, vmap
from turbompc.solvers.linear_systems_solvers.backends import AdmmBackend
from turbompc.solvers.linear_systems_solvers.pcg_primal import SchurComplementMatrices
from turbompc.solvers.linear_systems_solvers.schur_solver import SchurSystemSolver
from turbompc.solvers.qp_data import QPData, QPEqualityBlocks
from turbompc.solvers.qp_utils import ZShape, pack_x, unpack_x


class ADMMParams(NamedTuple):
    sigma: float
    max_iter: int
    eps_abs: float
    eps_rel: float
    rho_min: float
    rho_max: float
    check_termination_every: int
    adapt_rho_every: int
    adaptive_rho_tolerance: float
    rho_f_factor: float
    alpha: float  # Over-relaxation parameter


class ADMMState(NamedTuple):
    x_blocks: jnp.ndarray  # (N+1, nx+nu)
    y_g: jnp.ndarray  # (N+1, m)
    y_f_0: jnp.ndarray  # (n0,)
    y_f_dyn: jnp.ndarray  # (N, nx)
    z_g: jnp.ndarray  # (N+1, m)
    xi_g: jnp.ndarray  # (N+1, m)
    rho_bar: jnp.ndarray


class ADMMSolveResult(NamedTuple):
    """Result from a fused ADMM FFI solve.

    Using a NamedTuple prevents unpacking errors when new fields are added.
    Callers should use attribute access: result.x_out, result.state, etc.
    """

    x_out: jnp.ndarray  # (T, n) primal solution
    iters_out: jnp.ndarray  # scalar, number of ADMM iterations
    state: ADMMState  # full ADMM state for warm-starting
    kernel_ns: jnp.ndarray  # GPU kernel time in nanoseconds


class ADMMStats(NamedTuple):
    primal_residuals: jnp.ndarray
    dual_residuals: jnp.ndarray
    primal_residual_norm_terms: jnp.ndarray
    dual_residual_norm_terms: jnp.ndarray
    primal_residual_margins: jnp.ndarray
    dual_residual_margins: jnp.ndarray
    num_pcg_iters: jnp.ndarray
    num_iter: jnp.ndarray
    rhos: jnp.ndarray


class ADMMResiduals(NamedTuple):
    primal_residual: jnp.ndarray
    dual_residual: jnp.ndarray
    primal_residual_normalized: jnp.ndarray
    dual_residual_normalized: jnp.ndarray
    primal_residual_norm_term: jnp.ndarray
    dual_residual_norm_term: jnp.ndarray


def compute_S_Phiinv(
    qp_data: QPData,
    rho_f: float,
    sigma: float,
    rho_ineq: float = 0.0,
) -> SchurComplementMatrices:
    """Assemble the block-tridiagonal Schur matrix and Jacobi preconditioner."""
    D = qp_data.cost.D
    E = qp_data.cost.E
    A0 = qp_data.eq.A0
    A_minus = qp_data.eq.A_minus
    A_plus = qp_data.eq.A_plus
    G = qp_data.ineq.G

    N = D.shape[0] - 1
    n = D.shape[1]

    if G.shape[0] == 0:
        GtG = jnp.zeros((D.shape[0], n, n), dtype=D.dtype)
    else:
        GtG = vmap(lambda Gt: Gt.T @ Gt)(G)

    Dtilde = D + sigma * jnp.eye(n, dtype=D.dtype) + rho_ineq * GtG
    A0tA0 = A0.T @ A0
    Aminus_tAminus = vmap(lambda A: A.T @ A)(A_minus)
    Aplus_tAplus = vmap(lambda A: A.T @ A)(A_plus)
    Dtilde = Dtilde.at[0].add(rho_f * (A0tA0 + Aminus_tAminus[0]))
    if N > 1:
        Dtilde = Dtilde.at[1:N].add(rho_f * (Aplus_tAplus[:-1] + Aminus_tAminus[1:]))
    Dtilde = Dtilde.at[N].add(rho_f * Aplus_tAplus[-1])
    Etilde = E + rho_f * vmap(lambda Am, Ap: Ap.T @ Am)(A_minus, A_plus)

    thetas = Dtilde
    phis = Etilde
    thetas_inv = vmap(jnp.linalg.inv)(thetas)

    def get_S(thetas, phis):
        def get_S_block(theta, phi, phi_prev):
            return jnp.concatenate([phi_prev, theta, phi.T], axis=1)

        phis_padded = jnp.concatenate(
            [
                jnp.zeros((1, n, n)),
                phis,
                jnp.zeros((1, n, n)),
            ],
            axis=0,
        )
        return vmap(get_S_block)(thetas, phis_padded[1:], phis_padded[:-1])

    S = get_S(thetas, phis)

    def get_Phiinv(thetas_inv, phis):
        def get_Phiinv_block(theta_prev_inv, theta_inv, theta_next_inv, phi, phi_prev):
            prod_term_prev = theta_inv @ phi_prev @ theta_prev_inv
            prod_term_next = theta_inv @ phi.T @ theta_next_inv
            return jnp.concatenate([prod_term_prev, -theta_inv, prod_term_next], axis=1)

        thetas_inv_padded = jnp.concatenate(
            [
                jnp.zeros((1, n, n)),
                thetas_inv,
                jnp.zeros((1, n, n)),
            ],
            axis=0,
        )
        phis_padded = jnp.concatenate(
            [
                jnp.zeros((1, n, n)),
                phis,
                jnp.zeros((1, n, n)),
            ],
            axis=0,
        )
        return vmap(get_Phiinv_block)(
            thetas_inv_padded[:-2],
            thetas_inv_padded[1:-1],
            thetas_inv_padded[2:],
            phis_padded[1:],
            phis_padded[:-1],
        )

    Phiinv = get_Phiinv(thetas_inv, phis)
    return SchurComplementMatrices(S=S, preconditioner_Phiinv=Phiinv)


def compute_gamma(
    qp_data: QPData,
    x_blocks: jnp.ndarray,
    z_g: jnp.ndarray,
    y_g: jnp.ndarray,
    y_f_0: jnp.ndarray,
    y_f_dyn: jnp.ndarray,
    rho_f: float,
    rho_ineq: float,
    sigma: float,
) -> jnp.ndarray:
    """Assemble the Schur RHS gamma in time-stacked form."""
    q = qp_data.cost.q
    c0_tilde = rho_f * qp_data.eq.c0 - y_f_0
    c_tilde = rho_f * qp_data.eq.c - y_f_dyn
    eq_term = _apply_Ct(qp_data, c0_tilde, c_tilde)
    ineq_term = _apply_Gt(qp_data, rho_ineq * z_g - y_g)
    return sigma * x_blocks - q + eq_term + ineq_term


def _project_box(x: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
    return jnp.minimum(jnp.maximum(x, lower), upper)


def _inf_norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.max(jnp.abs(x))


def _apply_block_tridiag(
    D: jnp.ndarray, E: jnp.ndarray, x_blocks: jnp.ndarray
) -> jnp.ndarray:
    """Apply a block-tridiagonal matrix with blocks (D, E) to x_blocks."""
    n = D.shape[1]
    x_prev = jnp.concatenate(
        [jnp.zeros((1, n), dtype=x_blocks.dtype), x_blocks[:-1]], axis=0
    )
    x_next = jnp.concatenate(
        [x_blocks[1:], jnp.zeros((1, n), dtype=x_blocks.dtype)], axis=0
    )
    E_prev = jnp.concatenate([jnp.zeros((1, n, n), dtype=E.dtype), E], axis=0)
    E_next = jnp.concatenate([E, jnp.zeros((1, n, n), dtype=E.dtype)], axis=0)

    def _apply(Dt, Eprev, Enext, x_prev_t, x_t, x_next_t):
        return Eprev @ x_prev_t + Dt @ x_t + Enext.T @ x_next_t

    return vmap(_apply)(D, E_prev, E_next, x_prev, x_blocks, x_next)


def _apply_P(qp_data: QPData, x_blocks: jnp.ndarray) -> jnp.ndarray:
    return _apply_block_tridiag(qp_data.cost.D, qp_data.cost.E, x_blocks)


def _apply_C_parts(
    qp_data: QPData, x_blocks: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    A0 = qp_data.eq.A0
    A_minus = qp_data.eq.A_minus
    A_plus = qp_data.eq.A_plus
    x_curr = x_blocks[:-1]
    x_next = x_blocks[1:]
    row0 = A0 @ x_blocks[0]
    rows = vmap(lambda Am, Ap, x_t, x_tp1: Am @ x_t + Ap @ x_tp1)(
        A_minus, A_plus, x_curr, x_next
    )
    return row0, rows


def _apply_C(qp_data: QPData, x_blocks: jnp.ndarray) -> jnp.ndarray:
    row0, rows = _apply_C_parts(qp_data, x_blocks)
    return jnp.concatenate([row0, rows.reshape(-1)], axis=0)


def _apply_Ct(
    qp_data: QPData,
    y_f_0: jnp.ndarray,
    y_f_dyn: jnp.ndarray,
) -> jnp.ndarray:
    A0 = qp_data.eq.A0
    A_minus = qp_data.eq.A_minus
    A_plus = qp_data.eq.A_plus
    N = y_f_dyn.shape[0]
    y_curr = y_f_dyn[:-1]
    y_next = y_f_dyn[1:]
    x0 = A0.T @ y_f_0 + A_minus[0].T @ y_f_dyn[0]
    if N == 1:
        xN = A_plus[0].T @ y_f_dyn[0]
        return jnp.stack([x0, xN], axis=0)

    mid = vmap(lambda Ap, Am, y_t, y_tp1: Ap.T @ y_t + Am.T @ y_tp1)(
        A_plus[:-1], A_minus[1:], y_curr, y_next
    )
    xN = A_plus[-1].T @ y_f_dyn[-1]
    return jnp.concatenate([x0[jnp.newaxis], mid, xN[jnp.newaxis]], axis=0)


def _apply_G(qp_data: QPData, x_blocks: jnp.ndarray) -> jnp.ndarray:
    G = qp_data.ineq.G
    if G.shape[1] == 0:
        return jnp.zeros((x_blocks.shape[0], 0), dtype=x_blocks.dtype)
    return vmap(lambda Gt, xt: Gt @ xt)(G, x_blocks)


def _apply_Gt(qp_data: QPData, y_g: jnp.ndarray) -> jnp.ndarray:
    G = qp_data.ineq.G
    if G.shape[1] == 0:
        return jnp.zeros((y_g.shape[0], qp_data.cost.D.shape[1]), dtype=y_g.dtype)
    return vmap(lambda Gt, yt: Gt.T @ yt)(G, y_g)


def _compute_residuals(
    qp_data: QPData,
    state: ADMMState,
) -> ADMMResiduals:
    x_blocks = state.x_blocks

    Cx = _apply_C(qp_data, x_blocks)
    Gx = _apply_G(qp_data, x_blocks)

    c_stack = jnp.concatenate(
        [qp_data.eq.c0, qp_data.eq.c.reshape(-1)],
        axis=0,
    )

    # 1) primal residual
    primal_residual = _inf_norm(Cx - c_stack)
    Gx_inf_norm = 0.0
    z_g_inf_norm = 0.0
    if qp_data.ineq.G.shape[1] > 0:
        primal_residual = jnp.maximum(primal_residual, _inf_norm(Gx - state.z_g))
        Gx_inf_norm = _inf_norm(Gx)
        z_g_inf_norm = _inf_norm(state.z_g)

    primal_norm_term = jnp.maximum(
        jnp.maximum(_inf_norm(Cx), Gx_inf_norm),
        jnp.maximum(_inf_norm(c_stack), z_g_inf_norm),
    )

    # 2) dual residual
    Px = _apply_P(qp_data, x_blocks)
    Ct_yf = _apply_Ct(qp_data, state.y_f_0, state.y_f_dyn)
    Gt_yg = _apply_Gt(qp_data, state.y_g)

    dual_vec = Px + qp_data.cost.q + Ct_yf + Gt_yg
    dual_residual = _inf_norm(dual_vec)
    if qp_data.ineq.use_slack_variables and qp_data.ineq.G.shape[1] > 0:
        dual_residual = jnp.maximum(
            dual_residual,
            _inf_norm(qp_data.ineq.slack_penalization_weight * state.xi_g + state.y_g),
        )

    Aty = Ct_yf + Gt_yg
    dual_norm_term = jnp.maximum(
        jnp.maximum(_inf_norm(Px), _inf_norm(Aty)),
        _inf_norm(qp_data.cost.q),
    )
    if qp_data.ineq.use_slack_variables and qp_data.ineq.G.shape[1] > 0:
        dual_norm_term = jnp.maximum(
            dual_norm_term,
            jnp.maximum(
                qp_data.ineq.slack_penalization_weight * _inf_norm(state.xi_g),
                _inf_norm(state.y_g),
            ),
        )

    primal_residual_normalized = primal_residual / (1e-10 + primal_norm_term)
    dual_residual_normalized = dual_residual / (1e-10 + dual_norm_term)

    return ADMMResiduals(
        primal_residual,
        dual_residual,
        primal_residual_normalized,
        dual_residual_normalized,
        primal_norm_term,
        dual_norm_term,
    )


def _update_rho(
    rho_bar: jnp.ndarray,
    residuals: ADMMResiduals,
    rho_min: float,
    rho_max: float,
) -> jnp.ndarray:
    ratio = jnp.sqrt(
        residuals.primal_residual_normalized / residuals.dual_residual_normalized
    )
    rho_new = jnp.clip(rho_bar * ratio, rho_min, rho_max)
    return rho_new


def _residuals_too_large(
    residuals: ADMMResiduals, eps_abs: float, eps_rel: float
) -> jnp.ndarray:
    return jnp.logical_or(
        residuals.primal_residual
        > (eps_abs + eps_rel * residuals.primal_residual_norm_term),
        residuals.dual_residual
        > (eps_abs + eps_rel * residuals.dual_residual_norm_term),
    )


def _compute_dynamics_dual_terms(
    As_next: jnp.ndarray,
    As: jnp.ndarray,
    Bs_next: jnp.ndarray,
    Bs: jnp.ndarray,
    y_f_dyn: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute dynamics-only C^T y_f terms (no initial constraint)."""
    N = y_f_dyn.shape[0]

    x0 = As[0].T @ y_f_dyn[0]
    u0 = Bs[0].T @ y_f_dyn[0]
    if N == 1:
        xN = As_next[0].T @ y_f_dyn[0]
        uN = Bs_next[0].T @ y_f_dyn[0]
        return jnp.stack([x0, xN], axis=0), jnp.stack([u0, uN], axis=0)

    mid_states = vmap(lambda Ap, Am, y_prev, y_curr: Ap.T @ y_prev + Am.T @ y_curr)(
        As_next[:-1], As[1:], y_f_dyn[:-1], y_f_dyn[1:]
    )
    mid_controls = vmap(lambda Bp, Bm, y_prev, y_curr: Bp.T @ y_prev + Bm.T @ y_curr)(
        Bs_next[:-1], Bs[1:], y_f_dyn[:-1], y_f_dyn[1:]
    )
    xN = As_next[-1].T @ y_f_dyn[-1]
    uN = Bs_next[-1].T @ y_f_dyn[-1]
    dual_states = jnp.concatenate(
        [x0[jnp.newaxis], mid_states, xN[jnp.newaxis]], axis=0
    )
    dual_controls = jnp.concatenate(
        [u0[jnp.newaxis], mid_controls, uN[jnp.newaxis]], axis=0
    )
    return dual_states, dual_controls


def _update_residuals_and_rho(
    *,
    it: jnp.ndarray,
    params: ADMMParams,
    qp_data: QPData,
    state: ADMMState,
    x_blocks: jnp.ndarray,
    z_g: jnp.ndarray,
    xi_g: jnp.ndarray,
    y_g: jnp.ndarray,
    y_f_0: jnp.ndarray,
    y_f_dyn: jnp.ndarray,
    residuals_prev: ADMMResiduals,
    schur: SchurComplementMatrices,
    res_primals: jnp.ndarray,
    res_duals: jnp.ndarray,
    rhos_local: jnp.ndarray,
    prim_terms: jnp.ndarray,
    dual_terms: jnp.ndarray,
    prim_margins: jnp.ndarray,
    dual_margins: jnp.ndarray,
    pcg_iterations: jnp.ndarray,
    pcg_num_iterations: jnp.ndarray,
) -> Tuple[
    ADMMResiduals,
    jnp.ndarray,
    SchurComplementMatrices,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    def _maybe_compute_residuals(should_check: jnp.ndarray) -> ADMMResiduals:
        return lax.cond(
            should_check,
            lambda _: _compute_residuals(
                qp_data,
                state._replace(
                    x_blocks=x_blocks,
                    y_g=y_g,
                    y_f_0=y_f_0,
                    y_f_dyn=y_f_dyn,
                    z_g=z_g,
                    xi_g=xi_g,
                ),
            ),
            lambda _: residuals_prev,
            operand=None,
        )

    def _maybe_update_stats(
        should_check: jnp.ndarray,
        residuals: ADMMResiduals,
    ) -> Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        def _update_stats(_):
            primal_margin = (
                params.eps_abs + params.eps_rel * residuals.primal_residual_norm_term
            )
            dual_margin = (
                params.eps_abs + params.eps_rel * residuals.dual_residual_norm_term
            )
            return (
                res_primals.at[it].set(residuals.primal_residual),
                res_duals.at[it].set(residuals.dual_residual),
                rhos_local.at[it].set(state.rho_bar),
                prim_terms.at[it].set(residuals.primal_residual_norm_term),
                dual_terms.at[it].set(residuals.dual_residual_norm_term),
                prim_margins.at[it].set(primal_margin),
                dual_margins.at[it].set(dual_margin),
                pcg_iterations.at[it].set(pcg_num_iterations.astype(jnp.int32)),
            )

        def _keep_stats(_):
            return (
                res_primals,
                res_duals,
                rhos_local,
                prim_terms,
                dual_terms,
                prim_margins,
                dual_margins,
                pcg_iterations.at[it].set(pcg_num_iterations.astype(jnp.int32)),
            )

        return lax.cond(should_check, _update_stats, _keep_stats, operand=None)

    def _maybe_update_rho_and_schur(
        should_check: jnp.ndarray, residuals: ADMMResiduals
    ) -> Tuple[jnp.ndarray, SchurComplementMatrices]:
        def _update_rho_and_schur(_):
            rho_candidate = _update_rho(
                state.rho_bar, residuals, params.rho_min, params.rho_max
            )
            rhos_ratio = jnp.maximum(
                rho_candidate / state.rho_bar, state.rho_bar / rho_candidate
            )
            admm_has_converged = jnp.logical_not(
                _residuals_too_large(residuals, params.eps_abs, params.eps_rel)
            )
            keep_rho = jnp.logical_or(it < 2, it % params.adapt_rho_every != 0)
            keep_rho = jnp.logical_or(keep_rho, admm_has_converged)
            keep_rho = jnp.logical_or(
                keep_rho, rhos_ratio < params.adaptive_rho_tolerance
            )
            rho_new = jnp.where(keep_rho, state.rho_bar, rho_candidate)
            schur_new = lax.cond(
                keep_rho,
                lambda schur_in: schur_in,
                lambda _: compute_S_Phiinv(
                    qp_data,
                    rho_new * params.rho_f_factor,
                    params.sigma,
                    rho_ineq=rho_new,
                ),
                schur,
            )
            return rho_new, schur_new

        return lax.cond(
            should_check,
            _update_rho_and_schur,
            lambda _: (state.rho_bar, schur),
            operand=None,
        )

    should_check = (it % params.check_termination_every) == 0
    residuals = _maybe_compute_residuals(should_check)
    (
        res_primals,
        res_duals,
        rhos_local,
        prim_terms,
        dual_terms,
        prim_margins,
        dual_margins,
        pcg_iterations,
    ) = _maybe_update_stats(should_check, residuals)
    rho_new, schur = _maybe_update_rho_and_schur(should_check, residuals)

    return (
        residuals,
        rho_new,
        schur,
        res_primals,
        res_duals,
        rhos_local,
        prim_terms,
        dual_terms,
        prim_margins,
        dual_margins,
        pcg_iterations,
    )


def _dynamics_residual_and_values(
    eq: QPEqualityBlocks,
    x_blocks: jnp.ndarray,
) -> jnp.ndarray:
    """Compute stacked dynamics residuals Cx - c."""
    A0 = eq.A0
    A_minus = eq.A_minus
    A_plus = eq.A_plus
    x0 = x_blocks[0]
    Cx0 = A0 @ x0
    Cx_rest = vmap(lambda Am, Ap, x, xnext: Am @ x + Ap @ xnext)(
        A_minus, A_plus, x_blocks[:-1], x_blocks[1:]
    )
    Cx = jnp.concatenate([Cx0, Cx_rest.reshape(-1)], axis=0)
    c_stack = jnp.concatenate([eq.c0, eq.c.reshape(-1)], axis=0)
    return Cx - c_stack, Cx, c_stack


def _dynamics_residual(
    eq: QPEqualityBlocks,
    x_blocks: jnp.ndarray,
) -> jnp.ndarray:
    return _dynamics_residual_and_values(eq, x_blocks)[0]


class ADMMSolver:
    """ADMM loop over primal (state-control) variables."""

    def __init__(
        self,
        zshape: ZShape,
        pcg_params: dict,
        *,
        schur_solver: Optional[SchurSystemSolver] = None,
        sigma: float = 1.0e-6,
        max_iter: int = 50,
        eps_abs: float = 1.0e-5,
        eps_rel: float = 1.0e-4,
        rho_min: float = 1.0e-6,
        rho_max: float = 1.0e6,
        check_termination_every: int = 1,
        adapt_rho_every: int = 5,
        adaptive_rho_tolerance: float = 5.0,
        rho_f_factor: float = 1000.0,
        admm_backend: AdmmBackend = AdmmBackend.JAX_LOOP,
        use_slack: bool = False,
    ):
        if admm_backend == AdmmBackend.JAX_LOOP and schur_solver is None:
            raise ValueError("admm_backend=JAX_LOOP requires schur_solver")
        self._schur_solver = schur_solver
        self._pcg_params = dict(pcg_params)
        self._sigma = float(sigma)
        self._max_iter = max_iter
        self._eps_abs = float(eps_abs)
        self._eps_rel = float(eps_rel)
        self._rho_min = float(rho_min)
        self._rho_max = float(rho_max)
        self._check_termination_every = int(check_termination_every)
        self._adapt_rho_every = int(adapt_rho_every)
        self._adaptive_rho_tolerance = float(adaptive_rho_tolerance)
        self._rho_f_factor = float(rho_f_factor)
        self._admm_backend = admm_backend
        self._use_slack = bool(use_slack)
        self._zshape = zshape

    @property
    def schur_solver(self) -> Optional[SchurSystemSolver]:
        return self._schur_solver

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @property
    def eps_abs(self) -> float:
        return self._eps_abs

    @property
    def eps_rel(self) -> float:
        return self._eps_rel

    @property
    def rho_min(self) -> float:
        return self._rho_min

    @property
    def rho_max(self) -> float:
        return self._rho_max

    @property
    def check_termination_every(self) -> int:
        return self._check_termination_every

    @property
    def adapt_rho_every(self) -> int:
        return self._adapt_rho_every

    @property
    def adaptive_rho_tolerance(self) -> float:
        return self._adaptive_rho_tolerance

    @property
    def rho_f_factor(self) -> float:
        return self._rho_f_factor

    @property
    def admm_backend(self) -> AdmmBackend:
        return self._admm_backend

    def solve(
        self,
        qp_data: QPData,
        admm_state0: Optional[ADMMState] = None,
        rho_bar: float = 1.0,
        alpha: float = 1.0,
        *,
        slack_weight: float = 0.0,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], ADMMStats, ADMMState]:
        if self._admm_backend != AdmmBackend.JAX_LOOP:
            return self._solve_fused(
                qp_data, admm_state0, rho_bar, alpha, slack_weight=slack_weight
            )
        return self._solve_jax_loop(qp_data, admm_state0, rho_bar, alpha)

    def _solve_fused(
        self,
        qp_data: QPData,
        admm_state0: Optional[ADMMState] = None,
        rho_bar: float = 1.0,
        alpha: float = 1.0,
        *,
        slack_weight: float = 0.0,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], ADMMStats, ADMMState]:
        """ADMM solve via fused CUDA mega-kernel (single FFI call).

        Replaces the entire JAX while_loop ADMM with one kernel launch.
        """
        params = ADMMParams(
            sigma=self.sigma,
            max_iter=self.max_iter,
            eps_abs=self._eps_abs,
            eps_rel=self._eps_rel,
            rho_min=self._rho_min,
            rho_max=self._rho_max,
            check_termination_every=self._check_termination_every,
            adapt_rho_every=self._adapt_rho_every,
            adaptive_rho_tolerance=self._adaptive_rho_tolerance,
            rho_f_factor=self._rho_f_factor,
            alpha=alpha,
        )

        if admm_state0 is None:
            state0 = self.initial_state(qp_data, rho_bar=rho_bar)
        else:
            state0 = admm_state0

        schur = compute_S_Phiinv(
            qp_data,
            state0.rho_bar * params.rho_f_factor,
            params.sigma,
            rho_ineq=state0.rho_bar,
        )

        if self._admm_backend == AdmmBackend.FUSED_PCG:
            from turbompc.solvers.admm.admm_ffi_backend import admm_ffi_solve_single

            result = admm_ffi_solve_single(
                qp_data,
                schur,
                state0,
                max_iter=params.max_iter,
                pcg_max_iter=int(self._pcg_params.get("max_iter", 200)),
                check_every=params.check_termination_every,
                eps_abs=params.eps_abs,
                eps_rel=params.eps_rel,
                sigma=params.sigma,
                rho_f_factor=params.rho_f_factor,
                alpha=params.alpha,
                pcg_eps=float(self._pcg_params.get("tol_epsilon", 1e-8)),
                adapt_rho_every=params.adapt_rho_every,
                adaptive_rho_tolerance=params.adaptive_rho_tolerance,
                rho_min=params.rho_min,
                rho_max=params.rho_max,
                slack_weight=slack_weight,
                use_slack=self._use_slack,
            )
        elif self._admm_backend == AdmmBackend.FUSED_CUDSS:
            from turbompc.solvers.admm.admm_cudss_ffi_backend import (
                admm_cudss_ffi_solve_single,
            )

            result = admm_cudss_ffi_solve_single(
                qp_data,
                schur,
                state0,
                max_iter=params.max_iter,
                check_every=params.check_termination_every,
                adapt_rho_every=params.adapt_rho_every,
                eps_abs=params.eps_abs,
                eps_rel=params.eps_rel,
                sigma=params.sigma,
                rho_f_factor=params.rho_f_factor,
                alpha=params.alpha,
                adaptive_rho_tolerance=params.adaptive_rho_tolerance,
                rho_min=params.rho_min,
                rho_max=params.rho_max,
                slack_weight=slack_weight,
                use_slack=self._use_slack,
            )
        else:
            raise ValueError(f"Unknown ADMM backend: {self._admm_backend}")

        # Unpack x_blocks → (states, controls) to match _solve_jax_loop interface
        states, controls = unpack_x(result.state.x_blocks, self._zshape)

        # Minimal stats (fused kernel doesn't track per-iter residuals)
        dtype = qp_data.cost.q.dtype
        max_iter = params.max_iter
        empty = jnp.zeros((max_iter,), dtype=dtype)
        stats = ADMMStats(
            primal_residuals=empty,
            dual_residuals=empty,
            primal_residual_norm_terms=empty,
            dual_residual_norm_terms=empty,
            primal_residual_margins=empty,
            dual_residual_margins=empty,
            num_pcg_iters=jnp.zeros((max_iter,), dtype=jnp.int32),
            num_iter=result.iters_out,
            rhos=empty,
        )
        return (states, controls), stats, result.state

    def _solve_jax_loop(
        self,
        qp_data: QPData,
        admm_state0: Optional[ADMMState] = None,
        rho_bar: float = 1.0,
        alpha: float = 1.0,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], ADMMStats, ADMMState]:
        schur_solver = self._schur_solver
        if schur_solver is None:
            raise RuntimeError("Missing schur_solver for admm_backend=JAX_LOOP")
        params = ADMMParams(
            sigma=self.sigma,
            max_iter=self.max_iter,
            eps_abs=self._eps_abs,
            eps_rel=self._eps_rel,
            rho_min=self._rho_min,
            rho_max=self._rho_max,
            check_termination_every=self._check_termination_every,
            adapt_rho_every=self._adapt_rho_every,
            adaptive_rho_tolerance=self._adaptive_rho_tolerance,
            rho_f_factor=self._rho_f_factor,
            alpha=alpha,
        )

        if admm_state0 is None:
            state0 = self.initial_state(
                qp_data,
                rho_bar=rho_bar,
            )
        else:
            state0 = admm_state0

        schur = compute_S_Phiinv(
            qp_data,
            state0.rho_bar * params.rho_f_factor,
            params.sigma,
            rho_ineq=state0.rho_bar,
        )
        dtype = qp_data.cost.q.dtype
        primals = jnp.zeros((params.max_iter,), dtype=dtype)
        duals = jnp.zeros((params.max_iter,), dtype=dtype)
        prim_terms = jnp.zeros((params.max_iter,), dtype=dtype)
        dual_terms = jnp.zeros((params.max_iter,), dtype=dtype)
        prim_margins = jnp.zeros((params.max_iter,), dtype=dtype)
        dual_margins = jnp.zeros((params.max_iter,), dtype=dtype)
        pcg_iters = jnp.zeros((params.max_iter,), dtype=jnp.int32)
        rhos = jnp.zeros((params.max_iter,), dtype=dtype)

        def cond_fun(carry):
            it, _, _, _, _, _, _, _, _, _, _, residuals = carry
            should_check = (it % params.check_termination_every) == 0
            too_large = _residuals_too_large(residuals, params.eps_abs, params.eps_rel)
            not_maxed = it < params.max_iter
            keep_going = jnp.logical_or(jnp.logical_not(should_check), too_large)
            _continue = jnp.logical_and(not_maxed, keep_going)
            _continue = jnp.logical_or(_continue, it < 1)
            return _continue

        def body_fun(carry):
            (
                it,
                state,
                schur,
                res_primals,
                res_duals,
                rhos_local,
                prim_terms,
                dual_terms,
                prim_margins,
                dual_margins,
                pcg_iters,
                residuals_prev,
            ) = carry

            rho_f = state.rho_bar * params.rho_f_factor
            gammas = compute_gamma(
                qp_data,
                state.x_blocks,
                state.z_g.reshape((state.x_blocks.shape[0], -1)),
                state.y_g.reshape((state.x_blocks.shape[0], -1)),
                state.y_f_0,
                state.y_f_dyn,
                rho_f=rho_f,
                rho_ineq=state.rho_bar,
                sigma=params.sigma,
            )

            zs_guess = state.x_blocks
            x_blocks, pcg_debug = schur_solver.solve(schur, gammas, zs_guess)
            Cx0, Cx = _apply_C_parts(qp_data, x_blocks)
            c0 = qp_data.eq.c0
            c = qp_data.eq.c
            ineq_vals = _apply_G(qp_data, x_blocks)
            # end primal update
            # -------------------------------------

            # over-relaxation
            alpha = params.alpha
            x_blocks = alpha * x_blocks + (1.0 - alpha) * state.x_blocks

            # -------------------------------------
            # slack update
            alpha = params.alpha
            if qp_data.ineq.G.shape[1] == 0:
                z_g = state.z_g
                xi_g = state.xi_g
            else:
                z_tilde = (
                    alpha * ineq_vals
                    + (1.0 - alpha) * state.z_g
                    + state.y_g / state.rho_bar
                )
                proj = _project_box(
                    z_tilde,
                    qp_data.ineq.l,
                    qp_data.ineq.u,
                )
                if qp_data.ineq.use_slack_variables:
                    gamma = qp_data.ineq.slack_penalization_weight
                    frac = gamma / (gamma + state.rho_bar)
                    z_g = (1.0 - frac) * z_tilde + frac * proj
                    xi_g = (state.rho_bar / (gamma + state.rho_bar)) * (proj - z_tilde)
                else:
                    z_g = proj
                    xi_g = jnp.zeros_like(z_g)
            # -------------------------------------

            # -------------------------------------
            # dual update
            y_f_0 = state.y_f_0 + rho_f * (alpha * Cx0 + (1.0 - alpha) * c0 - c0)
            y_f_dyn = state.y_f_dyn + rho_f * (alpha * Cx + (1.0 - alpha) * c - c)
            if qp_data.ineq.G.shape[1] == 0:
                y_g = jnp.zeros((x_blocks.shape[0], 0), dtype=x_blocks.dtype)
            else:
                y_g = state.y_g + state.rho_bar * (
                    alpha * ineq_vals + (1.0 - alpha) * state.z_g - z_g
                )
            # -------------------------------------

            # -------------------------------------
            # residual/rho updates + stats
            (
                residuals,
                rho_new,
                schur,
                res_primals,
                res_duals,
                rhos_local,
                prim_terms,
                dual_terms,
                prim_margins,
                dual_margins,
                pcg_iters,
            ) = _update_residuals_and_rho(
                it=it,
                params=params,
                qp_data=qp_data,
                state=state,
                x_blocks=x_blocks,
                z_g=z_g,
                y_g=y_g,
                y_f_0=y_f_0,
                y_f_dyn=y_f_dyn,
                residuals_prev=residuals_prev,
                schur=schur,
                res_primals=res_primals,
                res_duals=res_duals,
                rhos_local=rhos_local,
                prim_terms=prim_terms,
                dual_terms=dual_terms,
                prim_margins=prim_margins,
                dual_margins=dual_margins,
                pcg_iterations=pcg_iters,
                pcg_num_iterations=pcg_debug.num_iterations,
                xi_g=xi_g,
            )

            next_state = ADMMState(
                x_blocks=x_blocks,
                y_g=y_g,
                y_f_0=y_f_0,
                y_f_dyn=y_f_dyn,
                z_g=z_g,
                xi_g=xi_g,
                rho_bar=rho_new,
            )
            it_next = it + 1
            return (
                it_next,
                next_state,
                schur,
                res_primals,
                res_duals,
                rhos_local,
                prim_terms,
                dual_terms,
                prim_margins,
                dual_margins,
                pcg_iters,
                residuals,
            )

        it0 = jnp.array(0)
        init_residuals = ADMMResiduals(
            primal_residual=jnp.array(jnp.inf),
            dual_residual=jnp.array(jnp.inf),
            primal_residual_normalized=jnp.array(jnp.inf),
            dual_residual_normalized=jnp.array(jnp.inf),
            primal_residual_norm_term=jnp.array(1.0),
            dual_residual_norm_term=jnp.array(1.0),
        )
        (
            it,
            state,
            _,
            res_primals,
            res_duals,
            rhos,
            prim_terms,
            dual_terms,
            prim_margins,
            dual_margins,
            pcg_iters,
            _,
        ) = lax.while_loop(
            cond_fun,
            body_fun,
            (
                it0,
                state0,
                schur,
                primals,
                duals,
                rhos,
                prim_terms,
                dual_terms,
                prim_margins,
                dual_margins,
                pcg_iters,
                init_residuals,
            ),
        )
        stats = ADMMStats(
            primal_residuals=res_primals,
            dual_residuals=res_duals,
            primal_residual_norm_terms=prim_terms,
            dual_residual_norm_terms=dual_terms,
            primal_residual_margins=prim_margins,
            dual_residual_margins=dual_margins,
            num_pcg_iters=pcg_iters,
            num_iter=it,
            rhos=rhos,
        )
        states, controls = unpack_x(state.x_blocks, self._zshape)
        return (states, controls), stats, state

    def initial_state(
        self,
        qp_data: QPData,
        rho_bar: float = 1.0,
        states0: Optional[jnp.ndarray] = None,
        controls0: Optional[jnp.ndarray] = None,
    ) -> ADMMState:
        dtype = qp_data.cost.q.dtype
        if states0 is None:
            states0 = jnp.zeros(
                (self._zshape.horizon + 1, self._zshape.num_states),
                dtype=dtype,
            )
        if controls0 is None:
            controls0 = jnp.zeros(
                (self._zshape.horizon + 1, self._zshape.num_controls),
                dtype=dtype,
            )
        x_blocks0 = pack_x(states0, controls0)
        if qp_data.ineq.G.shape[1] == 0:
            z_g0 = jnp.zeros((self._zshape.horizon + 1, 0), dtype=dtype)
            y_g0 = jnp.zeros_like(z_g0)
            xi_g0 = jnp.zeros_like(z_g0)
        else:
            ineq_vals = vmap(lambda G, z: G @ z)(qp_data.ineq.G, x_blocks0)  # (N+1, m)
            proj = _project_box(ineq_vals, qp_data.ineq.l, qp_data.ineq.u)  # (N+1, m)
            z_g0 = proj
            xi_g0 = jnp.zeros_like(z_g0)
            y_g0 = jnp.zeros_like(z_g0)
        return ADMMState(
            x_blocks=x_blocks0,
            y_g=y_g0,
            y_f_0=jnp.zeros_like(qp_data.eq.c0),
            y_f_dyn=jnp.zeros_like(qp_data.eq.c),
            z_g=z_g0,  # shape (N+1, m)
            xi_g=xi_g0,
            rho_bar=jnp.array(rho_bar, dtype=dtype),
        )
