"""ADMM solver for box-constrained optimal control QPs."""
from math import isfinite, isnan
from typing import NamedTuple, Optional, Tuple

import jax.numpy as jnp
from jax import lax, vmap

from turbompc.solvers.admm.inequality_log_barrier import (
    log_barrier_prox_update,
    prox_two_sided_log_barrier,
    relaxed_log_barrier_prox_update,
)
from turbompc.solvers.constraint_handling import ConstraintHandling
from turbompc.solvers.linear_systems_solvers.backends import AdmmBackend
from turbompc.solvers.linear_systems_solvers.pcg_primal import SchurComplementMatrices
from turbompc.solvers.linear_systems_solvers.schur_solver import SchurSystemSolver
from turbompc.solvers.qp_data import QPData, QPEqualityBlocks, QPInequalityBlocks
from turbompc.solvers.qp_utils import ZShape, pack_x
from turbompc.solvers.status import ADMMStatus
from turbompc.utils.jax_utils import tree_all_finite


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
    rho_update_max_factor: float
    rho_f_factor: float
    alpha: float  # Over-relaxation parameter
    eps_prim_inf: float = 1.0e-4
    eps_dual_inf: float = 1.0e-4
    check_infeasibility_every: int = 25


class ADMMState(NamedTuple):
    x_blocks: jnp.ndarray  # (N+1, nx+nu)
    y_g: jnp.ndarray  # (N+1, m)
    y_f_0: jnp.ndarray  # (n0,)
    y_f_dyn: jnp.ndarray  # (N, nx)
    z_g: jnp.ndarray  # (N+1, m)
    xi_g: jnp.ndarray  # (N+1, m)
    rho_bar: jnp.ndarray


class ADMMStats(NamedTuple):
    """ADMM solve statistics."""

    status: jnp.ndarray
    num_iter: jnp.ndarray
    primal_residual: jnp.ndarray
    primal_residual_limit: jnp.ndarray
    dual_residual: jnp.ndarray
    dual_residual_limit: jnp.ndarray


class ADMMIterationStats(NamedTuple):
    """ADMM solve statistics over multiple iterates."""

    residuals_valid: jnp.ndarray  # True where residuals and limits were evaluated.
    primal_residuals: jnp.ndarray
    dual_residuals: jnp.ndarray
    primal_residual_limits: jnp.ndarray
    dual_residual_limits: jnp.ndarray
    rhos: jnp.ndarray
    linear_solver_iterations: Optional[jnp.ndarray] = None


class ADMMResult(NamedTuple):
    """ADMM result."""

    state: ADMMState
    stats: ADMMStats
    iterations: Optional[ADMMIterationStats] = None


class FusedADMMSolveResult(NamedTuple):
    """Result returned by a fused ADMM FFI wrapper."""

    state: ADMMState
    stats: ADMMStats
    kernel_ns: jnp.ndarray


class ADMMResiduals(NamedTuple):
    primal_residual: jnp.ndarray
    dual_residual: jnp.ndarray
    primal_residual_normalized: jnp.ndarray
    dual_residual_normalized: jnp.ndarray
    primal_residual_norm_term: jnp.ndarray
    dual_residual_norm_term: jnp.ndarray


class _ADMMInfeasibilityDetection(NamedTuple):
    """Infeasibility decisions from one pair of consecutive iterates."""

    primal_infeasible_detected: jnp.ndarray
    dual_infeasible_detected: jnp.ndarray


def empty_admm_stats(dtype: jnp.dtype, shape: Tuple[int, ...] = ()) -> ADMMStats:
    """ADMM solve statistics for a solve that has not run."""
    zero = jnp.zeros(shape, dtype=dtype)
    return ADMMStats(
        status=jnp.full(shape, jnp.int32(ADMMStatus.NOT_RUN), dtype=jnp.int32),
        num_iter=jnp.zeros(shape, dtype=jnp.int32),
        primal_residual=zero,
        primal_residual_limit=zero,
        dual_residual=zero,
        dual_residual_limit=zero,
    )


def _select_admm_status(
    nonfinite: jnp.ndarray,
    solved: jnp.ndarray,
    primal_infeasible: jnp.ndarray,
    dual_infeasible: jnp.ndarray,
    iteration_limit_reached: jnp.ndarray,
) -> jnp.ndarray:
    """Select status in priority order.

    NONFINITE > SOLVED > PRIMAL_INFEASIBLE > DUAL_INFEASIBLE > MAX_ITER > NOT_RUN.
    """
    return jnp.where(
        nonfinite,
        jnp.int32(ADMMStatus.NONFINITE),
        jnp.where(
            solved,
            jnp.int32(ADMMStatus.SOLVED),
            jnp.where(
                primal_infeasible,
                jnp.int32(ADMMStatus.PRIMAL_INFEASIBLE),
                jnp.where(
                    dual_infeasible,
                    jnp.int32(ADMMStatus.DUAL_INFEASIBLE),
                    jnp.where(
                        iteration_limit_reached,
                        jnp.int32(ADMMStatus.MAX_ITER),
                        jnp.int32(ADMMStatus.NOT_RUN),
                    ),
                ),
            ),
        ),
    )


def _make_admm_stats(
    status: jnp.ndarray,
    num_iter: jnp.ndarray,
    residuals: ADMMResiduals,
    eps_abs: float,
    eps_rel: float,
) -> ADMMStats:
    return ADMMStats(
        status=status,
        num_iter=num_iter.astype(jnp.int32),
        primal_residual=residuals.primal_residual,
        primal_residual_limit=(eps_abs + eps_rel * residuals.primal_residual_norm_term),
        dual_residual=residuals.dual_residual,
        dual_residual_limit=eps_abs + eps_rel * residuals.dual_residual_norm_term,
    )


def _compute_S_blocks(
    qp_data: QPData,
    rho_f: float,
    sigma: float,
    rho_ineq: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble the Schur matrix and return its diagonal/off-diagonal blocks."""
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

    def get_S(thetas, phis):
        def get_S_block(theta, phi, phi_prev):
            return jnp.concatenate([phi_prev, theta, phi.T], axis=1)

        pad = jnp.zeros((1, n, n), dtype=phis.dtype)
        phis_padded = jnp.concatenate([pad, phis, pad], axis=0)
        return vmap(get_S_block)(thetas, phis_padded[1:], phis_padded[:-1])

    S = get_S(thetas, phis)
    return S, thetas, phis


def compute_S(
    qp_data: QPData,
    rho_f: float,
    sigma: float,
    rho_ineq: float = 0.0,
) -> SchurComplementMatrices:
    """Assemble the block-tridiagonal Schur matrix without a preconditioner."""
    S, _, _ = _compute_S_blocks(qp_data, rho_f, sigma, rho_ineq)
    return SchurComplementMatrices(S=S, preconditioner_Phiinv=None)


def compute_S_Phiinv(
    qp_data: QPData,
    rho_f: float,
    sigma: float,
    rho_ineq: float = 0.0,
) -> SchurComplementMatrices:
    """Assemble the block-tridiagonal Schur matrix and Jacobi preconditioner."""
    S, thetas, phis = _compute_S_blocks(qp_data, rho_f, sigma, rho_ineq)
    n = thetas.shape[1]
    thetas_inv = vmap(jnp.linalg.inv)(thetas)

    def get_Phiinv(thetas_inv, phis):
        def get_Phiinv_block(theta_prev_inv, theta_inv, theta_next_inv, phi, phi_prev):
            prod_term_prev = theta_inv @ phi_prev @ theta_prev_inv
            prod_term_next = theta_inv @ phi.T @ theta_next_inv
            return jnp.concatenate([prod_term_prev, -theta_inv, prod_term_next], axis=1)

        pad = jnp.zeros((1, n, n), dtype=thetas_inv.dtype)
        thetas_inv_padded = jnp.concatenate([pad, thetas_inv, pad], axis=0)
        phis_padded = jnp.concatenate([pad, phis, pad], axis=0)
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


def _projection_inequality_update(
    z_tilde: jnp.ndarray,
    ineq: QPInequalityBlocks,
    rho_bar: jnp.ndarray,
    *,
    use_slack_variables: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(z_g, xi_g)`` for hard or quadratically relaxed projection."""
    proj = _project_box(z_tilde, ineq.l, ineq.u)
    if not use_slack_variables:
        return proj, jnp.zeros_like(proj)
    gamma = ineq.slack_penalization_weight
    frac = gamma / (gamma + rho_bar)
    z_g = (1.0 - frac) * z_tilde + frac * proj
    xi_g = (rho_bar / (gamma + rho_bar)) * (proj - z_tilde)
    return z_g, xi_g


def _log_barrier_inequality_update(
    z_tilde: jnp.ndarray,
    ineq: QPInequalityBlocks,
    rho_bar: jnp.ndarray,
    log_barrier_kappa: float,
    *,
    use_slack_variables: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(z_g, xi_g)`` for pure or slack-relaxed log-barrier handling."""
    log_barrier_kappa_array = jnp.asarray(log_barrier_kappa, dtype=z_tilde.dtype)
    if not use_slack_variables:
        return log_barrier_prox_update(
            z_tilde, ineq.l, ineq.u, log_barrier_kappa_array, rho_bar
        )
    return relaxed_log_barrier_prox_update(
        z_tilde,
        ineq.l,
        ineq.u,
        log_barrier_kappa_array,
        rho_bar,
        ineq.slack_penalization_weight,
    )


def _inf_norm(x: jnp.ndarray) -> jnp.ndarray:
    if not x.size:
        return jnp.asarray(0.0, dtype=x.dtype)
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
    *,
    use_slack_variables: bool = False,
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
    if use_slack_variables and qp_data.ineq.G.shape[1] > 0:
        dual_residual = jnp.maximum(
            dual_residual,
            _inf_norm(qp_data.ineq.slack_penalization_weight * state.xi_g + state.y_g),
        )

    Aty = Ct_yf + Gt_yg
    dual_norm_term = jnp.maximum(
        jnp.maximum(_inf_norm(Px), _inf_norm(Aty)),
        _inf_norm(qp_data.cost.q),
    )
    if use_slack_variables and qp_data.ineq.G.shape[1] > 0:
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


def _bound_support(
    direction: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray
) -> jnp.ndarray:
    """Support of a box, avoiding undefined ``0 * inf`` products."""
    positive = direction > 0
    negative = direction < 0
    finite_upper_product = (
        jnp.where(jnp.isposinf(upper), jnp.zeros_like(upper), upper) * direction
    )
    finite_lower_product = (
        jnp.where(jnp.isneginf(lower), jnp.zeros_like(lower), lower) * direction
    )
    positive_terms = jnp.where(
        positive,
        jnp.where(jnp.isposinf(upper), jnp.inf, finite_upper_product),
        0.0,
    )
    negative_terms = jnp.where(
        negative,
        jnp.where(jnp.isneginf(lower), jnp.inf, finite_lower_product),
        0.0,
    )
    return jnp.sum(positive_terms + negative_terms)


def _bound_direction_violation(
    direction: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray
) -> jnp.ndarray:
    """Return how much a direction violates the directions allowed by box bounds."""
    lower_finite = jnp.isfinite(lower)
    upper_finite = jnp.isfinite(upper)
    both = lower_finite & upper_finite
    lower_only = lower_finite & ~upper_finite
    upper_only = ~lower_finite & upper_finite
    violations = jnp.where(
        both,
        jnp.abs(direction),
        jnp.where(
            lower_only,
            jnp.maximum(-direction, 0.0),
            jnp.where(upper_only, jnp.maximum(direction, 0.0), 0.0),
        ),
    )
    return _inf_norm(violations)


def _compute_infeasibility_certificates(
    qp_data: QPData,
    previous: ADMMState,
    current: ADMMState,
    eps_prim_inf: float,
    eps_dual_inf: float,
    *,
    use_slack_variables: bool = False,
) -> _ADMMInfeasibilityDetection:
    """Evaluate infeasibility certificates from consecutive ADMM differences."""
    dx = current.x_blocks - previous.x_blocks
    dxi = current.xi_g - previous.xi_g
    dy_g = current.y_g - previous.y_g
    dy_f_0 = current.y_f_0 - previous.y_f_0
    dy_f_dyn = current.y_f_dyn - previous.y_f_dyn
    c_stack = jnp.concatenate([qp_data.eq.c0, qp_data.eq.c.reshape(-1)])
    dy_f_stack = jnp.concatenate([dy_f_0, dy_f_dyn.reshape(-1)])

    dy_norm = jnp.maximum(_inf_norm(dy_f_stack), _inf_norm(dy_g))
    primal_stationarity = _inf_norm(
        _apply_Ct(qp_data, dy_f_0, dy_f_dyn) + _apply_Gt(qp_data, dy_g)
    )
    if use_slack_variables and dy_g.size:
        primal_stationarity = jnp.maximum(primal_stationarity, _inf_norm(dy_g))
    primal_support = jnp.vdot(c_stack, dy_f_stack)
    if dy_g.size:
        primal_support = primal_support + _bound_support(
            dy_g, qp_data.ineq.l, qp_data.ineq.u
        )

    dx_norm = _inf_norm(dx)
    if use_slack_variables and dxi.size:
        dx_norm = jnp.maximum(dx_norm, _inf_norm(dxi))
    dual_hessian = _inf_norm(_apply_P(qp_data, dx))
    if use_slack_variables and dxi.size:
        dual_hessian = jnp.maximum(
            dual_hessian,
            qp_data.ineq.slack_penalization_weight * _inf_norm(dxi),
        )
    dual_objective = jnp.vdot(qp_data.cost.q, dx)
    equality_direction = _inf_norm(_apply_C(qp_data, dx))
    inequality_direction = _apply_G(qp_data, dx)
    if use_slack_variables:
        inequality_direction = inequality_direction + dxi
    dual_direction_violation = jnp.maximum(
        equality_direction,
        _bound_direction_violation(
            inequality_direction, qp_data.ineq.l, qp_data.ineq.u
        ),
    )

    dtype = dx.dtype
    direction_floor = jnp.asarray(10.0 * jnp.finfo(dtype).eps, dtype=dtype)
    primal_infeasible_detected = (
        (dy_norm > direction_floor)
        & (primal_stationarity <= eps_prim_inf * dy_norm)
        & (primal_support <= -eps_prim_inf * dy_norm)
    )
    dual_infeasible_detected = (
        (dx_norm > direction_floor)
        & (dual_hessian <= eps_dual_inf * dx_norm)
        & (dual_objective <= -eps_dual_inf * dx_norm)
        & (dual_direction_violation <= eps_dual_inf * dx_norm)
    )
    return _ADMMInfeasibilityDetection(
        primal_infeasible_detected=primal_infeasible_detected,
        dual_infeasible_detected=dual_infeasible_detected,
    )


def _update_rho(
    rho_bar: jnp.ndarray,
    residuals: ADMMResiduals,
    rho_min: float,
    rho_max: float,
    rho_update_max_factor: float,
) -> jnp.ndarray:
    ratio = jnp.sqrt(
        residuals.primal_residual_normalized / residuals.dual_residual_normalized
    )
    ratio = jnp.clip(ratio, 1.0 / rho_update_max_factor, rho_update_max_factor)
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
    primal_limits: jnp.ndarray,
    dual_limits: jnp.ndarray,
    pcg_iterations: jnp.ndarray,
    pcg_num_iterations: jnp.ndarray,
    requires_preconditioner: bool,
    use_slack_variables: bool,
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
]:
    compute_schur = compute_S_Phiinv if requires_preconditioner else compute_S

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
                use_slack_variables=use_slack_variables,
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
    ]:
        def _update_stats(_):
            primal_limit = (
                params.eps_abs + params.eps_rel * residuals.primal_residual_norm_term
            )
            dual_limit = (
                params.eps_abs + params.eps_rel * residuals.dual_residual_norm_term
            )
            return (
                res_primals.at[it].set(residuals.primal_residual),
                res_duals.at[it].set(residuals.dual_residual),
                rhos_local,
                primal_limits.at[it].set(primal_limit),
                dual_limits.at[it].set(dual_limit),
                pcg_iterations.at[it].set(pcg_num_iterations.astype(jnp.int32)),
            )

        def _keep_stats(_):
            return (
                res_primals,
                res_duals,
                rhos_local,
                primal_limits,
                dual_limits,
                pcg_iterations.at[it].set(pcg_num_iterations.astype(jnp.int32)),
            )

        return lax.cond(should_check, _update_stats, _keep_stats, operand=None)

    def _maybe_update_rho_and_schur(
        should_check: jnp.ndarray, residuals: ADMMResiduals
    ) -> Tuple[jnp.ndarray, SchurComplementMatrices, jnp.ndarray]:
        if params.adapt_rho_every <= 0:
            return state.rho_bar, schur, jnp.asarray(False)

        def _update_rho_and_schur(_):
            rho_candidate = _update_rho(
                state.rho_bar,
                residuals,
                params.rho_min,
                params.rho_max,
                params.rho_update_max_factor,
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
                lambda _: compute_schur(
                    qp_data,
                    rho_new * params.rho_f_factor,
                    params.sigma,
                    rho_ineq=rho_new,
                ),
                schur,
            )
            return rho_new, schur_new, jnp.logical_not(keep_rho)

        return lax.cond(
            should_check,
            _update_rho_and_schur,
            lambda _: (state.rho_bar, schur, jnp.asarray(False)),
            operand=None,
        )

    should_check_termination = ((it % params.check_termination_every) == 0) | (
        (it + 1) == params.max_iter
    )
    rhos_local = rhos_local.at[it].set(state.rho_bar)
    residuals = _maybe_compute_residuals(should_check_termination)
    (
        res_primals,
        res_duals,
        rhos_local,
        primal_limits,
        dual_limits,
        pcg_iterations,
    ) = _maybe_update_stats(should_check_termination, residuals)
    rho_new, schur, refactorize = _maybe_update_rho_and_schur(
        should_check_termination, residuals
    )

    return (
        residuals,
        rho_new,
        schur,
        refactorize,
        res_primals,
        res_duals,
        rhos_local,
        primal_limits,
        dual_limits,
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
        check_infeasibility_every: int = 25,
        adapt_rho_every: int = 5,
        adaptive_rho_tolerance: float = 5.0,
        rho_update_max_factor: float = 1.0e6,
        rho_f_factor: float = 1000.0,
        admm_backend: AdmmBackend = AdmmBackend.JAX_LOOP,
        use_slack_variables: bool = False,
        eps_prim_inf: float = 1.0e-4,
        eps_dual_inf: float = 1.0e-4,
        constraint_handling: ConstraintHandling = ConstraintHandling.PROJECTION,
        log_barrier_kappa: float = 0.0,
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
        self._check_infeasibility_every = int(check_infeasibility_every)
        self._adapt_rho_every = int(adapt_rho_every)
        if self._max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self._check_termination_every < 1:
            raise ValueError("check_termination_every must be at least 1")
        if self._check_infeasibility_every < 1:
            raise ValueError("check_infeasibility_every must be at least 1")
        self._adaptive_rho_tolerance = float(adaptive_rho_tolerance)
        self._rho_update_max_factor = float(rho_update_max_factor)
        if isnan(self._rho_update_max_factor) or self._rho_update_max_factor < 1.0:
            raise ValueError("rho_update_max_factor must be at least 1.0")
        self._rho_f_factor = float(rho_f_factor)
        self._admm_backend = admm_backend
        self._use_slack_variables = bool(use_slack_variables)
        self._eps_prim_inf = float(eps_prim_inf)
        self._eps_dual_inf = float(eps_dual_inf)
        if not isfinite(self._eps_prim_inf) or self._eps_prim_inf < 0.0:
            raise ValueError("eps_prim_inf must be finite and nonnegative")
        if not isfinite(self._eps_dual_inf) or self._eps_dual_inf < 0.0:
            raise ValueError("eps_dual_inf must be finite and nonnegative")
        self._constraint_handling = ConstraintHandling(constraint_handling)
        self._log_barrier_kappa = float(log_barrier_kappa)
        if self._constraint_handling == ConstraintHandling.LOG_BARRIER:
            if not isfinite(self._log_barrier_kappa) or self._log_barrier_kappa <= 0.0:
                raise ValueError(
                    "log_barrier_kappa must be positive and finite for "
                    "log-barrier handling"
                )
        elif self._log_barrier_kappa != 0.0:
            raise ValueError("log_barrier_kappa must be zero for projection handling")
        self._zshape = zshape
        self._cudss_workspace = None

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
    def check_infeasibility_every(self) -> int:
        return self._check_infeasibility_every

    @property
    def adapt_rho_every(self) -> int:
        return self._adapt_rho_every

    @property
    def adaptive_rho_tolerance(self) -> float:
        return self._adaptive_rho_tolerance

    @property
    def rho_update_max_factor(self) -> float:
        return self._rho_update_max_factor

    @property
    def rho_f_factor(self) -> float:
        return self._rho_f_factor

    @property
    def admm_backend(self) -> AdmmBackend:
        return self._admm_backend

    @property
    def constraint_handling(self) -> ConstraintHandling:
        return self._constraint_handling

    @property
    def log_barrier_kappa(self) -> float:
        return self._log_barrier_kappa

    @property
    def _detects_infeasibility(self) -> bool:
        """Hard-QP infeasibility certificates only apply to projection."""
        return self._constraint_handling == ConstraintHandling.PROJECTION

    def _inequality_update(
        self,
        ineq: QPInequalityBlocks,
        z_tilde: jnp.ndarray,
        rho_bar: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the configured inequality proximal update to ``z_tilde``."""
        if self._constraint_handling == ConstraintHandling.LOG_BARRIER:
            return _log_barrier_inequality_update(
                z_tilde,
                ineq,
                rho_bar,
                self._log_barrier_kappa,
                use_slack_variables=self._use_slack_variables,
            )
        return _projection_inequality_update(
            z_tilde,
            ineq,
            rho_bar,
            use_slack_variables=self._use_slack_variables,
        )

    def solve(
        self,
        qp_data: QPData,
        admm_state0: Optional[ADMMState] = None,
        rho_bar: float = 1.0,
        alpha: float = 1.0,
    ) -> ADMMResult:
        if self._admm_backend != AdmmBackend.JAX_LOOP:
            return self._solve_fused(qp_data, admm_state0, rho_bar, alpha)
        return self._solve_jax_loop(qp_data, admm_state0, rho_bar, alpha)

    def _solve_fused(
        self,
        qp_data: QPData,
        admm_state0: Optional[ADMMState] = None,
        rho_bar: float = 1.0,
        alpha: float = 1.0,
    ) -> ADMMResult:
        """Run fused ADMM with the same termination rules as the JAX loop."""
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
            rho_update_max_factor=self._rho_update_max_factor,
            rho_f_factor=self._rho_f_factor,
            alpha=alpha,
            eps_prim_inf=self._eps_prim_inf,
            eps_dual_inf=self._eps_dual_inf,
            check_infeasibility_every=self._check_infeasibility_every,
        )

        if admm_state0 is None:
            state0 = self.initial_state(qp_data, rho_bar=rho_bar)
        else:
            state0 = admm_state0

        compute_schur = (
            compute_S_Phiinv
            if self._admm_backend == AdmmBackend.FUSED_PCG
            else compute_S
        )
        schur = compute_schur(
            qp_data,
            state0.rho_bar * params.rho_f_factor,
            params.sigma,
            rho_ineq=state0.rho_bar,
        )

        if self._admm_backend == AdmmBackend.FUSED_PCG:
            from turbompc.solvers.admm.admm_ffi_backend import admm_ffi_solve

            result = admm_ffi_solve(
                qp_data,
                schur,
                state0,
                max_iter=params.max_iter,
                pcg_max_iter=int(self._pcg_params.get("max_iter", 200)),
                check_termination_every=params.check_termination_every,
                check_infeasibility_every=params.check_infeasibility_every,
                eps_abs=params.eps_abs,
                eps_rel=params.eps_rel,
                eps_prim_inf=params.eps_prim_inf,
                eps_dual_inf=params.eps_dual_inf,
                sigma=params.sigma,
                rho_f_factor=params.rho_f_factor,
                alpha=params.alpha,
                pcg_eps=float(self._pcg_params.get("tol_epsilon", 1e-8)),
                adapt_rho_every=params.adapt_rho_every,
                adaptive_rho_tolerance=params.adaptive_rho_tolerance,
                rho_update_max_factor=params.rho_update_max_factor,
                rho_min=params.rho_min,
                rho_max=params.rho_max,
                slack_weight=qp_data.ineq.slack_penalization_weight,
                use_slack=self._use_slack_variables,
                constraint_handling=self._constraint_handling,
                log_barrier_kappa=self._log_barrier_kappa,
            )

        elif self._admm_backend == AdmmBackend.FUSED_CUDSS:
            from turbompc.solvers.admm.admm_cudss_ffi_backend import (
                AdmmCudssWorkspace,
                admm_cudss_ffi_solve,
            )

            if self._cudss_workspace is None:
                self._cudss_workspace = AdmmCudssWorkspace()

            result = admm_cudss_ffi_solve(
                qp_data,
                schur,
                state0,
                workspace=self._cudss_workspace,
                max_iter=params.max_iter,
                check_termination_every=params.check_termination_every,
                check_infeasibility_every=params.check_infeasibility_every,
                adapt_rho_every=params.adapt_rho_every,
                eps_abs=params.eps_abs,
                eps_rel=params.eps_rel,
                eps_prim_inf=params.eps_prim_inf,
                eps_dual_inf=params.eps_dual_inf,
                sigma=params.sigma,
                rho_f_factor=params.rho_f_factor,
                alpha=params.alpha,
                adaptive_rho_tolerance=params.adaptive_rho_tolerance,
                rho_update_max_factor=params.rho_update_max_factor,
                rho_min=params.rho_min,
                rho_max=params.rho_max,
                slack_weight=qp_data.ineq.slack_penalization_weight,
                use_slack=self._use_slack_variables,
                constraint_handling=self._constraint_handling,
                log_barrier_kappa=self._log_barrier_kappa,
            )

        else:
            raise ValueError(f"Unknown ADMM backend: {self._admm_backend}")
        return ADMMResult(state=result.state, stats=result.stats, iterations=None)

    def _solve_jax_loop(
        self,
        qp_data: QPData,
        admm_state0: Optional[ADMMState] = None,
        rho_bar: float = 1.0,
        alpha: float = 1.0,
    ) -> ADMMResult:
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
            rho_update_max_factor=self._rho_update_max_factor,
            rho_f_factor=self._rho_f_factor,
            alpha=alpha,
            eps_prim_inf=self._eps_prim_inf,
            eps_dual_inf=self._eps_dual_inf,
            check_infeasibility_every=self._check_infeasibility_every,
        )

        if admm_state0 is None:
            state0 = self.initial_state(
                qp_data,
                rho_bar=rho_bar,
            )
        else:
            state0 = admm_state0

        compute_schur = (
            compute_S_Phiinv if schur_solver.requires_preconditioner else compute_S
        )
        schur = compute_schur(
            qp_data,
            state0.rho_bar * params.rho_f_factor,
            params.sigma,
            rho_ineq=state0.rho_bar,
        )
        dtype = qp_data.cost.q.dtype
        primals = jnp.zeros((params.max_iter,), dtype=dtype)
        duals = jnp.zeros((params.max_iter,), dtype=dtype)
        primal_limits = jnp.zeros((params.max_iter,), dtype=dtype)
        dual_limits = jnp.zeros((params.max_iter,), dtype=dtype)
        pcg_iters = jnp.zeros((params.max_iter,), dtype=jnp.int32)
        rhos = jnp.zeros((params.max_iter,), dtype=dtype)
        detect_infeasibility = self._detects_infeasibility

        def cond_fun(carry):
            it, _, _, _, _, _, _, _, _, _, _, status, _ = carry
            return (it < params.max_iter) & (status == jnp.int32(ADMMStatus.NOT_RUN))

        def body_fun(carry):
            (
                it,
                state,
                schur,
                refactorize,
                res_primals,
                res_duals,
                rhos_local,
                primal_limits,
                dual_limits,
                pcg_iters,
                residuals_prev,
                _,
                certificate_prev,
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
            x_blocks, pcg_debug = schur_solver.solve(
                schur, gammas, zs_guess, factorize=refactorize
            )
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
                z_g, xi_g = self._inequality_update(
                    qp_data.ineq, z_tilde, state.rho_bar
                )
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
                refactorize,
                res_primals,
                res_duals,
                rhos_local,
                primal_limits,
                dual_limits,
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
                primal_limits=primal_limits,
                dual_limits=dual_limits,
                pcg_iterations=pcg_iters,
                pcg_num_iterations=pcg_debug.num_iterations,
                xi_g=xi_g,
                requires_preconditioner=schur_solver.requires_preconditioner,
                use_slack_variables=self._use_slack_variables,
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
            should_check_termination = ((it % params.check_termination_every) == 0) | (
                (it + 1) == params.max_iter
            )
            if detect_infeasibility:
                should_check_infeasibility = (
                    (it % params.check_infeasibility_every) == 0
                ) | ((it + 1) == params.max_iter)
                residuals = lax.cond(
                    should_check_infeasibility & ~should_check_termination,
                    lambda _: _compute_residuals(
                        qp_data,
                        next_state,
                        use_slack_variables=self._use_slack_variables,
                    ),
                    lambda _: residuals,
                    operand=None,
                )
                certificate = lax.cond(
                    should_check_infeasibility,
                    lambda _: _compute_infeasibility_certificates(
                        qp_data,
                        state,
                        next_state,
                        params.eps_prim_inf,
                        params.eps_dual_inf,
                        use_slack_variables=self._use_slack_variables,
                    ),
                    lambda _: certificate_prev,
                    operand=None,
                )
            else:
                should_check_infeasibility = jnp.asarray(False)
                certificate = certificate_prev
            should_check_status = should_check_termination | should_check_infeasibility
            residuals_finite = (
                jnp.isfinite(residuals.primal_residual)
                & jnp.isfinite(residuals.dual_residual)
                & jnp.isfinite(residuals.primal_residual_norm_term)
                & jnp.isfinite(residuals.dual_residual_norm_term)
            )
            finite = residuals_finite & tree_all_finite(next_state)
            converged = (
                should_check_termination
                & finite
                & ~_residuals_too_large(residuals, params.eps_abs, params.eps_rel)
            )
            status = _select_admm_status(
                nonfinite=should_check_status & ~finite,
                solved=converged,
                primal_infeasible=(
                    should_check_infeasibility & certificate.primal_infeasible_detected
                ),
                dual_infeasible=(
                    should_check_infeasibility & certificate.dual_infeasible_detected
                ),
                iteration_limit_reached=(it + 1) >= params.max_iter,
            )
            # Rho updates prepare a subsequent iteration.  Keep the rho used by the
            # final iterate when this check terminates the solve.
            next_state = next_state._replace(
                rho_bar=jnp.where(
                    status == jnp.int32(ADMMStatus.NOT_RUN),
                    next_state.rho_bar,
                    state.rho_bar,
                )
            )
            it_next = it + 1
            return (
                it_next,
                next_state,
                schur,
                refactorize,
                res_primals,
                res_duals,
                rhos_local,
                primal_limits,
                dual_limits,
                pcg_iters,
                residuals,
                status,
                certificate,
            )

        it0 = jnp.asarray(0, dtype=jnp.int32)
        init_residuals = ADMMResiduals(
            primal_residual=jnp.asarray(jnp.inf, dtype=dtype),
            dual_residual=jnp.asarray(jnp.inf, dtype=dtype),
            primal_residual_normalized=jnp.asarray(jnp.inf, dtype=dtype),
            dual_residual_normalized=jnp.asarray(jnp.inf, dtype=dtype),
            primal_residual_norm_term=jnp.asarray(1.0, dtype=dtype),
            dual_residual_norm_term=jnp.asarray(1.0, dtype=dtype),
        )
        init_certificate = _ADMMInfeasibilityDetection(
            jnp.asarray(False), jnp.asarray(False)
        )
        (
            it,
            state,
            _,
            _,
            res_primals,
            res_duals,
            rhos,
            primal_limits,
            dual_limits,
            pcg_iters,
            residuals,
            status,
            certificate,
        ) = lax.while_loop(
            cond_fun,
            body_fun,
            (
                it0,
                state0,
                schur,
                jnp.asarray(True),
                primals,
                duals,
                rhos,
                primal_limits,
                dual_limits,
                pcg_iters,
                init_residuals,
                jnp.int32(ADMMStatus.NOT_RUN),
                init_certificate,
            ),
        )
        stats = _make_admm_stats(
            status,
            it,
            residuals,
            params.eps_abs,
            params.eps_rel,
        )
        indices = jnp.arange(params.max_iter)
        residuals_valid = ((indices % params.check_termination_every) == 0) & (
            indices < it
        )
        residuals_valid = residuals_valid | (
            (indices == (it - 1)) & (it == params.max_iter)
        )
        iterations = ADMMIterationStats(
            residuals_valid=residuals_valid,
            primal_residuals=res_primals,
            dual_residuals=res_duals,
            primal_residual_limits=primal_limits,
            dual_residual_limits=dual_limits,
            rhos=rhos,
            linear_solver_iterations=(
                pcg_iters if schur_solver.requires_preconditioner else None
            ),
        )
        return ADMMResult(state=state, stats=stats, iterations=iterations)

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
            if self._constraint_handling == ConstraintHandling.LOG_BARRIER:
                z_g0 = prox_two_sided_log_barrier(
                    ineq_vals,
                    qp_data.ineq.l,
                    qp_data.ineq.u,
                    jnp.asarray(self._log_barrier_kappa / rho_bar, dtype=dtype),
                )  # (N+1, m)
            else:
                z_g0 = _project_box(
                    ineq_vals, qp_data.ineq.l, qp_data.ineq.u
                )  # (N+1, m)
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
