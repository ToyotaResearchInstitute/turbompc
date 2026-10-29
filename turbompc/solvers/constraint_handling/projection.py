"""Projection (active-set) inequality handling for the TurboMPC SQP solver."""

from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from turbompc.solvers.admm import ADMMState
from turbompc.solvers.constraint_handling.base import (
    BackwardQPContext,
    ConstraintHandler,
    Iterate,
)
from turbompc.solvers.linesearch import linesearch
from turbompc.solvers.qp_data import (
    QPCostBlocks,
    QPData,
    QPEqualityBlocks,
    QPInequalityBlocks,
)
from turbompc.solvers.qp_residuals import _compute_qp_first_order_errors
from turbompc.solvers.qp_utils import pack_x

if TYPE_CHECKING:
    from turbompc.solvers.turbompc_solver import KKTState, TurboMPCSolution


class _ProjectionBackwardData(NamedTuple):
    G_active: jnp.ndarray  # sign * G, with inactive rows zeroed


class ProjectionHandler(ConstraintHandler):
    """Inequalities projected onto their bounds; derivatives use the active set."""

    mask_inactive_constraints = True

    def init_carry(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        problem_params: Dict[str, Any],
    ) -> None:
        return None

    def apply_linesearch(
        self,
        problem_params: Dict[str, Any],
        qp_data: QPData,
        admm_state: ADMMState,
        carry: None,
        current: Iterate,
        candidate: Iterate,
    ) -> Tuple[Iterate, None, jnp.ndarray]:
        accepted, alpha = linesearch(
            self._linesearch_method,
            self.program,
            problem_params,
            self._params["linesearch"],
            *current,
            *candidate,
        )
        return accepted, None, alpha

    def first_order_errors(
        self,
        qp_data: QPData,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        carry: None,
        admm_state: ADMMState,
        problem_params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        states_qp, controls_qp = self.program.scale_states_controls(
            states, controls, problem_params
        )
        x_blocks = pack_x(states_qp, controls_qp)
        errors = _compute_qp_first_order_errors(
            qp_data,
            x_blocks,
            slacks,
            admm_state,
            self.program.use_slack_variables,
        )
        return errors.convergence_error, errors.equality, errors.inequality

    def _identify_active_inequalities(
        self,
        g: jnp.ndarray,
        l: jnp.ndarray,
        u: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Identify active lower/upper inequality constraints.

        Uses the rule:
          min(g-l, u-g) <= eps_abs + eps_rel * |bound|.
        """
        eps_abs = self._params["admm"]["eps_abs"]
        eps_rel = self._params["admm"]["eps_rel"]
        g = jnp.asarray(g)
        l = jnp.asarray(l)
        u = jnp.asarray(u)
        delta_lower = g - l
        delta_upper = u - g
        use_upper = delta_upper < delta_lower
        bound_magnitude = jnp.where(use_upper, jnp.abs(u), jnp.abs(l))
        tol = eps_abs + eps_rel * bound_magnitude
        active = jnp.minimum(delta_lower, delta_upper) <= tol
        active_lower = jnp.logical_and(active, jnp.logical_not(use_upper))
        active_upper = jnp.logical_and(active, use_upper)
        return active_lower, active_upper

    def active_sets(
        self,
        g: Optional[jnp.ndarray],
        l: jnp.ndarray,
        u: jnp.ndarray,
        slacks: jnp.ndarray,
        y_ineq: Optional[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Identify primal/dual active-set."""
        if self.program.use_slack_variables and y_ineq is not None:
            return slacks > 1.0e-12, slacks < -1.0e-12

        if g is None:
            raise ValueError("Inequality values are required without slack activity.")
        active_lower, active_upper = self._identify_active_inequalities(g, l, u)
        if y_ineq is None or y_ineq.shape[-1] == 0:
            return active_lower, active_upper

        eps_abs = self._params["admm"]["eps_abs"]
        dual_lower = y_ineq < -eps_abs
        dual_upper = y_ineq > eps_abs
        undecided = ~(dual_lower | dual_upper)
        return (
            dual_lower | (undecided & active_lower),
            dual_upper | (undecided & active_upper),
        )

    def build_backward_qp(
        self,
        qp_data: QPData,
        eq: QPEqualityBlocks,
        forward_solution: "TurboMPCSolution",
        dl_dstates: jnp.ndarray,
        dl_dcontrols: jnp.ndarray,
        dl_dslack: jnp.ndarray,
        problem_params: Dict[str, Any],
        slack_penalization_weight: jnp.ndarray,
    ) -> BackwardQPContext:
        kkt_state = forward_solution.kkt_state
        sign = kkt_state.ineq_active_lower_idx.astype(
            qp_data.ineq.G.dtype
        ) - kkt_state.ineq_active_upper_idx.astype(qp_data.ineq.G.dtype)
        # G_act = sign * G, with inactive rows zeroed.
        G_active = qp_data.ineq.G * sign[..., jnp.newaxis]

        if self.program.use_slack_variables:
            dl_rhs = jnp.concatenate([dl_dstates, dl_dcontrols], axis=-1)
            gamma = qp_data.ineq.slack_penalization_weight
            dl_dslack = dl_dslack.reshape((dl_rhs.shape[0], -1))
            G_active_t = jnp.swapaxes(G_active, -1, -2)
            D = qp_data.cost.D + gamma * jnp.matmul(G_active_t, G_active)
            rhs_shift = jax.vmap(lambda A, v: A @ v)(G_active_t, dl_dslack)
            q = -(dl_rhs - rhs_shift)
            cost = QPCostBlocks(D=D, E=qp_data.cost.E, q=q)
            n = qp_data.cost.D.shape[-1]
            ineq = QPInequalityBlocks(
                G=jnp.zeros((G_active.shape[0], 0, n), dtype=G_active.dtype),
                l=jnp.zeros((G_active.shape[0], 0), dtype=G_active.dtype),
                u=jnp.zeros((G_active.shape[0], 0), dtype=G_active.dtype),
                slack_penalization_weight=gamma,
            )
        else:
            q = -jnp.concatenate([dl_dstates, dl_dcontrols], axis=-1)
            cost = QPCostBlocks(D=qp_data.cost.D, E=qp_data.cost.E, q=q)
            ineq = QPInequalityBlocks(
                G=G_active,
                l=jnp.zeros_like(qp_data.ineq.l),
                u=jnp.zeros_like(qp_data.ineq.u),
                slack_penalization_weight=qp_data.ineq.slack_penalization_weight,
            )
        return BackwardQPContext(
            base_qp=qp_data,
            solve_qp=QPData(cost=cost, eq=eq, ineq=ineq),
            handler_data=_ProjectionBackwardData(G_active=G_active),
        )

    def recover_ineq_gradient(
        self,
        backward_qp: BackwardQPContext,
        forward_solution: "TurboMPCSolution",
        dx_states: jnp.ndarray,
        dx_controls: jnp.ndarray,
        dl_dslack: jnp.ndarray,
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
        # Without slacks the active rows stay in the adjoint QP.
        data = backward_qp.handler_data
        if not self.program.use_slack_variables or data is None:
            return None
        kkt_state = forward_solution.kkt_state
        active_mask = (
            kkt_state.ineq_active_lower_idx | kkt_state.ineq_active_upper_idx
        ).astype(dl_dslack.dtype)
        dl_dslack_active = dl_dslack * active_mask
        dx = jnp.concatenate([dx_states, dx_controls], axis=-1)
        g_dx = jnp.matmul(data.G_active, dx[..., jnp.newaxis])[..., 0]
        gamma = backward_qp.base_qp.ineq.slack_penalization_weight
        y_ineq_lin = (dl_dslack_active + gamma * g_dx).reshape(-1)
        xi_active = forward_solution.slack * active_mask
        # dL/dgamma needs xi_paper = sign * xi_code.
        sign = kkt_state.ineq_active_lower_idx.astype(
            dl_dslack.dtype
        ) - kkt_state.ineq_active_upper_idx.astype(dl_dslack.dtype)
        dL_dgamma = jnp.sum(sign * xi_active * g_dx)
        return y_ineq_lin, dL_dgamma

    def residual_multipliers(
        self, kkt_state: "KKTState", y_ineq: jnp.ndarray
    ) -> jnp.ndarray:
        sign = kkt_state.ineq_active_lower_idx.astype(
            kkt_state.y_ineq.dtype
        ) - kkt_state.ineq_active_upper_idx.astype(kkt_state.y_ineq.dtype)
        if y_ineq.shape[1] > 0:
            y_ineq = sign * y_ineq
        return y_ineq.reshape(-1)

    def bound_weights(
        self,
        forward_solution: "TurboMPCSolution",
        problem_params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        kkt_state = forward_solution.kkt_state
        return kkt_state.ineq_active_lower_idx, kkt_state.ineq_active_upper_idx

    def inequality_residual(
        self,
        g: jnp.ndarray,
        l: jnp.ndarray,
        u: jnp.ndarray,
        lower_weight: jnp.ndarray,
        upper_weight: jnp.ndarray,
    ) -> jnp.ndarray:
        return lower_weight * (g - l) + upper_weight * (u - g)
