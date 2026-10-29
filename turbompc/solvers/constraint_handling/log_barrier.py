"""Two-sided log-barrier inequality handling for the TurboMPC SQP solver."""

from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Tuple

import jax.numpy as jnp

from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.admm import ADMMState
from turbompc.solvers.admm.admm import (
    _apply_C,
    _apply_Ct,
    _apply_G,
    _apply_Gt,
    _apply_P,
    _inf_norm,
)
from turbompc.solvers.admm.inequality_log_barrier import (
    barrier_gradient,
    initialize_barrier_image,
    prox_two_sided_log_barrier,
)
from turbompc.solvers.constraint_handling.base import (
    BackwardQPContext,
    ConstraintHandler,
    Iterate,
)
from turbompc.solvers.linesearch import (
    BarrierLinesearchTerms,
    LinesearchMethod,
    barrier_linesearch,
)
from turbompc.solvers.qp_data import (
    QPCostBlocks,
    QPData,
    QPEqualityBlocks,
    QPInequalityBlocks,
)
from turbompc.solvers.qp_utils import pack_x

if TYPE_CHECKING:
    from turbompc.solvers.turbompc_solver import KKTState, TurboMPCSolution


class BarrierState(NamedTuple):
    """Log-barrier SQP state carried beside ``_SQPState``.

    ``s_g`` is the independent outer image of the nonlinear inequality, kept
    strictly inside every finite bound. SQP drives the consistency residual
    ``g(x) + slack - s_g`` to zero.
    """

    s_g: jnp.ndarray  # (N + 1, m)


class _BarrierBackwardData(NamedTuple):
    barrier_weight: jnp.ndarray  # rowwise weight folded into the adjoint Hessian


# Default fraction of each barrier clearance an SQP step may consume.
_DEFAULT_FRACTION_TO_BOUNDARY = 0.995


class LogBarrierHandler(ConstraintHandler):
    """Inequalities smoothed by a two-sided log barrier."""

    mask_inactive_constraints = False

    def __init__(
        self,
        program: OptimalControlProblem,
        params: Dict[str, Any],
        *,
        linesearch_method: LinesearchMethod,
        convergence_criterion: str,
    ):
        super().__init__(
            program,
            params,
            linesearch_method=linesearch_method,
            convergence_criterion=convergence_criterion,
        )
        self._log_barrier_kappa = float(params["log_barrier_kappa"])
        self._fraction_to_boundary = float(
            params["linesearch"].get(
                "fraction_to_boundary", _DEFAULT_FRACTION_TO_BOUNDARY
            )
        )
        if not 0.0 < self._fraction_to_boundary < 1.0:
            raise ValueError("linesearch.fraction_to_boundary must be between 0 and 1")
        if convergence_criterion != "first_order":
            raise NotImplementedError(
                "constraint_handling='log_barrier' requires"
                " convergence_criterion='first_order', got"
                f" {convergence_criterion!r}."
            )

    def _inequality_bounds_and_image(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        problem_params: Dict[str, Any],
        *,
        with_slack: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return ``(g(x)[+slack], l, u)`` in original constraint coordinates."""
        values, lower, upper = self.program.inequality_constraints(
            states, controls, problem_params
        )
        num_stages = states.shape[0]
        values = values.reshape((num_stages, -1))
        if with_slack and self.program.use_slack_variables:
            values = values + slacks
        return (
            values,
            lower.reshape((num_stages, -1)),
            upper.reshape((num_stages, -1)),
        )

    # --- forward SQP ---------------------------------------------------------

    def init_carry(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        problem_params: Dict[str, Any],
    ) -> BarrierState:
        """Start ``s_g`` at ``g(x) + slack`` moved into the strict interior.

        An infeasible primal start then still has a well-defined barrier.
        """
        values, lower, upper = self._inequality_bounds_and_image(
            states, controls, slacks, problem_params
        )
        return BarrierState(s_g=initialize_barrier_image(values, lower, upper))

    def _barrier_image_from_qp(
        self,
        qp_data: QPData,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        admm_state: ADMMState,
        problem_params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Map the solved inner barrier copy back to constraint coordinates.

        With ``a_g = g(x_j) - G x_j`` formed in the same coordinates as the QP,
        the inner result maps to ``a_g + z_g + xi_g``.
        """
        states_qp, controls_qp = self.program.scale_states_controls(
            states, controls, problem_params
        )
        values, lower, upper = self._inequality_bounds_and_image(
            states, controls, None, problem_params, with_slack=False
        )
        offset = values - _apply_G(qp_data, pack_x(states_qp, controls_qp))
        return offset + admm_state.z_g + admm_state.xi_g, lower, upper

    def apply_linesearch(
        self,
        problem_params: Dict[str, Any],
        qp_data: QPData,
        admm_state: ADMMState,
        carry: BarrierState,
        current: Iterate,
        candidate: Iterate,
    ) -> Tuple[Iterate, BarrierState, jnp.ndarray]:
        """Interpolate a log-barrier SQP step and its barrier image."""
        states, controls, _ = current
        s_g_qp, lower, upper = self._barrier_image_from_qp(
            qp_data, states, controls, admm_state, problem_params
        )
        terms = BarrierLinesearchTerms(
            s_g=carry.s_g,
            s_g_qp=s_g_qp,
            lower=lower,
            upper=upper,
            log_barrier_kappa=jnp.asarray(self._log_barrier_kappa, dtype=states.dtype),
            fraction_to_boundary=self._fraction_to_boundary,
        )
        (
            accepted_states,
            accepted_controls,
            accepted_slacks,
            accepted_s_g,
        ), alpha = barrier_linesearch(
            self._linesearch_method,
            self.program,
            problem_params,
            self._params["linesearch"],
            *current,
            *candidate,
            terms,
        )
        return (
            (accepted_states, accepted_controls, accepted_slacks),
            BarrierState(s_g=accepted_s_g),
            alpha,
        )

    def first_order_errors(
        self,
        qp_data: QPData,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        carry: BarrierState,
        admm_state: ADMMState,
        problem_params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evaluate the smoothed NLP KKT residuals at a log-barrier iterate."""
        states_qp, controls_qp = self.program.scale_states_controls(
            states, controls, problem_params
        )
        x_blocks = pack_x(states_qp, controls_qp)
        stationarity = _inf_norm(
            _apply_P(qp_data, x_blocks)
            + qp_data.cost.q
            + _apply_Ct(qp_data, admm_state.y_f_0, admm_state.y_f_dyn)
            + _apply_Gt(qp_data, admm_state.y_g)
        )
        c_stack = jnp.concatenate([qp_data.eq.c0, qp_data.eq.c.reshape(-1)])
        equality = _inf_norm(_apply_C(qp_data, x_blocks) - c_stack)

        zero = jnp.asarray(0.0, dtype=states.dtype)
        if self.program.num_inequality_constraints == 0:
            return jnp.maximum(stationarity, equality), equality, zero

        s_g = carry.s_g
        values, lower, upper = self._inequality_bounds_and_image(
            states, controls, slacks, problem_params
        )
        consistency = _inf_norm(values - s_g)
        # Report the violation of the original two-sided constraint.
        violation = _inf_norm(
            jnp.maximum(jnp.maximum(lower - values, values - upper), 0.0)
        )
        barrier_dual = _inf_norm(
            admm_state.y_g
            - barrier_gradient(
                s_g,
                lower,
                upper,
                jnp.asarray(self._log_barrier_kappa, dtype=states.dtype),
            )
        )
        if self.program.use_slack_variables:
            slack_stationarity = _inf_norm(
                qp_data.ineq.slack_penalization_weight * slacks + admm_state.y_g
            )
        else:
            slack_stationarity = zero

        convergence_error = jnp.max(
            jnp.stack(
                [
                    stationarity,
                    equality,
                    consistency,
                    slack_stationarity,
                    barrier_dual,
                ]
            )
        )
        return convergence_error, equality, violation

    def active_sets(
        self,
        g: Optional[jnp.ndarray],
        l: jnp.ndarray,
        u: jnp.ndarray,
        slacks: jnp.ndarray,
        y_ineq: Optional[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Every finite bound contributes barrier curvature, so the masks mark
        # finite bounds instead of an identified active set.
        return jnp.isfinite(l), jnp.isfinite(u)

    # --- backward / implicit differentiation ---------------------------------

    def _prepare_backward_solution(self, solution, problem_params):
        """Recompute slack and its multiplier for the saved states and controls."""
        if (
            not self.program.use_slack_variables
            or not self.program.num_inequality_constraints
        ):
            return solution
        values, lower, upper = self._inequality_bounds_and_image(
            solution.states,
            solution.controls,
            None,
            problem_params,
            with_slack=False,
        )
        kappa = jnp.asarray(self._log_barrier_kappa, dtype=values.dtype)
        gamma = problem_params.get(
            "slack_penalization_weight", self.program.slack_penalization_weight
        )
        image = prox_two_sided_log_barrier(values, lower, upper, kappa / gamma)
        slacks = image - values
        dual = barrier_gradient(image, lower, upper, kappa)
        return solution._replace(
            slack=slacks,
            admm_state=solution.admm_state._replace(xi_g=slacks, y_g=dual),
            kkt_state=solution.kkt_state._replace(y_ineq=dual),
        )

    def _barrier_bound_curvatures(
        self,
        forward_solution: "TurboMPCSolution",
        problem_params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Per-row lower and upper barrier curvatures at the converged point.

        These are ``kappa/(s_g - l)**2`` and ``kappa/(u - s_g)**2``, the two
        halves of ``d/ds_g`` of the barrier gradient. Infinite bounds
        contribute nothing.
        """
        states = forward_solution.states
        controls = forward_solution.controls
        values, lower, upper = self.program.inequality_constraints(
            states, controls, problem_params
        )
        shape = (states.shape[0], -1)
        s_g = values.reshape(shape)
        lower = lower.reshape(shape)
        upper = upper.reshape(shape)
        if self.program.use_slack_variables:
            s_g = s_g + forward_solution.slack.reshape(shape)
        kappa = jnp.asarray(self._log_barrier_kappa, dtype=s_g.dtype)
        zero = jnp.zeros_like(s_g)
        return (
            jnp.where(jnp.isfinite(lower), kappa / jnp.square(s_g - lower), zero),
            jnp.where(jnp.isfinite(upper), kappa / jnp.square(upper - s_g), zero),
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
        """Fold the two-sided barrier into an equality-only adjoint QP.

        The barrier image and the signed relaxation are eliminated rowwise, so
        the reduced Hessian gets ``G.T diag(W) G`` added.
        """
        G = qp_data.ineq.G
        lower_term, upper_term = self._barrier_bound_curvatures(
            forward_solution, problem_params
        )
        barrier_weight = lower_term + upper_term
        gamma = jnp.asarray(slack_penalization_weight, dtype=G.dtype)
        if self.program.use_slack_variables:
            barrier_weight = gamma * barrier_weight / (gamma + barrier_weight)
        G_t = jnp.swapaxes(G, -1, -2)
        D = qp_data.cost.D + jnp.matmul(G_t, barrier_weight[..., jnp.newaxis] * G)

        dl_rhs = jnp.concatenate([dl_dstates, dl_dcontrols], axis=-1)
        if self.program.use_slack_variables:
            weighted = (barrier_weight / gamma) * dl_dslack.reshape(
                barrier_weight.shape
            )
            dl_rhs = dl_rhs - _apply_Gt(qp_data, weighted)
        q = -dl_rhs

        n = qp_data.cost.D.shape[-1]
        empty = jnp.zeros((G.shape[0], 0), dtype=G.dtype)
        return BackwardQPContext(
            base_qp=qp_data,
            solve_qp=QPData(
                cost=QPCostBlocks(D=D, E=qp_data.cost.E, q=q),
                eq=eq,
                ineq=QPInequalityBlocks(
                    G=jnp.zeros((G.shape[0], 0, n), dtype=G.dtype),
                    l=empty,
                    u=empty,
                    slack_penalization_weight=qp_data.ineq.slack_penalization_weight,
                ),
            ),
            handler_data=_BarrierBackwardData(barrier_weight=barrier_weight),
        )

    def recover_ineq_gradient(
        self,
        backward_qp: BackwardQPContext,
        forward_solution: "TurboMPCSolution",
        dx_states: jnp.ndarray,
        dx_controls: jnp.ndarray,
        dl_dslack: jnp.ndarray,
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Recover the eliminated barrier and relaxation adjoints rowwise."""
        data = backward_qp.handler_data
        if data is None:
            return None
        barrier_weight = data.barrier_weight
        dx = jnp.concatenate([dx_states, dx_controls], axis=-1)
        g_dx = _apply_G(backward_qp.base_qp, dx)
        y_ineq = barrier_weight * g_dx
        dtype = barrier_weight.dtype
        if not self.program.use_slack_variables:
            return y_ineq.reshape(-1), jnp.array(0.0, dtype=dtype)

        gamma = jnp.asarray(
            backward_qp.base_qp.ineq.slack_penalization_weight, dtype=dtype
        )
        weighted = (barrier_weight / gamma) * dl_dslack.reshape(barrier_weight.shape)
        y_ineq = y_ineq + weighted
        xi = forward_solution.slack.reshape(barrier_weight.shape)
        dL_dgamma = jnp.sum(
            (y_ineq - dl_dslack.reshape(barrier_weight.shape)) * xi / gamma
        )
        return y_ineq.reshape(-1), dL_dgamma

    def residual_multipliers(
        self, kkt_state: "KKTState", y_ineq: jnp.ndarray
    ) -> jnp.ndarray:
        return y_ineq.reshape(-1)

    def bound_weights(
        self,
        forward_solution: "TurboMPCSolution",
        problem_params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Fractions with which each bound shifts the reduced inequality.

        The bounds enter the barrier-dual residual, not ``g``, so perturbing
        them acts like an inequality shift of ``-(H_l dl + H_u du) / H_bar``.
        """
        lower_term, upper_term = self._barrier_bound_curvatures(
            forward_solution, problem_params
        )
        total = lower_term + upper_term
        positive = total > 0.0
        safe = jnp.where(positive, total, jnp.ones_like(total))
        zero = jnp.zeros_like(total)
        return (
            jnp.where(positive, lower_term / safe, zero),
            jnp.where(positive, upper_term / safe, zero),
        )

    def inequality_residual(
        self,
        g: jnp.ndarray,
        l: jnp.ndarray,
        u: jnp.ndarray,
        lower_weight: jnp.ndarray,
        upper_weight: jnp.ndarray,
    ) -> jnp.ndarray:
        # Barrier bound weights are curvature fractions.
        return g - lower_weight * l - upper_weight * u
