"""Interface between the TurboMPC SQP solver and an inequality-handling method."""

import abc
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Tuple

import jax.numpy as jnp

from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.admm import ADMMState
from turbompc.solvers.linesearch import LinesearchMethod
from turbompc.solvers.qp_data import QPData, QPEqualityBlocks

if TYPE_CHECKING:
    from turbompc.solvers.turbompc_solver import KKTState, TurboMPCSolution

# (states, controls, slacks)
Iterate = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


class BackwardQPContext(NamedTuple):
    """Adjoint QP built from the linearized forward QP.

    Attributes:
        base_qp: Linearized forward QP at the solution.
        solve_qp: Adjoint QP passed to the backward solver.
        handler_data: Data used only by the relevant constraint handler.
    """

    base_qp: QPData
    solve_qp: QPData
    handler_data: Any = None


class ConstraintHandler(abc.ABC):
    """Inequality-specific pieces of the SQP forward and backward passes."""

    # Restrict the inequality multipliers in the backward Lagrangian Hessian to
    # the identified active set.
    mask_inactive_constraints: bool = False

    def __init__(
        self,
        program: OptimalControlProblem,
        params: Dict[str, Any],
        *,
        linesearch_method: LinesearchMethod,
        convergence_criterion: str,
    ):
        self._program = program
        self._params = params
        self._linesearch_method = linesearch_method
        self._convergence_criterion = convergence_criterion

    @property
    def program(self) -> OptimalControlProblem:
        return self._program

    # --- forward SQP ---------------------------------------------------------

    @abc.abstractmethod
    def init_carry(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        problem_params: Dict[str, Any],
    ) -> Any:
        """Extra SQP loop carry required by the method (a pytree)."""

    @abc.abstractmethod
    def apply_linesearch(
        self,
        problem_params: Dict[str, Any],
        qp_data: QPData,
        admm_state: ADMMState,
        carry: Any,
        current: Iterate,
        candidate: Iterate,
    ) -> Tuple[Iterate, Any, jnp.ndarray]:
        """Accept a step from ``current`` towards ``candidate``.

        Returns the accepted iterate, the next carry, and the step size.
        """

    @abc.abstractmethod
    def first_order_errors(
        self,
        qp_data: QPData,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        carry: Any,
        admm_state: ADMMState,
        problem_params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return ``(convergence_error, eq_violation, ineq_violation)``."""

    @abc.abstractmethod
    def active_sets(
        self,
        g: Optional[jnp.ndarray],
        l: jnp.ndarray,
        u: jnp.ndarray,
        slacks: jnp.ndarray,
        y_ineq: Optional[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Lower/upper bound masks consumed by the backward pass."""

    # --- backward / implicit differentiation ---------------------------------

    def _prepare_backward_solution(self, solution, problem_params):
        """Return the retained point used by the implicit adjoint."""
        return solution

    @abc.abstractmethod
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
        """Adjoint QP for a problem with at least one inequality row."""

    @abc.abstractmethod
    def recover_ineq_gradient(
        self,
        backward_qp: BackwardQPContext,
        forward_solution: "TurboMPCSolution",
        dx_states: jnp.ndarray,
        dx_controls: jnp.ndarray,
        dl_dslack: jnp.ndarray,
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Recover ``(y_ineq_lin, dL_dgamma)`` for eliminated inequality rows.

        ``None`` means the inequality multipliers are read from the backward
        solver output.
        """

    @abc.abstractmethod
    def residual_multipliers(
        self, kkt_state: "KKTState", y_ineq: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward inequality multipliers matching ``inequality_residual``."""

    @abc.abstractmethod
    def bound_weights(
        self,
        forward_solution: "TurboMPCSolution",
        problem_params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Lower/upper bound weights passed to ``inequality_residual``."""

    @abc.abstractmethod
    def inequality_residual(
        self,
        g: jnp.ndarray,
        l: jnp.ndarray,
        u: jnp.ndarray,
        lower_weight: jnp.ndarray,
        upper_weight: jnp.ndarray,
    ) -> jnp.ndarray:
        """Inequality residual whose parameter derivative enters the gradient."""
