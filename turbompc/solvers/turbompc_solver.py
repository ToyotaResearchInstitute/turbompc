"""TurboMPC SQP solver: Uses ADMM to solve the QP subproblems."""

import copy
import enum
from functools import partial
from numbers import Integral
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.admm import ADMMSolver, ADMMState, ADMMStats, empty_admm_stats
from turbompc.solvers.admm.admm import _apply_G
from turbompc.solvers.backward.backward_kkt_jax import solve_backward_kkt
from turbompc.solvers.constraint_handling import (
    ConstraintHandling,
    _validate_constraint_settings,
)
from turbompc.solvers.constraint_handling.base import (
    BackwardQPContext,
    ConstraintHandler,
)
from turbompc.solvers.constraint_handling.log_barrier import LogBarrierHandler
from turbompc.solvers.constraint_handling.projection import ProjectionHandler
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver
from turbompc.solvers.qp_data import (
    QPCostBlocks,
    QPData,
    QPEqualityBlocks,
    qpdata_from_ocp_blocks,
    scale_qp_data,
)

try:
    from turbompc.solvers.backward.backward_kkt_cudss_ffi import CudssBackwardKktSolver

    _HAS_CUDSS_FFI = True
except (FileNotFoundError, OSError, ImportError):
    _HAS_CUDSS_FFI = False
    CudssBackwardKktSolver = None
from turbompc.solvers.linear_systems_solvers.backends import (
    AdmmBackend,
    SchurSolverBackend,
)
from turbompc.solvers.linesearch import (
    LinesearchMethod,
    evaluate_constraints_with_bounds,
    parse_linesearch_method,
)
from turbompc.solvers.qp_residuals import _compute_qp_constraint_errors
from turbompc.solvers.qp_utils import ZShape, pack_x, pack_z, unpack_x
from turbompc.solvers.status import ADMMStatus, SQPStatus
from turbompc.utils.load_params import load_solver_params, normalize_problem_params

DEFAULT_SOLVER_PARAMS = load_solver_params("turbompc.yaml")

_CONSTRAINT_HANDLERS: dict[ConstraintHandling, type[ConstraintHandler]] = {
    ConstraintHandling.PROJECTION: ProjectionHandler,
    ConstraintHandling.LOG_BARRIER: LogBarrierHandler,
}


class ForwardBackend(enum.IntEnum):
    """Backend used to solve the forward QP (one SQP iteration)."""

    ADMM_JAX_LOOP_PCG = 0
    ADMM_JAX_LOOP_PCG_FFI = 1
    ADMM_JAX_LOOP_CUDSS_FFI = 2
    ADMM_JAX_LOOP_JAX_DENSE = 3
    ADMM_FUSED_PCG = 4
    ADMM_FUSED_CUDSS = 5
    ADMM_JAX_LOOP_EIGEN = 6


class BackwardBackend(enum.IntEnum):
    """Backend used to compute gradients."""

    ADMM_JAX_LOOP_PCG = 0
    ADMM_JAX_LOOP_PCG_FFI = 1
    ADMM_JAX_LOOP_CUDSS_FFI = 2
    ADMM_JAX_LOOP_JAX_DENSE = 3
    ADMM_FUSED_PCG = 4
    ADMM_FUSED_CUDSS = 5
    DIRECT_JAX_DENSE = 6
    DIRECT_CUDSS_FFI = 7
    ADMM_JAX_LOOP_EIGEN = 8


class HessianApproximation(enum.IntEnum):
    """Hessian approximation used for forward QPs and the backward pass."""

    GAUSS_NEWTON = 0
    FULL_HESSIAN = 1
    # FULL_HESSIAN_PROJECTED adds the exact Lagrangian Hessian terms and changes the
    # diagonal block using neighboring off-diagonal Frobenius norms, preserving the
    # block-tridiagonal sparsity pattern while making the full Hessian PSD.
    FULL_HESSIAN_PROJECTED = 2  # Only supported  for the forward pass.


_FORWARD_BACKEND_COMPONENTS: dict[
    ForwardBackend, tuple[AdmmBackend, SchurSolverBackend]
] = {
    ForwardBackend.ADMM_JAX_LOOP_PCG: (AdmmBackend.JAX_LOOP, SchurSolverBackend.PCG),
    ForwardBackend.ADMM_JAX_LOOP_PCG_FFI: (
        AdmmBackend.JAX_LOOP,
        SchurSolverBackend.PCG_FFI,
    ),
    ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI: (
        AdmmBackend.JAX_LOOP,
        SchurSolverBackend.CUDSS_FFI,
    ),
    ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE: (
        AdmmBackend.JAX_LOOP,
        SchurSolverBackend.JAX_DENSE,
    ),
    ForwardBackend.ADMM_JAX_LOOP_EIGEN: (
        AdmmBackend.JAX_LOOP,
        SchurSolverBackend.EIGEN,
    ),
    ForwardBackend.ADMM_FUSED_PCG: (AdmmBackend.FUSED_PCG, SchurSolverBackend.IGNORED),
    ForwardBackend.ADMM_FUSED_CUDSS: (
        AdmmBackend.FUSED_CUDSS,
        SchurSolverBackend.IGNORED,
    ),
}


_BACKWARD_ADMM_COMPONENTS: dict[
    BackwardBackend, tuple[AdmmBackend, SchurSolverBackend]
] = {
    BackwardBackend.ADMM_JAX_LOOP_PCG: (AdmmBackend.JAX_LOOP, SchurSolverBackend.PCG),
    BackwardBackend.ADMM_JAX_LOOP_PCG_FFI: (
        AdmmBackend.JAX_LOOP,
        SchurSolverBackend.PCG_FFI,
    ),
    BackwardBackend.ADMM_JAX_LOOP_CUDSS_FFI: (
        AdmmBackend.JAX_LOOP,
        SchurSolverBackend.CUDSS_FFI,
    ),
    BackwardBackend.ADMM_JAX_LOOP_JAX_DENSE: (
        AdmmBackend.JAX_LOOP,
        SchurSolverBackend.JAX_DENSE,
    ),
    BackwardBackend.ADMM_JAX_LOOP_EIGEN: (
        AdmmBackend.JAX_LOOP,
        SchurSolverBackend.EIGEN,
    ),
    BackwardBackend.ADMM_FUSED_PCG: (AdmmBackend.FUSED_PCG, SchurSolverBackend.IGNORED),
    BackwardBackend.ADMM_FUSED_CUDSS: (
        AdmmBackend.FUSED_CUDSS,
        SchurSolverBackend.IGNORED,
    ),
}


_NAME_TO_FORWARD_BACKEND: dict[str, ForwardBackend] = {
    "admm_jax_loop_pcg": ForwardBackend.ADMM_JAX_LOOP_PCG,
    "admm_jax_loop_pcg_ffi": ForwardBackend.ADMM_JAX_LOOP_PCG_FFI,
    "admm_jax_loop_cudss_ffi": ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
    "admm_jax_loop_jax_dense": ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
    "admm_jax_loop_eigen": ForwardBackend.ADMM_JAX_LOOP_EIGEN,
    "admm_fused_pcg": ForwardBackend.ADMM_FUSED_PCG,
    "admm_fused_cudss": ForwardBackend.ADMM_FUSED_CUDSS,
}


_NAME_TO_BACKWARD_BACKEND: dict[str, BackwardBackend] = {
    "admm_jax_loop_pcg": BackwardBackend.ADMM_JAX_LOOP_PCG,
    "admm_jax_loop_pcg_ffi": BackwardBackend.ADMM_JAX_LOOP_PCG_FFI,
    "admm_jax_loop_cudss_ffi": BackwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
    "admm_jax_loop_jax_dense": BackwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
    "admm_jax_loop_eigen": BackwardBackend.ADMM_JAX_LOOP_EIGEN,
    "admm_fused_pcg": BackwardBackend.ADMM_FUSED_PCG,
    "admm_fused_cudss": BackwardBackend.ADMM_FUSED_CUDSS,
    "direct_jax_dense": BackwardBackend.DIRECT_JAX_DENSE,
    "direct_cudss_ffi": BackwardBackend.DIRECT_CUDSS_FFI,
}

FORWARD_BACKEND_CHOICES: tuple[str, ...] = tuple(sorted(_NAME_TO_FORWARD_BACKEND))
BACKWARD_BACKEND_CHOICES: tuple[str, ...] = tuple(sorted(_NAME_TO_BACKWARD_BACKEND))
CONVERGENCE_CRITERION_CHOICES: tuple[str, ...] = ("first_order", "step")


def parse_forward_backend(value: Union[str, ForwardBackend]) -> ForwardBackend:
    if isinstance(value, ForwardBackend):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Expected str or ForwardBackend, got {type(value)}")
    try:
        return _NAME_TO_FORWARD_BACKEND[value.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown forward backend: {value!r}. Options:"
            f" {sorted(_NAME_TO_FORWARD_BACKEND)}"
        ) from exc


def parse_backward_backend(value: Union[str, BackwardBackend]) -> BackwardBackend:
    if isinstance(value, BackwardBackend):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Expected str or BackwardBackend, got {type(value)}")
    try:
        return _NAME_TO_BACKWARD_BACKEND[value.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown backward backend: {value!r}. Options:"
            f" {sorted(_NAME_TO_BACKWARD_BACKEND)}"
        ) from exc


def parse_convergence_criterion(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected str convergence criterion, got {type(value)}")
    criterion = value.lower()
    if criterion not in CONVERGENCE_CRITERION_CHOICES:
        raise ValueError(
            f"Unknown convergence criterion: {value!r}. Options:"
            f" {CONVERGENCE_CRITERION_CHOICES}"
        )
    return criterion


class KKTState(NamedTuple):
    """KKT quantities for differentiability."""

    y_f: jnp.ndarray  # (N+1, nx)
    y_ineq: jnp.ndarray  # (N+1, m)
    ineq_active_lower_idx: jnp.ndarray  # (N+1, m) bool
    ineq_active_upper_idx: jnp.ndarray  # (N+1, m) bool


class SQPStats(NamedTuple):
    """SQP statistics."""

    status: jnp.ndarray
    num_iter: jnp.ndarray
    qp_stats: ADMMStats  # Each leaf has a leading SQP-iteration dimension.
    convergence_errors: jnp.ndarray
    equality_violations: jnp.ndarray
    inequality_violations: jnp.ndarray
    line_search_step_sizes: Optional[jnp.ndarray] = None

    @property
    def _last_index(self) -> jnp.ndarray:
        return jnp.maximum(self.num_iter - 1, 0)

    @property
    def last_qp(self) -> ADMMStats:
        return jax.tree.map(lambda value: value[self._last_index], self.qp_stats)

    @property
    def convergence_error(self) -> jnp.ndarray:
        return self.convergence_errors[self._last_index]

    @property
    def equality_violation(self) -> jnp.ndarray:
        return self.equality_violations[self._last_index]

    @property
    def inequality_violation(self) -> jnp.ndarray:
        return self.inequality_violations[self._last_index]


def _should_stop_after_admm(status: jnp.ndarray) -> jnp.ndarray:
    return status == jnp.int32(ADMMStatus.NONFINITE)


def _sqp_iteration_status(
    outer_finite: jnp.ndarray, qp_status: jnp.ndarray
) -> jnp.ndarray:
    return jnp.where(
        ~outer_finite,
        jnp.int32(SQPStatus.NONFINITE),
        jnp.where(
            _should_stop_after_admm(qp_status),
            jnp.int32(SQPStatus.QP_FAILED),
            jnp.int32(SQPStatus.NOT_RUN),
        ),
    )


def _terminal_sqp_status(
    status: jnp.ndarray,
    qp_status: jnp.ndarray,
    convergence_error: jnp.ndarray,
    tolerance: float,
    max_iterations: int,
) -> jnp.ndarray:
    qp_infeasible = (qp_status == jnp.int32(ADMMStatus.PRIMAL_INFEASIBLE)) | (
        qp_status == jnp.int32(ADMMStatus.DUAL_INFEASIBLE)
    )
    # Preserve an outer failure, then reject an infeasible final QP.
    solved = (max_iterations > 1) & (convergence_error <= tolerance)
    return jnp.where(
        status != jnp.int32(SQPStatus.NOT_RUN),
        status,
        jnp.where(
            qp_infeasible,
            jnp.int32(SQPStatus.QP_FAILED),
            jnp.where(
                solved,
                jnp.int32(SQPStatus.SOLVED),
                jnp.int32(SQPStatus.MAX_ITER),
            ),
        ),
    )


class TurboMPCConvergenceCheck(NamedTuple):
    qp_data: QPData
    convergence_error: jnp.ndarray
    eq_violation: jnp.ndarray
    ineq_violation: jnp.ndarray


class TurboMPCSolution(NamedTuple):
    """TurboMPC primal solution, solver state, and convergence information."""

    states: jnp.ndarray  # (N+1, nx)
    controls: jnp.ndarray  # (N+1, nu)
    slack: jnp.ndarray  # (N+1, m)
    stats: SQPStats
    admm_state: Optional[ADMMState] = None
    dual_backward_guess: Optional[jnp.ndarray] = None
    kkt_state: Optional[KKTState] = None

    @property
    def status(self):
        return self.stats.status

    @property
    def num_iter(self):
        return self.stats.num_iter

    @property
    def convergence_error(self):
        return self.stats.convergence_error

    @property
    def admm_iters(self):
        return self.stats.qp_stats.num_iter

    @property
    def linesearch_alphas(self):
        return self.stats.line_search_step_sizes

    @property
    def admm_stats(self):
        return self.stats.last_qp


class _BackwardSolveResult(NamedTuple):
    dx_states: jnp.ndarray
    dx_controls: jnp.ndarray
    y_eq_lin: jnp.ndarray
    y_ineq_lin: jnp.ndarray
    dL_dgamma: jnp.ndarray
    dL_dx_init: jnp.ndarray
    dL_du_init: Optional[jnp.ndarray]


class _MixedDerivatives(NamedTuple):
    f_dx_theta: jnp.ndarray
    f_du_theta: jnp.ndarray
    g_dx_theta_weighted: jnp.ndarray
    g_du_theta_weighted: jnp.ndarray
    h_dx_theta_weighted: jnp.ndarray
    h_du_theta_weighted: jnp.ndarray
    g_theta: jnp.ndarray
    h_theta: jnp.ndarray


class TurboMPCSolver:
    """SQP solver that uses ADMM for the QP subproblem."""

    def __init__(
        self,
        program: OptimalControlProblem,
        params: Optional[Dict[str, Any]] = None,
        name: str = "TurboMPCSolver",
        *,
        forward_backend: ForwardBackend = ForwardBackend.ADMM_FUSED_CUDSS,
        backward_backend: BackwardBackend = BackwardBackend.DIRECT_CUDSS_FFI,
        forward_hessian: HessianApproximation = HessianApproximation.GAUSS_NEWTON,
        backward_hessian: HessianApproximation = HessianApproximation.FULL_HESSIAN,
    ):
        self._program = program
        self._name = name
        self._params = DEFAULT_SOLVER_PARAMS if params is None else params
        self._verbose = self._normalize_verbose(self._params.get("verbose", False))
        self._constraint_handling = _validate_constraint_settings(
            self._params,
            program.use_slack_variables,
            program.slack_penalization_weight,
        )
        self._log_barrier_kappa = float(self._params["log_barrier_kappa"])

        self._forward_hessian = HessianApproximation(forward_hessian)
        self._backward_hessian = HessianApproximation(backward_hessian)
        if self._backward_hessian == HessianApproximation.FULL_HESSIAN_PROJECTED:
            raise ValueError(
                "FULL_HESSIAN_PROJECTED is only supported for forward_hessian."
            )
        forward_hessian_min_eig = float(
            self.params.get("forward_hessian_min_eig", 1e-12)
        )
        if forward_hessian_min_eig < 0.0:
            raise ValueError(
                "params['forward_hessian_min_eig'] must be nonnegative, got "
                f"{forward_hessian_min_eig}."
            )

        self._forward_backend = parse_forward_backend(forward_backend)
        self._backward_backend = parse_backward_backend(backward_backend)
        self._convergence_criterion = parse_convergence_criterion(
            self.params.get("convergence_criterion", "first_order")
        )
        self._linesearch_method = parse_linesearch_method(
            self.params["linesearch"]["method"]
        )

        self._forward_admm_backend, self._forward_schur_backend = (
            _FORWARD_BACKEND_COMPONENTS[self._forward_backend]
        )
        self._constraint_handler = _CONSTRAINT_HANDLERS[self._constraint_handling](
            program,
            self.params,
            linesearch_method=self._linesearch_method,
            convergence_criterion=self._convergence_criterion,
        )

        if self._backward_backend in {
            BackwardBackend.DIRECT_JAX_DENSE,
            BackwardBackend.DIRECT_CUDSS_FFI,
        }:
            self._backward_admm_backend = AdmmBackend.JAX_LOOP
            self._backward_schur_backend = SchurSolverBackend.IGNORED
            if (
                self._backward_backend == BackwardBackend.DIRECT_CUDSS_FFI
                and not _HAS_CUDSS_FFI
            ):
                raise ValueError(
                    "BackwardBackend.DIRECT_CUDSS_FFI selected, but cuDSS backward FFI"
                    " is not available."
                )
        else:
            self._backward_admm_backend, self._backward_schur_backend = (
                _BACKWARD_ADMM_COMPONENTS[self._backward_backend]
            )

        if not isinstance(program, OptimalControlProblem):
            raise NotImplementedError(str(program.name) + " is not supported.")

        self._zshape = ZShape(
            horizon=self.program.horizon,
            num_states=self.program.num_state_variables,
            num_controls=self.program.num_control_variables,
        )
        self._cudss_backward_solver = (
            CudssBackwardKktSolver()
            if self._backward_backend == BackwardBackend.DIRECT_CUDSS_FFI
            else None
        )
        self._schur_solver_fwd = (
            make_schur_solver(
                self._forward_schur_backend,
                self.program.horizon,
                self.program.num_state_variables,
                self.program.num_control_variables,
                pcg_params=self.params["admm"]["pcg"],
            )
            if self._forward_admm_backend == AdmmBackend.JAX_LOOP
            else None
        )
        self._schur_solver_bwd = (
            make_schur_solver(
                self._backward_schur_backend,
                self.program.horizon,
                self.program.num_state_variables,
                self.program.num_control_variables,
                pcg_params=self.params["admm"]["pcg"],
            )
            if (
                self._backward_backend
                not in {
                    BackwardBackend.DIRECT_JAX_DENSE,
                    BackwardBackend.DIRECT_CUDSS_FFI,
                }
                and self._backward_admm_backend == AdmmBackend.JAX_LOOP
            )
            else None
        )

        admm_params = self.params["admm"]
        self._admm_solver_fwd = ADMMSolver(
            zshape=self._zshape,
            schur_solver=self._schur_solver_fwd,
            pcg_params=admm_params["pcg"],
            sigma=admm_params["sigma"],
            max_iter=admm_params["max_iter"],
            eps_abs=admm_params.get("eps_abs", 1.0e-4),
            eps_rel=admm_params.get("eps_rel", 1.0e-3),
            rho_min=admm_params.get("rho_min", 1.0e-6),
            rho_max=admm_params.get("rho_max", 1.0e6),
            check_termination_every=admm_params.get("check_termination_every", 1),
            check_infeasibility_every=admm_params.get("check_infeasibility_every", 25),
            adapt_rho_every=admm_params.get("adapt_rho_every", 5),
            adaptive_rho_tolerance=admm_params.get("adaptive_rho_tolerance", 5.0),
            rho_update_max_factor=admm_params.get("rho_update_max_factor", 1.0e6),
            rho_f_factor=admm_params.get(
                "rho_f_factor", admm_params.get("active_constraint_rho_factor", 1000.0)
            ),
            admm_backend=self._forward_admm_backend,
            use_slack_variables=self.program.use_slack_variables,
            eps_prim_inf=admm_params.get("eps_prim_inf", 1.0e-4),
            eps_dual_inf=admm_params.get("eps_dual_inf", 1.0e-4),
            constraint_handling=self._constraint_handling,
            log_barrier_kappa=self._log_barrier_kappa,
        )
        self._admm_solver_bwd = (
            ADMMSolver(
                zshape=self._zshape,
                schur_solver=self._schur_solver_bwd,
                pcg_params=admm_params["pcg"],
                sigma=admm_params["sigma"],
                max_iter=admm_params["max_iter"],
                eps_abs=admm_params.get("eps_abs", 1.0e-6),
                eps_rel=admm_params.get("eps_rel", 1.0e-6),
                rho_min=admm_params.get("rho_min", 1.0e-6),
                rho_max=admm_params.get("rho_max", 1.0e6),
                check_termination_every=admm_params.get("check_termination_every", 1),
                check_infeasibility_every=admm_params.get(
                    "check_infeasibility_every", 25
                ),
                adapt_rho_every=admm_params.get("adapt_rho_every", 5),
                adaptive_rho_tolerance=admm_params.get("adaptive_rho_tolerance", 5.0),
                rho_update_max_factor=admm_params.get("rho_update_max_factor", 1.0e6),
                rho_f_factor=1.0,
                admm_backend=self._backward_admm_backend,
                use_slack_variables=self.program.use_slack_variables,
                eps_prim_inf=admm_params.get("eps_prim_inf", 1.0e-4),
                eps_dual_inf=admm_params.get("eps_dual_inf", 1.0e-4),
            )
            if self._backward_backend
            not in {BackwardBackend.DIRECT_JAX_DENSE, BackwardBackend.DIRECT_CUDSS_FFI}
            else None
        )
        if self._verbose_enabled():
            self._print_init_summary()
        self.solve = self.get_differentiable_solve_function()

    def _normalize_verbose(self, value: Any) -> bool:
        if isinstance(value, bool):
            level = int(value)
        elif isinstance(value, Integral):
            level = int(value)
        else:
            raise ValueError(
                f"verbose must be a bool or integer in {{0, 1}}, got {value!r}."
            )
        if level not in {0, 1}:
            raise ValueError(
                f"verbose must be a bool or integer in {{0, 1}}, got {value!r}."
            )
        return bool(level)

    def _verbose_enabled(self) -> bool:
        return self._verbose

    def _print_init_summary(self) -> None:
        admm_params = self.params["admm"]
        print(
            "[turbompc] init "
            f"name={self.name} "
            f"horizon={self.program.horizon} "
            f"nx={self.program.num_state_variables} "
            f"nu={self.program.num_control_variables} "
            f"nineq={self.program.num_inequality_constraints} "
            f"slack={self.program.use_slack_variables}"
        )
        print(
            "[turbompc] backends "
            f"forward={self.forward_backend.name} "
            f"backward={self.backward_backend.name}"
        )
        print(
            "[turbompc] sqp "
            f"max_iter={self.params['num_sqp_iteration_max']} "
            f"tol={self.params['tol_convergence']:.3e} "
            f"criterion={self._convergence_criterion} "
            f"linesearch={self.linesearch_method.name.lower()}"
        )
        print(
            "[turbompc] admm "
            f"backend={self.forward_admm_backend.name} "
            f"max_iter={admm_params['max_iter']} "
            f"rho={admm_params['rho']:.3e} "
            f"sigma={admm_params['sigma']:.3e} "
            f"eps_abs={admm_params.get('eps_abs', 1.0e-4):.3e} "
            f"eps_rel={admm_params.get('eps_rel', 1.0e-3):.3e} "
            f"check_termination_every={admm_params.get('check_termination_every', 1)} "
            "check_infeasibility_every="
            f"{admm_params.get('check_infeasibility_every', 25)} "
            f"adapt_rho_every={admm_params.get('adapt_rho_every', 5)}"
        )

    def _debug_print_sqp_iteration(
        self,
        debug_info: Dict[str, Union[float, jnp.ndarray]],
    ) -> None:
        if not self._verbose_enabled():
            return

        value_formats = {
            "it": "{it:02d}",
            "admm": "{admm:03d}",
            "alpha": "{alpha:.3f}",
        }
        fields = []
        for key in debug_info:
            value_format = value_formats.get(key)
            if value_format is None:
                value_format = "{" + key + ":.3e}"
            fields.append(f"{key}={value_format}")
        jax.debug.print("[turbompc] sqp " + " ".join(fields), **debug_info)

    def _debug_print_solve_summary(
        self,
        *,
        status: int,
        num_iter: jnp.ndarray,
        convergence_error: jnp.ndarray,
        admm_iters: jnp.ndarray,
    ) -> None:
        if not self._verbose_enabled():
            return

        total_admm = jnp.sum(admm_iters)
        jax.debug.print(
            (
                "[turbompc] done status={status} sqp={sqp} conv={conv:.3e} "
                "total_admm={total_admm}"
            ),
            status=status,
            sqp=num_iter,
            conv=convergence_error,
            total_admm=total_admm,
        )

    @property
    def program(self) -> OptimalControlProblem:
        return self._program

    @property
    def params(self) -> Dict:
        return self._params

    @property
    def name(self) -> str:
        return self._name

    @property
    def linesearch_method(self) -> LinesearchMethod:
        return self._linesearch_method

    @property
    def forward_admm_backend(self) -> AdmmBackend:
        return self._forward_admm_backend

    @property
    def forward_backend(self) -> ForwardBackend:
        return self._forward_backend

    @property
    def backward_admm_backend(self) -> AdmmBackend:
        return self._backward_admm_backend

    @property
    def backward_backend(self) -> BackwardBackend:
        return self._backward_backend

    @property
    def forward_schur_backend(self) -> SchurSolverBackend:
        return self._forward_schur_backend

    @property
    def backward_schur_backend(self) -> SchurSolverBackend:
        return self._backward_schur_backend

    def initial_guess(
        self, params: Optional[Dict[str, Any]] = None
    ) -> TurboMPCSolution:
        """Returns an SQP-ADMM initial guess."""
        if params is None:
            params = self.program.params
        states, controls = self.program.initial_guess(params)
        max_iter = self.params["num_sqp_iteration_max"]
        linesearch_alphas = None
        if self.linesearch_method != LinesearchMethod.NONE:
            linesearch_alphas = jnp.zeros((max_iter,), dtype=states.dtype)
        stats = SQPStats(
            status=jnp.int32(SQPStatus.NOT_RUN),
            num_iter=jnp.asarray(0, dtype=jnp.int32),
            qp_stats=empty_admm_stats(states.dtype, (max_iter,)),
            convergence_errors=jnp.zeros(max_iter, dtype=states.dtype),
            equality_violations=jnp.zeros(max_iter, dtype=states.dtype),
            inequality_violations=jnp.zeros(max_iter, dtype=states.dtype),
            line_search_step_sizes=linesearch_alphas,
        )
        return TurboMPCSolution(
            states=states,
            controls=controls,
            slack=jnp.zeros(
                (states.shape[0], self.program.num_inequality_constraints),
                dtype=states.dtype,
            ),
            stats=stats,
            admm_state=None,
            dual_backward_guess=None,
            kkt_state=None,
        )

    def make_params_with_weights(self, weights, problem_params=None):
        def _deep_merge_dict(dst, src):
            for key, value in src.items():
                if isinstance(value, dict) and isinstance(dst.get(key), dict):
                    _deep_merge_dict(dst[key], value)
                else:
                    dst[key] = value

        if problem_params is None:
            new_params = copy.deepcopy(self.program.params)
        else:
            new_params = copy.deepcopy(problem_params)
        _deep_merge_dict(new_params, weights)
        normalize_problem_params(new_params)
        return new_params

    def _resolve_slack_penalization_weight(self, weights, problem_params, dtype):
        if "slack_penalization_weight" in weights:
            if not self.program.use_slack_variables:
                raise ValueError(
                    "slack_penalization_weight can only be varied when "
                    "use_slack_variables is true"
                )
            value = weights["slack_penalization_weight"]
        else:
            value = problem_params.get(
                "slack_penalization_weight",
                self.program.slack_penalization_weight,
            )
        value = jnp.asarray(value, dtype=dtype)
        if value.ndim != 0:
            raise ValueError("slack_penalization_weight must be a scalar")
        return value

    @staticmethod
    def _weights_without_slack_penalty(weights):
        return {
            key: value
            for key, value in weights.items()
            if key != "slack_penalization_weight"
        }

    def _pack_admm_eq_multipliers(self, admm_state: ADMMState) -> jnp.ndarray:
        nx = self.program.num_state_variables
        y_f0 = admm_state.y_f_0
        if self.program.constrain_initial_control:
            y_f0_state = y_f0[:nx]
            y_u0 = y_f0[nx:]
            return jnp.concatenate(
                [y_f0_state, admm_state.y_f_dyn.reshape(-1), y_u0], axis=0
            )
        return jnp.concatenate([y_f0, admm_state.y_f_dyn.reshape(-1)], axis=0)

    def _uses_direct_backward(self) -> bool:
        return self._backward_backend in {
            BackwardBackend.DIRECT_JAX_DENSE,
            BackwardBackend.DIRECT_CUDSS_FFI,
        }

    def _build_backward_qp(
        self,
        forward_solution: TurboMPCSolution,
        dl_dstates: jnp.ndarray,
        dl_dcontrols: jnp.ndarray,
        dl_dslack: jnp.ndarray,
        problem_params: Dict[str, Any],
        slack_penalization_weight: jnp.ndarray,
    ) -> BackwardQPContext:
        # Backward adjoint QP: keep primal variables in true coordinates,
        # because cotangents are taken w.r.t. true state/control variables.
        qp_data = self._build_qp(
            forward_solution.states,
            forward_solution.controls,
            forward_solution.slack,
            problem_params,
            forward_solution.admm_state,
            hessian=self._backward_hessian,
            scale_variables=False,
            mask_inactive_constraints=(
                self._constraint_handler.mask_inactive_constraints
            ),
            slack_penalization_weight=slack_penalization_weight,
        )

        eq = qp_data.eq
        eq = QPEqualityBlocks(
            A0=eq.A0,
            A_minus=eq.A_minus,
            A_plus=eq.A_plus,
            c0=jnp.zeros_like(eq.c0),
            c=jnp.zeros_like(eq.c),
        )

        if qp_data.ineq.G.shape[1] > 0:
            return self._constraint_handler.build_backward_qp(
                qp_data,
                eq,
                forward_solution,
                dl_dstates,
                dl_dcontrols,
                dl_dslack,
                problem_params,
                slack_penalization_weight,
            )

        q = -jnp.concatenate([dl_dstates, dl_dcontrols], axis=-1)
        cost = QPCostBlocks(D=qp_data.cost.D, E=qp_data.cost.E, q=q)
        return BackwardQPContext(
            base_qp=qp_data,
            solve_qp=QPData(cost=cost, eq=eq, ineq=qp_data.ineq),
        )

    def _solve_backward_qp_admm(
        self,
        backward_qp: BackwardQPContext,
        forward_solution: TurboMPCSolution,
        dl_dslack: jnp.ndarray,
        problem_params: Dict[str, Any],
    ) -> _BackwardSolveResult:
        """Solve the reduced adjoint QP via ADMM."""
        del problem_params
        admm_params = self.params["admm"]
        rho_active = admm_params["rho"] * admm_params.get(
            "rho_f_factor", admm_params.get("active_constraint_rho_factor", 1000.0)
        )
        admm_solver = self._admm_solver_bwd
        if admm_solver is None:
            raise RuntimeError(
                "Backward ADMM solver is not initialized for the selected backward"
                " backend."
            )
        admm_state0 = admm_solver.initial_state(
            qp_data=backward_qp.solve_qp,
            rho_bar=rho_active,
        )
        admm_result = admm_solver.solve(
            qp_data=backward_qp.solve_qp,
            admm_state0=admm_state0,
            rho_bar=rho_active,
        )
        admm_state = admm_result.state
        dx_states, dx_controls = unpack_x(admm_state.x_blocks, self._zshape)

        y_eq_lin = self._pack_admm_eq_multipliers(admm_state)
        recovered = self._constraint_handler.recover_ineq_gradient(
            backward_qp,
            forward_solution,
            dx_states,
            dx_controls,
            dl_dslack,
        )
        if recovered is not None:
            y_ineq_lin, dL_dgamma = recovered
        else:
            y_ineq_lin = admm_state.y_g.reshape(-1)
            dL_dgamma = jnp.array(0.0, dtype=dl_dslack.dtype)

        nx = self.program.num_state_variables
        y_f_0_lin = admm_state.y_f_0
        dL_dx_init = y_f_0_lin[:nx]
        if self.program.constrain_initial_control:
            dL_du_init = y_f_0_lin[nx:]
        else:
            dL_du_init = None
        return _BackwardSolveResult(
            dx_states=dx_states,
            dx_controls=dx_controls,
            y_eq_lin=y_eq_lin,
            y_ineq_lin=y_ineq_lin,
            dL_dgamma=dL_dgamma,
            dL_dx_init=dL_dx_init,
            dL_du_init=dL_du_init,
        )

    def _solve_backward_qp_direct(
        self,
        backward_qp: BackwardQPContext,
        forward_solution: TurboMPCSolution,
        dl_dslack: jnp.ndarray,
    ) -> _BackwardSolveResult:
        """Solve the reduced adjoint system via a direct KKT solve."""
        if self._backward_backend == BackwardBackend.DIRECT_CUDSS_FFI:
            (dx_states, dx_controls), multipliers = self._cudss_backward_solver.solve(
                qp_data=backward_qp.solve_qp,
                zshape=self._zshape,
            )
        else:
            (dx_states, dx_controls), multipliers = solve_backward_kkt(
                qp_data=backward_qp.solve_qp, zshape=self._zshape
            )

        solve_qp = backward_qp.solve_qp
        nx = self.program.num_state_variables
        m0_back = solve_qp.eq.A0.shape[0]
        N_back = solve_qp.eq.A_minus.shape[0]
        m_eq_dyn = solve_qp.eq.A_minus.shape[1]
        m_eq_total = m0_back + N_back * m_eq_dyn

        multipliers_eq_flat = (
            multipliers[..., :m_eq_total]
            if m_eq_total > 0
            else jnp.zeros(multipliers.shape[:-1] + (0,), dtype=multipliers.dtype)
        )
        y_f_0 = multipliers_eq_flat[..., :m0_back]
        y_f_dyn_flat = multipliers_eq_flat[..., m0_back : m0_back + N_back * m_eq_dyn]
        if self.program.constrain_initial_control:
            y_f0_state = y_f_0[..., :nx]
            y_u0 = y_f_0[..., nx:]
            y_eq_lin = jnp.concatenate([y_f0_state, y_f_dyn_flat, y_u0], axis=-1)
        else:
            y_eq_lin = jnp.concatenate([y_f_0, y_f_dyn_flat], axis=-1)

        recovered = self._constraint_handler.recover_ineq_gradient(
            backward_qp,
            forward_solution,
            dx_states,
            dx_controls,
            dl_dslack,
        )
        if recovered is not None:
            y_ineq_lin, dL_dgamma = recovered
        else:
            m_ineq_total = solve_qp.ineq.G.shape[0] * solve_qp.ineq.G.shape[1]
            multipliers_ineq_flat = (
                multipliers[..., m_eq_total : m_eq_total + m_ineq_total]
                if m_ineq_total > 0
                else jnp.zeros(multipliers.shape[:-1] + (0,), dtype=multipliers.dtype)
            )
            # Direct KKT solvers return one multiplier per inequality row in solve_qp.
            y_ineq_lin = multipliers_ineq_flat
            dL_dgamma = jnp.array(0.0, dtype=dl_dslack.dtype)

        y_f_0_lin = multipliers_eq_flat[..., :m0_back]
        dL_dx_init = y_f_0_lin[..., :nx]
        if self.program.constrain_initial_control:
            dL_du_init = y_f_0_lin[..., nx:]
        else:
            dL_du_init = None
        return _BackwardSolveResult(
            dx_states=dx_states,
            dx_controls=dx_controls,
            y_eq_lin=y_eq_lin,
            y_ineq_lin=y_ineq_lin,
            dL_dgamma=dL_dgamma,
            dL_dx_init=dL_dx_init,
            dL_du_init=dL_du_init,
        )

    def _compute_mixed_derivatives(
        self,
        problem_params,
        weights,
        states,
        controls,
        y_eq,
        y_ineq,
        lower_bound_weight,
        upper_bound_weight,
    ) -> _MixedDerivatives:
        def _cost_gradient(states, controls, params):
            def cost(states, controls):
                return self.program.cost(states, controls, params)

            return jax.grad(cost, argnums=(0, 1))(states, controls)

        def _eq_constraints(states, controls, params):
            return self.program.equality_constraints(states, controls, params)

        def _inequality_residual(states, controls, params):
            g, l, u = self.program.inequality_constraints(states, controls, params)
            g = g.reshape((-1,))
            l = l.reshape((-1,))
            u = u.reshape((-1,))
            lower_weight = lower_bound_weight.reshape((-1,)).astype(states.dtype)
            upper_weight = upper_bound_weight.reshape((-1,)).astype(states.dtype)
            return self._constraint_handler.inequality_residual(
                g, l, u, lower_weight, upper_weight
            )

        def _constraints_weighted(states, controls, params, lambdas, func):
            return lambdas.flatten() @ func(states, controls, params)

        def _constraints_weighted_dz(states, controls, params, lambdas, func):
            return jax.jacfwd(
                lambda x, u: _constraints_weighted(x, u, params, lambdas, func),
                argnums=(0, 1),
            )(states, controls)

        def _bundle(w):
            params = self.make_params_with_weights(w, problem_params)
            f_dx, f_du = _cost_gradient(states, controls, params)
            g_val = _eq_constraints(states, controls, params)
            h_val = _inequality_residual(states, controls, params)
            g_dx, g_du = _constraints_weighted_dz(
                states, controls, params, y_eq, _eq_constraints
            )
            h_dx, h_du = _constraints_weighted_dz(
                states, controls, params, y_ineq, _inequality_residual
            )
            return f_dx, f_du, g_val, h_val, g_dx, g_du, h_dx, h_du

        (
            f_dx_theta,
            f_du_theta,
            g_theta,
            h_theta,
            g_dx_theta_weighted,
            g_du_theta_weighted,
            h_dx_theta_weighted,
            h_du_theta_weighted,
        ) = jax.jacfwd(_bundle)(weights)
        flat_weights, _ = jax.tree_util.tree_flatten(weights)
        return _MixedDerivatives(
            f_dx_theta=self._flatten_theta_tree(f_dx_theta, flat_weights),
            f_du_theta=self._flatten_theta_tree(f_du_theta, flat_weights),
            g_dx_theta_weighted=self._flatten_theta_tree(
                g_dx_theta_weighted, flat_weights
            ),
            g_du_theta_weighted=self._flatten_theta_tree(
                g_du_theta_weighted, flat_weights
            ),
            h_dx_theta_weighted=self._flatten_theta_tree(
                h_dx_theta_weighted, flat_weights
            ),
            h_du_theta_weighted=self._flatten_theta_tree(
                h_du_theta_weighted, flat_weights
            ),
            g_theta=self._flatten_theta_tree(g_theta, flat_weights),
            h_theta=self._flatten_theta_tree(h_theta, flat_weights),
        )

    def _flatten_theta_tree(self, deriv_tree, flat_weights) -> jnp.ndarray:
        flat_derivs, _ = jax.tree_util.tree_flatten(deriv_tree)
        return jnp.concatenate(
            [
                self._flatten_theta_leaf(deriv, weight)
                for deriv, weight in zip(flat_derivs, flat_weights)
            ],
            axis=-1,
        )

    def _flatten_theta_leaf(self, deriv_leaf, weight_leaf) -> jnp.ndarray:
        weight_leaf = jnp.asarray(weight_leaf)
        if weight_leaf.ndim == 0:
            return deriv_leaf[..., jnp.newaxis]
        out_ndim = deriv_leaf.ndim - weight_leaf.ndim
        out_shape = deriv_leaf.shape[:out_ndim]
        return jnp.reshape(deriv_leaf, out_shape + (weight_leaf.size,))

    def _forward_multipliers(
        self, forward_solution: TurboMPCSolution
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        kkt_state = forward_solution.kkt_state
        if forward_solution.admm_state is not None:
            y_eq = self._pack_admm_eq_multipliers(forward_solution.admm_state)
            y_ineq = forward_solution.admm_state.y_g
        else:
            y_eq = kkt_state.y_f.reshape(-1)
            y_ineq = kkt_state.y_ineq
        return y_eq, self._constraint_handler.residual_multipliers(kkt_state, y_ineq)

    def _compute_weight_gradient(
        self,
        forward_solution: TurboMPCSolution,
        problem_params: Dict[str, Any],
        weights: Any,
        backward_result: _BackwardSolveResult,
    ) -> jnp.ndarray:
        weights = self._weights_without_slack_penalty(weights)
        if not jax.tree_util.tree_leaves(weights):
            return jnp.zeros((0,), dtype=forward_solution.states.dtype)

        y_eq_star, y_ineq_star = self._forward_multipliers(forward_solution)
        lower_weight, upper_weight = self._constraint_handler.bound_weights(
            forward_solution, problem_params
        )
        mixed = self._compute_mixed_derivatives(
            problem_params,
            weights,
            forward_solution.states,
            forward_solution.controls,
            y_eq_star,
            y_ineq_star,
            lower_weight,
            upper_weight,
        )
        return self._assemble_parameter_gradient(mixed, backward_result)

    def _assemble_parameter_gradient(
        self,
        mixed: _MixedDerivatives,
        backward_result: _BackwardSolveResult,
    ) -> jnp.ndarray:
        return -(
            jnp.sum(
                jnp.moveaxis(
                    mixed.f_dx_theta
                    + mixed.g_dx_theta_weighted
                    + mixed.h_dx_theta_weighted,
                    -1,
                    0,
                )
                * backward_result.dx_states,
                axis=(-2, -1),
            )
            + jnp.sum(
                jnp.moveaxis(
                    mixed.f_du_theta
                    + mixed.g_du_theta_weighted
                    + mixed.h_du_theta_weighted,
                    -1,
                    0,
                )
                * backward_result.dx_controls,
                axis=(-2, -1),
            )
            + jnp.sum(
                jnp.moveaxis(mixed.g_theta, -1, 0) * backward_result.y_eq_lin,
                axis=(-1),
            )
            + jnp.sum(
                jnp.moveaxis(mixed.h_theta, -1, 0) * backward_result.y_ineq_lin,
                axis=(-1),
            )
        )

    def _unflatten_weight_gradient(
        self,
        weights: Any,
        dL_dtheta_flat: jnp.ndarray,
        dL_dgamma: jnp.ndarray,
    ) -> Any:
        weights_without_slack_penalty = self._weights_without_slack_penalty(weights)
        flat_weights, tree_def = jax.tree_util.tree_flatten(
            weights_without_slack_penalty
        )
        flat_weights = [jnp.asarray(arr) for arr in flat_weights]
        start_idx = 0
        dL_dtheta_chunks = []
        for weight_leaf in flat_weights:
            length = weight_leaf.size
            chunk = dL_dtheta_flat[start_idx : start_idx + length]
            # Scalars must be returned as 0-d arrays, not shape (1,).
            dL_dtheta_chunks.append(jnp.reshape(chunk, weight_leaf.shape))
            start_idx += length
        dL_dtheta_unflattened = jax.tree_util.tree_unflatten(tree_def, dL_dtheta_chunks)
        if "slack_penalization_weight" in weights:
            dL_dtheta_unflattened["slack_penalization_weight"] = dL_dgamma
        return dL_dtheta_unflattened

    def _build_problem_params_cotangent(
        self,
        problem_params_orig: Dict[str, Any],
        problem_params: Dict[str, Any],
        backward_result: _BackwardSolveResult,
    ) -> Dict[str, Any]:
        def _zero_cotangent(x):
            if hasattr(x, "shape") and hasattr(x, "dtype"):
                return jnp.zeros_like(x)
            if isinstance(x, float):
                return jnp.asarray(0.0, dtype=jnp.float64)
            return None

        problem_params_cotangent = jax.tree_util.tree_map(
            _zero_cotangent, problem_params_orig
        )
        init_state_cotangent = backward_result.dL_dx_init
        if self.program.rescale_optimization_variables:
            _, _, _, _, state_diff, control_diff = self.program._get_rescaling_params(
                problem_params
            )
            init_state_cotangent = init_state_cotangent / state_diff
        problem_params_cotangent["initial_state"] = init_state_cotangent
        if backward_result.dL_du_init is not None:
            init_control_cotangent = backward_result.dL_du_init
            if self.program.rescale_optimization_variables:
                init_control_cotangent = init_control_cotangent / control_diff
            problem_params_cotangent["initial_control"] = init_control_cotangent
        return problem_params_cotangent

    def get_differentiable_solve_function(self):
        """Create a differentiable solve function with captured solver.

        This wraps the TurboMPC solve in a JAX custom_vjp.
        """

        def _solve_impl_entry(
            initial_guess: TurboMPCSolution,
            problem_params: Dict[str, Any],
            weights: Any = {},
        ) -> TurboMPCSolution:
            slack_penalization_weight = self._resolve_slack_penalization_weight(
                weights, problem_params, initial_guess.states.dtype
            )
            problem_params = self.make_params_with_weights(weights, problem_params)
            return self._solve_impl(
                initial_guess.states,
                initial_guess.controls,
                problem_params,
                initial_guess.admm_state,
                slack_penalization_weight,
            )

        @partial(jax.custom_vjp)
        def solve(
            initial_guess: TurboMPCSolution,
            problem_params: Dict[str, Any],
            weights: Any = {},
        ) -> TurboMPCSolution:
            return _solve_impl_entry(initial_guess, problem_params, weights)

        def _solve_fwd(initial_guess, problem_params, weights):
            solution = _solve_impl_entry(initial_guess, problem_params, weights)
            residual_for_backward_pass = (solution, weights, problem_params)
            return solution, residual_for_backward_pass

        def _solve_bwd(residual_for_backward_pass, cotangents: TurboMPCSolution):
            """VJP backward pass with captured self."""
            solution, weights, problem_params_orig = residual_for_backward_pass
            slack_penalization_weight = self._resolve_slack_penalization_weight(
                weights, problem_params_orig, solution.states.dtype
            )
            problem_params = self.make_params_with_weights(weights, problem_params_orig)
            solution = self._constraint_handler._prepare_backward_solution(
                solution, problem_params
            )
            backward_qp = self._build_backward_qp(
                solution,
                cotangents.states,
                cotangents.controls,
                cotangents.slack,
                problem_params,
                slack_penalization_weight,
            )
            if self._uses_direct_backward():
                backward_result = self._solve_backward_qp_direct(
                    backward_qp,
                    solution,
                    cotangents.slack,
                )
            else:
                backward_result = self._solve_backward_qp_admm(
                    backward_qp,
                    solution,
                    cotangents.slack,
                    problem_params,
                )

            dL_dtheta_flat = self._compute_weight_gradient(
                solution,
                problem_params,
                weights,
                backward_result,
            )
            dL_dtheta_unflattened = self._unflatten_weight_gradient(
                weights,
                dL_dtheta_flat,
                backward_result.dL_dgamma,
            )
            problem_params_cotangent = self._build_problem_params_cotangent(
                problem_params_orig,
                problem_params,
                backward_result,
            )
            return (None, problem_params_cotangent, dL_dtheta_unflattened)

        solve.defvjp(_solve_fwd, _solve_bwd)
        return solve

    def _build_qp(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        problem_params: Dict[str, Any],
        admm_state: Optional[ADMMState],
        *,
        hessian: HessianApproximation,
        scale_variables: bool,
        mask_inactive_constraints: bool = False,
        slack_penalization_weight: Optional[jnp.ndarray] = None,
    ) -> QPData:
        """Convexify the nonlinear optimization problem.

        The returned problem data is rescaled if ``scale_variables=True``.
        Inequality multipliers are restricted to the active set if
        ``mask_inactive_constraints=True``.
        """
        if slack_penalization_weight is None:
            slack_penalization_weight = jnp.asarray(
                problem_params.get(
                    "slack_penalization_weight",
                    self.program.slack_penalization_weight,
                ),
                dtype=states.dtype,
            )
        D, E, q = self.program.get_cost_linearized_blocks(
            states, controls, problem_params
        )
        As_next, Bs_next, As, Bs, Cs = self.program.get_dynamics_linearized_matrices(
            states, controls, problem_params
        )
        ineq_blocks, ineq_l, ineq_u = self.program.get_inequalities_linearized_matrices(
            states, controls, problem_params
        )
        A0, c0 = self.program.get_initial_equality_linearized_matrices(
            problem_params, states.dtype
        )
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
            slack_penalization_weight=slack_penalization_weight,
        )

        if hessian != HessianApproximation.GAUSS_NEWTON and admm_state is not None:
            hessian_terms = self.program.get_dynamics_lagrangian_hessian(
                states,
                controls,
                problem_params,
                admm_state.y_f_dyn,
            )
            if self.program.num_inequality_constraints > 0:
                inequality_multipliers = admm_state.y_g
                if mask_inactive_constraints:
                    values = (
                        None
                        if self.program.use_slack_variables
                        else _apply_G(qp_data, pack_x(states, controls))
                    )
                    active_lower, active_upper = self._constraint_handler.active_sets(
                        values,
                        qp_data.ineq.l,
                        qp_data.ineq.u,
                        slacks,
                        admm_state.y_g,
                    )
                    active = (active_lower | active_upper).astype(admm_state.y_g.dtype)
                    inequality_multipliers = active * inequality_multipliers
                hessian_terms = (
                    hessian_terms
                    + self.program.get_inequality_lagrangian_hessian(
                        states,
                        controls,
                        problem_params,
                        inequality_multipliers,
                    )
                )

            # QPs are formulated in absolute variables, not SQP increments.
            # Preserve the objective gradient at the linearization point:
            #   (D + H) z_bar + (q - H z_bar) = D z_bar + q.
            linearization_blocks = pack_x(states, controls)
            qp_data = QPData(
                cost=QPCostBlocks(
                    D=qp_data.cost.D + hessian_terms,
                    E=qp_data.cost.E,
                    q=qp_data.cost.q
                    - jnp.einsum("tij,tj->ti", hessian_terms, linearization_blocks),
                ),
                eq=qp_data.eq,
                ineq=qp_data.ineq,
            )

        if self.program.rescale_optimization_variables and scale_variables:
            _, _, _, _, state_diff, control_diff = self.program._get_rescaling_params(
                problem_params
            )
            qp_data = scale_qp_data(qp_data, state_diff, control_diff)
            states, controls = self.program.scale_states_controls(
                states, controls, problem_params
            )

        if hessian == HessianApproximation.FULL_HESSIAN_PROJECTED:
            qp_data = self._project_qp_hessian(
                qp_data,
                states=states,
                controls=controls,
            )
        return qp_data

    def _skipped_first_order_check(
        self,
        qp_data: QPData,
        convergence_error: jnp.ndarray,
    ) -> TurboMPCConvergenceCheck:
        zero = jnp.zeros_like(convergence_error)
        return TurboMPCConvergenceCheck(
            qp_data=qp_data,
            convergence_error=convergence_error,
            eq_violation=zero,
            ineq_violation=zero,
        )

    def _first_order_check(
        self,
        qp_data: QPData,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        constraint_carry: Any,
        admm_state: ADMMState,
        problem_params: Dict[str, Any],
    ) -> TurboMPCConvergenceCheck:
        conv, eq_violation, ineq_violation = (
            self._constraint_handler.first_order_errors(
                qp_data,
                states,
                controls,
                slacks,
                constraint_carry,
                admm_state,
                problem_params,
            )
        )
        return TurboMPCConvergenceCheck(
            qp_data=qp_data,
            convergence_error=conv,
            eq_violation=eq_violation,
            ineq_violation=ineq_violation,
        )

    def _step_check(
        self,
        qp_data: QPData,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        problem_params: Dict[str, Any],
        convergence_error: jnp.ndarray,
    ) -> TurboMPCConvergenceCheck:
        eq_violation, ineq_violation, *_ = self._evaluate_constraints_with_bounds(
            states, controls, slacks, problem_params
        )
        return TurboMPCConvergenceCheck(
            qp_data=qp_data,
            convergence_error=convergence_error,
            eq_violation=eq_violation,
            ineq_violation=ineq_violation,
        )

    def _step_check_from_current_qp(
        self,
        qp_data: QPData,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        problem_params: Dict[str, Any],
        convergence_error: jnp.ndarray,
    ) -> TurboMPCConvergenceCheck:
        """Evaluate step diagnostics from a QP built at the same iterate."""
        states_qp, controls_qp = self.program.scale_states_controls(
            states, controls, problem_params
        )
        x_blocks = pack_x(states_qp, controls_qp)
        eq_violation, ineq_violation = _compute_qp_constraint_errors(
            qp_data, x_blocks, slacks, self.program.use_slack_variables
        )
        return TurboMPCConvergenceCheck(
            qp_data=qp_data,
            convergence_error=convergence_error,
            eq_violation=eq_violation,
            ineq_violation=ineq_violation,
        )

    def _project_qp_hessian(
        self,
        qp_data: QPData,
        states: Optional[jnp.ndarray] = None,
        controls: Optional[jnp.ndarray] = None,
    ) -> QPData:
        """Project the Hessian while preserving the local objective gradient.

        When ``states`` and ``controls`` are provided, adjust the linear cost after
        projecting ``D`` so the QP objective has the same gradient at the SQP
        linearization point. For each stage, with ``z_bar = [state, control]``,

            D_projected @ z_bar + q_projected = D @ z_bar + q.

        The cross-stage blocks ``E`` are unchanged. If the linearization point is
        omitted, only ``D`` is projected and ``q`` is left unchanged.
        """
        D = qp_data.cost.D
        E = qp_data.cost.E

        if E.shape[0] == 0:
            local_offdiag_bound = jnp.zeros((D.shape[0],), dtype=D.dtype)
        else:
            offdiag_norms = jnp.linalg.norm(E, axis=(-2, -1))
            local_offdiag_bound = jnp.zeros((D.shape[0],), dtype=D.dtype)
            local_offdiag_bound = local_offdiag_bound.at[:-1].add(offdiag_norms)
            local_offdiag_bound = local_offdiag_bound.at[1:].add(offdiag_norms)

        eigvals, eigvecs = jnp.linalg.eigh(D)
        floor = local_offdiag_bound[:, jnp.newaxis] + jnp.asarray(
            self.params["forward_hessian_min_eig"], dtype=D.dtype
        )
        eigvals_projected = jnp.maximum(eigvals, floor)
        D_projected = jnp.matmul(
            eigvecs * eigvals_projected[..., jnp.newaxis, :],
            jnp.swapaxes(eigvecs, -1, -2),
        )
        q_projected = qp_data.cost.q
        if states is not None or controls is not None:
            if states is None or controls is None:
                raise ValueError("states and controls must be provided together.")
            linearization_blocks = pack_x(states, controls)
            projection_delta = D_projected - D
            q_projected = q_projected - jnp.einsum(
                "tij,tj->ti", projection_delta, linearization_blocks
            )
        return QPData(
            cost=QPCostBlocks(D=D_projected, E=E, q=q_projected),
            eq=qp_data.eq,
            ineq=qp_data.ineq,
        )

    def _build_kkt_state(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        admm_state: ADMMState,
        ineq_values: jnp.ndarray,
        ineq_lower: jnp.ndarray,
        ineq_upper: jnp.ndarray,
    ) -> KKTState:
        """Assemble the KKT quantities the backward pass consumes."""
        if admm_state is None:
            y_f = jnp.zeros_like(states)
            y_ineq = jnp.zeros((states.shape[0], 0), dtype=states.dtype)
        else:
            nx = self.program.num_state_variables
            y_f0 = admm_state.y_f_0
            if self.program.constrain_initial_control:
                y_f0 = y_f0[:nx]
            y_f = jnp.concatenate([y_f0[jnp.newaxis], admm_state.y_f_dyn], axis=0)
            y_ineq = admm_state.y_g

        g = ineq_values.reshape((states.shape[0], -1))
        l = ineq_lower.reshape((states.shape[0], -1))
        u = ineq_upper.reshape((states.shape[0], -1))
        active_lower, active_upper = self._constraint_handler.active_sets(
            g,
            l,
            u,
            slacks,
            None if admm_state is None else y_ineq,
        )
        return KKTState(
            y_f=y_f,
            y_ineq=y_ineq,
            ineq_active_lower_idx=active_lower,
            ineq_active_upper_idx=active_upper,
        )

    def _evaluate_constraints_with_bounds(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        problem_params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return evaluate_constraints_with_bounds(
            self.program,
            states,
            controls,
            slacks,
            problem_params,
            norm=jnp.inf,
        )

    def solve(
        self,
        initial_guess: TurboMPCSolution,
        problem_params: Dict[str, Any],
        weights: Any = {},
    ) -> TurboMPCSolution:
        problem_params = jax.lax.stop_gradient(problem_params)
        slack_penalization_weight = self._resolve_slack_penalization_weight(
            weights, problem_params, initial_guess.states.dtype
        )
        problem_params = self.make_params_with_weights(weights, problem_params)

        return self._solve_impl(
            initial_guess.states,
            initial_guess.controls,
            problem_params,
            initial_guess.admm_state,
            slack_penalization_weight,
        )

    def _solve_impl(  # noqa: C901
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        problem_params: Dict[str, Any],
        admm_state0: Optional[ADMMState] = None,
        slack_penalization_weight: Optional[jnp.ndarray] = None,
    ) -> TurboMPCSolution:
        if slack_penalization_weight is None:
            slack_penalization_weight = jnp.asarray(
                problem_params.get(
                    "slack_penalization_weight",
                    self.program.slack_penalization_weight,
                ),
                dtype=states.dtype,
            )
        max_iter = self.params["num_sqp_iteration_max"]
        linesearch_method = self.linesearch_method
        has_linesearch = linesearch_method != LinesearchMethod.NONE
        linesearch_alphas = None
        if has_linesearch:
            linesearch_alphas = jnp.zeros((max_iter,), dtype=states.dtype)

        z_prev = pack_z(states, controls)
        convergence_error = jnp.array(jnp.inf, dtype=states.dtype)

        qp_stats = empty_admm_stats(states.dtype, (max_iter,))
        equality_violations = jnp.zeros(max_iter, dtype=float)
        inequality_violations = jnp.zeros(max_iter, dtype=float)
        convergence_errors = jnp.zeros(max_iter, dtype=float)

        admm_params = self.params["admm"]
        admm_solver = self._admm_solver_fwd
        slacks0 = (
            admm_state0.xi_g
            if admm_state0 is not None
            else jnp.zeros(
                (states.shape[0], self.program.num_inequality_constraints),
                dtype=states.dtype,
            )
        )
        qp_data = self._build_qp(
            states,
            controls,
            slacks0,
            problem_params,
            admm_state0,
            slack_penalization_weight=slack_penalization_weight,
            hessian=self._forward_hessian,
            scale_variables=True,
        )
        if admm_state0 is None:
            admm_state0 = admm_solver.initial_state(
                qp_data=qp_data,
                rho_bar=admm_params["rho"],
                states0=jnp.zeros_like(states),
                controls0=jnp.zeros_like(controls),
            )
        constraint_carry0 = self._constraint_handler.init_carry(
            states, controls, slacks0, problem_params
        )

        class _SQPState(NamedTuple):
            it: jnp.ndarray
            states: jnp.ndarray
            controls: jnp.ndarray
            slacks: jnp.ndarray
            z_prev: jnp.ndarray
            conv: jnp.ndarray
            linesearch_alphas: Optional[jnp.ndarray]
            admm_state: ADMMState
            qp_stats: ADMMStats
            equality_violations: jnp.ndarray
            inequality_violations: jnp.ndarray
            convergence_errors: jnp.ndarray
            qp_data: QPData
            sqp_status: jnp.ndarray

        def body_fun(carry):
            """Body function for SQP iteration."""
            state, constraint_carry = carry
            qp_data = state.qp_data

            # Solve QP via ADMM
            admm_result = admm_solver.solve(
                qp_data=qp_data,
                admm_state0=state.admm_state,
                rho_bar=admm_params["rho"],
                alpha=self.params["admm"].get("relaxation_parameter", 1.0),
            )
            admm_stats = admm_result.stats
            admm_state_new = admm_result.state
            stop_after_qp = _should_stop_after_admm(admm_stats.status)
            states_new, controls_new = unpack_x(admm_state_new.x_blocks, self._zshape)
            if self.program.rescale_optimization_variables:
                states_new, controls_new = self.program.unscale_states_controls(
                    states_new, controls_new, problem_params
                )
            slacks_new = admm_state_new.xi_g

            def apply_linesearch(_):
                return self._constraint_handler.apply_linesearch(
                    problem_params,
                    qp_data,
                    admm_state_new,
                    constraint_carry,
                    (state.states, state.controls, state.slacks),
                    (states_new, controls_new, slacks_new),
                )

            # Linesearch
            linesearch_alphas_next = state.linesearch_alphas
            (
                (states_new, controls_new, slacks_new),
                constraint_carry_next,
                alpha,
            ) = jax.lax.cond(
                stop_after_qp,
                lambda _: (
                    (state.states, state.controls, state.slacks),
                    constraint_carry,
                    jnp.asarray(0.0, dtype=state.states.dtype),
                ),
                apply_linesearch,
                operand=None,
            )
            admm_state_accepted = jax.tree.map(
                lambda previous, candidate: jnp.where(
                    stop_after_qp, previous, candidate
                ),
                state.admm_state,
                admm_state_new,
            )
            if has_linesearch:
                linesearch_alphas_next = linesearch_alphas_next.at[state.it].set(alpha)

            z_new = pack_z(states_new, controls_new)
            step_conv = jnp.max(jnp.abs(z_new - state.z_prev))
            if slacks_new.shape[1] > 0:
                # non-empty inequality constraints
                step_conv += jnp.max(jnp.abs(slacks_new - state.slacks))

            next_it = state.it + 1

            # Formulate QP approximation for the next iteration & convergence check
            def _build_next_qp():
                return self._build_qp(
                    states_new,
                    controls_new,
                    slacks_new,
                    problem_params,
                    admm_state_accepted,
                    slack_penalization_weight=slack_penalization_weight,
                    hessian=self._forward_hessian,
                    scale_variables=True,
                )

            # Skip relinearization at the SQP iteration limit.
            if self._convergence_criterion == "first_order":

                def _build_qp_and_evaluate_first_order(_):
                    return self._first_order_check(
                        _build_next_qp(),
                        states_new,
                        controls_new,
                        slacks_new,
                        constraint_carry_next,
                        admm_state_accepted,
                        problem_params,
                    )

                def _evaluate_first_order_without_qp(_):
                    conv_next = jnp.where(state.it == 0, step_conv, state.conv)
                    return self._skipped_first_order_check(
                        state.qp_data,
                        conv_next,
                    )

                if max_iter <= 1:
                    check = _evaluate_first_order_without_qp(None)
                else:
                    check = jax.lax.cond(
                        next_it < max_iter,
                        _build_qp_and_evaluate_first_order,
                        _evaluate_first_order_without_qp,
                        operand=None,
                    )
            else:

                def _build_qp_and_evaluate_step(_):
                    return self._step_check_from_current_qp(
                        _build_next_qp(),
                        states_new,
                        controls_new,
                        slacks_new,
                        problem_params,
                        step_conv,
                    )

                def _evaluate_step_without_qp(_):
                    return self._step_check(
                        state.qp_data,
                        states_new,
                        controls_new,
                        slacks_new,
                        problem_params,
                        step_conv,
                    )

                if max_iter <= 1:
                    check = _evaluate_step_without_qp(None)
                else:
                    should_relinearize = jnp.logical_and(
                        next_it < max_iter,
                        step_conv > self.params["tol_convergence"],
                    )
                    check = jax.lax.cond(
                        should_relinearize,
                        _build_qp_and_evaluate_step,
                        _evaluate_step_without_qp,
                        operand=None,
                    )

            # Store solver stat data
            qp_stats_next = jax.tree.map(
                lambda history, value: history.at[state.it].set(value),
                state.qp_stats,
                admm_stats,
            )
            equality_violations_next = state.equality_violations.at[state.it].set(
                check.eq_violation
            )
            inequality_violations_next = state.inequality_violations.at[state.it].set(
                check.ineq_violation
            )
            convergence_errors_next = state.convergence_errors.at[state.it].set(
                check.convergence_error
            )
            convergence_key = (
                "step" if self._convergence_criterion == "step" else "conv"
            )
            debug_info = {
                "it": state.it,
                convergence_key: check.convergence_error,
                "eq": check.eq_violation,
                "ineq": check.ineq_violation,
                "admm": admm_stats.num_iter.astype(jnp.int32),
            }
            if has_linesearch:
                debug_info["alpha"] = alpha
            self._debug_print_sqp_iteration(debug_info)

            outer_finite = (
                jnp.all(jnp.isfinite(states_new))
                & jnp.all(jnp.isfinite(controls_new))
                & jnp.all(jnp.isfinite(slacks_new))
                & jnp.isfinite(check.convergence_error)
            )
            sqp_status = _sqp_iteration_status(outer_finite, admm_stats.status)

            state_next = _SQPState(
                it=state.it + 1,
                states=states_new,
                controls=controls_new,
                slacks=slacks_new,
                z_prev=z_new,
                conv=check.convergence_error,
                linesearch_alphas=linesearch_alphas_next,
                admm_state=admm_state_accepted,
                qp_stats=qp_stats_next,
                equality_violations=equality_violations_next,
                inequality_violations=inequality_violations_next,
                convergence_errors=convergence_errors_next,
                qp_data=check.qp_data,
                sqp_status=sqp_status,
            )
            return state_next, constraint_carry_next

        def cond_fun(carry) -> jnp.ndarray:
            # Run at least once, then stop at convergence, failure, or the limit.
            state = carry[0]
            _continue = jnp.logical_and(
                state.it < max_iter, state.conv > self.params["tol_convergence"]
            )
            _continue = jnp.logical_or(_continue, state.it < 1)
            _continue = _continue & (state.sqp_status == jnp.int32(SQPStatus.NOT_RUN))
            return _continue

        init_state = _SQPState(
            it=jnp.array(0),
            states=states,
            controls=controls,
            slacks=slacks0,
            z_prev=z_prev,
            conv=convergence_error,
            linesearch_alphas=linesearch_alphas,
            admm_state=admm_state0,
            qp_stats=qp_stats,
            equality_violations=equality_violations,
            inequality_violations=inequality_violations,
            convergence_errors=convergence_errors,
            qp_data=qp_data,
            sqp_status=jnp.int32(SQPStatus.NOT_RUN),
        )

        # The constraint carry is an empty pytree for projection handling.
        out, _ = jax.lax.while_loop(cond_fun, body_fun, (init_state, constraint_carry0))

        final_states_qp, final_controls_qp = self.program.scale_states_controls(
            out.states, out.controls, problem_params
        )
        final_x_blocks = pack_x(final_states_qp, final_controls_qp)
        ineq_values = _apply_G(out.qp_data, final_x_blocks)
        if self.program.use_slack_variables:
            ineq_values = ineq_values + out.slacks
        final_kkt_state = self._build_kkt_state(
            out.states,
            out.controls,
            out.slacks,
            out.admm_state,
            ineq_values,
            out.qp_data.ineq.l,
            out.qp_data.ineq.u,
        )
        last_iteration = jnp.maximum(out.it - 1, 0)
        terminal_status = _terminal_sqp_status(
            out.sqp_status,
            out.qp_stats.status[last_iteration],
            out.conv,
            self.params["tol_convergence"],
            max_iter,
        )
        stats = SQPStats(
            status=terminal_status,
            num_iter=out.it,
            qp_stats=out.qp_stats,
            convergence_errors=out.convergence_errors,
            equality_violations=out.equality_violations,
            inequality_violations=out.inequality_violations,
            line_search_step_sizes=out.linesearch_alphas,
        )
        self._debug_print_solve_summary(
            status=terminal_status,
            num_iter=out.it,
            convergence_error=out.conv,
            admm_iters=out.qp_stats.num_iter,
        )
        return TurboMPCSolution(
            states=out.states,
            controls=out.controls,
            slack=out.slacks,
            stats=stats,
            admm_state=out.admm_state,
            kkt_state=final_kkt_state,
        )
