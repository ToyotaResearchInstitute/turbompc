"""TurboMPC SQP solver: Uses ADMM to solve the QP subproblems."""
import copy
import enum
from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from turbompc.problems.optimal_control_problem import (
    OptimalControlProblem,
    SlackProblemAdapter,
)
from turbompc.solvers.admm import ADMMSolver, ADMMState
from turbompc.solvers.admm.admm import (
    _apply_C,
    _apply_Ct,
    _apply_G,
    _apply_Gt,
    _apply_P,
    _inf_norm,
)
from turbompc.solvers.backward.backward_kkt_jax import solve_backward_kkt
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver
from turbompc.solvers.qp_data import (
    QPCostBlocks,
    QPData,
    QPEqualityBlocks,
    QPInequalityBlocks,
    qpdata_from_ocp_blocks,
    scale_qp_data,
)

try:
    from turbompc.solvers.backward.backward_kkt_cudss_ffi import (
        solve_backward_kkt_cudss_ffi as _solve_backward_kkt_cudss_ffi,
    )

    _HAS_CUDSS_FFI = True
except (FileNotFoundError, OSError, ImportError):
    _HAS_CUDSS_FFI = False
    _solve_backward_kkt_cudss_ffi = None
from turbompc.solvers.linear_systems_solvers.backends import (
    AdmmBackend,
    SchurSolverBackend,
)
from turbompc.solvers.linesearch import (
    backtracking_linesearch,
    evaluate_constraints_with_bounds,
)
from turbompc.solvers.qp_utils import ZShape, pack_x, pack_z
from turbompc.utils.load_params import load_solver_params, normalize_problem_params

DEFAULT_SOLVER_PARAMS = load_solver_params("turbompc.yaml")


class ForwardBackend(enum.IntEnum):
    """Backend used to solve the forward QP (one SQP iteration)."""

    ADMM_JAX_LOOP_PCG = 0
    ADMM_JAX_LOOP_PCG_FFI = 1
    ADMM_JAX_LOOP_CUDSS_FFI = 2
    ADMM_JAX_LOOP_JAX_DENSE = 3
    ADMM_FUSED_PCG = 4
    ADMM_FUSED_CUDSS = 5


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
    "admm_fused_pcg": ForwardBackend.ADMM_FUSED_PCG,
    "admm_fused_cudss": ForwardBackend.ADMM_FUSED_CUDSS,
}


_NAME_TO_BACKWARD_BACKEND: dict[str, BackwardBackend] = {
    "admm_jax_loop_pcg": BackwardBackend.ADMM_JAX_LOOP_PCG,
    "admm_jax_loop_pcg_ffi": BackwardBackend.ADMM_JAX_LOOP_PCG_FFI,
    "admm_jax_loop_cudss_ffi": BackwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
    "admm_jax_loop_jax_dense": BackwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
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

    states: jnp.ndarray  # (N+1, nx)
    controls: jnp.ndarray  # (N+1, nu)
    slack: jnp.ndarray  # (N+1, m)
    y_f: jnp.ndarray  # (N+1, nx)
    y_ineq: jnp.ndarray  # (N+1, m)
    ineq_active_lower_idx: jnp.ndarray  # (N+1, m) bool
    ineq_active_upper_idx: jnp.ndarray  # (N+1, m) bool


class SolverStats(NamedTuple):
    """Solver statistics."""

    admm_num_iters: jnp.ndarray
    eq_constraints_violations: jnp.ndarray
    ineq_constraints_violations: jnp.ndarray
    convergence_errors: jnp.ndarray


class TurboMPCConvergenceCheck(NamedTuple):
    qp_data: QPData
    convergence_error: jnp.ndarray
    eq_violation: jnp.ndarray
    ineq_violation: jnp.ndarray
    ineq_values: jnp.ndarray
    ineq_lower: jnp.ndarray
    ineq_upper: jnp.ndarray


class TurboMPCSolution(NamedTuple):
    states: jnp.ndarray  # (N+1, nx)
    controls: jnp.ndarray  # (N+1, nu)
    slack: jnp.ndarray  # (N+1, m)
    status: int  # 0 success, negative failure
    num_iter: jnp.ndarray
    convergence_error: jnp.ndarray
    admm_iters: jnp.ndarray  # (num_iter,)
    linesearch_alphas: Optional[jnp.ndarray] = None
    admm_state: Optional[ADMMState] = None
    solver_stats: Optional[SolverStats] = None
    dual_backward_guess: Optional[jnp.ndarray] = None
    kkt_state: Optional[KKTState] = None


class _BackwardQPContext(NamedTuple):
    """Context retained while converting the linearized QP into the adjoint QP."""

    base_qp: QPData
    solve_qp: QPData
    G_active: jnp.ndarray


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


def identify_active_inequalities(
    g: jnp.ndarray,
    l: jnp.ndarray,
    u: jnp.ndarray,
    eps_abs: float,
    eps_rel: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Identify active lower/upper inequality constraints.

    Uses the rule:
      min(g-l, u-g) <= eps_abs + eps_rel * |bound|.
    """
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


class TurboMPCSolver:
    """SQP solver that uses ADMM for the QP subproblem."""

    STATUS_SUCCESS = 0
    STATUS_FAILED = -1

    _supported_program_types = [OptimalControlProblem, SlackProblemAdapter]

    def __init__(
        self,
        program: OptimalControlProblem,
        params: Optional[Dict[str, Any]] = None,
        name: str = "TurboMPCSolver",
        *,
        forward_backend: ForwardBackend = ForwardBackend.ADMM_FUSED_CUDSS,
        backward_backend: BackwardBackend = BackwardBackend.DIRECT_CUDSS_FFI,
        use_full_hessian: bool = True,
    ):
        self._program = program
        self._name = name
        self._params = DEFAULT_SOLVER_PARAMS if params is None else params

        self._use_full_hessian = use_full_hessian

        self._forward_backend = parse_forward_backend(forward_backend)
        self._backward_backend = parse_backward_backend(backward_backend)
        self._convergence_criterion = parse_convergence_criterion(
            self.params.get("convergence_criterion", "first_order")
        )

        self._forward_admm_backend, self._forward_schur_backend = (
            _FORWARD_BACKEND_COMPONENTS[self._forward_backend]
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

        program_is_supported = any(
            isinstance(program, t) for t in self._supported_program_types
        )
        if not program_is_supported:
            # class instances with matching names are also supported
            supported_names = {t.__name__ for t in self._supported_program_types}
            if program.__class__.__name__ in supported_names:
                program_is_supported = True
        if not program_is_supported:
            raise NotImplementedError(str(program.name) + " is not supported.")

        self._use_slack = bool(program.use_slack_variables)

        self._zshape = ZShape(
            horizon=self.program.horizon,
            num_states=self.program.num_state_variables,
            num_controls=self.program.num_control_variables,
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
            adapt_rho_every=admm_params.get("adapt_rho_every", 5),
            adaptive_rho_tolerance=admm_params.get("adaptive_rho_tolerance", 5.0),
            rho_f_factor=admm_params.get(
                "rho_f_factor", admm_params.get("active_constraint_rho_factor", 1000.0)
            ),
            admm_backend=self._forward_admm_backend,
            use_slack=self._use_slack,
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
                adapt_rho_every=admm_params.get("adapt_rho_every", 5),
                adaptive_rho_tolerance=admm_params.get("adaptive_rho_tolerance", 5.0),
                rho_f_factor=1.0,
                admm_backend=self._backward_admm_backend,
                use_slack=self._use_slack,
            )
            if self._backward_backend
            not in {BackwardBackend.DIRECT_JAX_DENSE, BackwardBackend.DIRECT_CUDSS_FFI}
            else None
        )
        self.solve = self.get_differentiable_solve_function()

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
        if self.params["linesearch"]:
            linesearch_alphas = jnp.zeros((max_iter,), dtype=states.dtype)
        solver_stats = SolverStats(
            admm_num_iters=jnp.zeros(max_iter, dtype=int),
            eq_constraints_violations=jnp.zeros(max_iter, dtype=float),
            ineq_constraints_violations=jnp.zeros(max_iter, dtype=float),
            convergence_errors=jnp.zeros(max_iter, dtype=float),
        )
        return TurboMPCSolution(
            states=states,
            controls=controls,
            slack=jnp.zeros(
                (states.shape[0], self.program.num_inequality_constraints),
                dtype=states.dtype,
            ),
            status=self.STATUS_SUCCESS,
            num_iter=jnp.array(0),
            convergence_error=jnp.array(0.0, dtype=states.dtype),
            admm_iters=jnp.zeros((max_iter,), dtype=jnp.int32),
            linesearch_alphas=linesearch_alphas,
            admm_state=None,
            solver_stats=solver_stats,
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
    ) -> _BackwardQPContext:
        kkt_state = forward_solution.kkt_state
        # Backward adjoint QP: keep primal variables in true coordinates,
        # because cotangents are taken w.r.t. true state/control variables.
        qp_data = self._build_qp_data(
            kkt_state.states,
            kkt_state.controls,
            problem_params,
            apply_variable_scaling=False,
        )
        if self._use_full_hessian:
            qp_data = self._augment_D_with_dynamics_hessian(
                qp_data, forward_solution, problem_params
            )
            qp_data = self._augment_D_with_inequality_hessian(
                qp_data, forward_solution, problem_params
            )

        eq = qp_data.eq
        eq = QPEqualityBlocks(
            A0=eq.A0,
            A_minus=eq.A_minus,
            A_plus=eq.A_plus,
            c0=jnp.zeros_like(eq.c0),
            c=jnp.zeros_like(eq.c),
        )

        if qp_data.ineq.G.shape[1] == 0:
            G_active = qp_data.ineq.G
        else:
            sign = kkt_state.ineq_active_lower_idx.astype(
                qp_data.ineq.G.dtype
            ) - kkt_state.ineq_active_upper_idx.astype(qp_data.ineq.G.dtype)
            # G_act = sign * G, with inactive rows zeroed.
            G_active = qp_data.ineq.G * sign[..., jnp.newaxis]

        if qp_data.ineq.use_slack_variables and qp_data.ineq.G.shape[1] > 0:
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
                use_slack_variables=False,
            )
        else:
            q = -jnp.concatenate([dl_dstates, dl_dcontrols], axis=-1)
            cost = QPCostBlocks(D=qp_data.cost.D, E=qp_data.cost.E, q=q)
            ineq = QPInequalityBlocks(
                G=G_active,
                l=jnp.zeros_like(qp_data.ineq.l),
                u=jnp.zeros_like(qp_data.ineq.u),
                slack_penalization_weight=qp_data.ineq.slack_penalization_weight,
                use_slack_variables=qp_data.ineq.use_slack_variables,
            )
        return _BackwardQPContext(
            base_qp=qp_data,
            solve_qp=QPData(cost=cost, eq=eq, ineq=ineq),
            G_active=G_active,
        )

    def _recover_slack_active_ineq_gradient(
        self,
        backward_qp: _BackwardQPContext,
        forward_solution: TurboMPCSolution,
        dx_states: jnp.ndarray,
        dx_controls: jnp.ndarray,
        dl_dslack: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        kkt_state = forward_solution.kkt_state
        active_mask = (
            kkt_state.ineq_active_lower_idx | kkt_state.ineq_active_upper_idx
        ).astype(dl_dslack.dtype)
        dl_dslack_active = dl_dslack * active_mask
        dx = jnp.concatenate([dx_states, dx_controls], axis=-1)
        g_dx = jnp.matmul(backward_qp.G_active, dx[..., jnp.newaxis])[..., 0]
        gamma = backward_qp.base_qp.ineq.slack_penalization_weight
        y_ineq_lin = (dl_dslack_active + gamma * g_dx).reshape(-1)
        xi_active = kkt_state.slack * active_mask
        # dL/dgamma needs xi_paper = sign * xi_code.
        sign = kkt_state.ineq_active_lower_idx.astype(
            dl_dslack.dtype
        ) - kkt_state.ineq_active_upper_idx.astype(dl_dslack.dtype)
        dL_dgamma = jnp.sum(sign * xi_active * g_dx)
        return y_ineq_lin, dL_dgamma

    def _solve_backward_qp_admm(
        self,
        backward_qp: _BackwardQPContext,
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
        (dx_states, dx_controls), _, admm_state = admm_solver.solve(
            qp_data=backward_qp.solve_qp,
            admm_state0=admm_state0,
            rho_bar=rho_active,
            slack_weight=backward_qp.solve_qp.ineq.slack_penalization_weight,
        )

        y_eq_lin = self._pack_admm_eq_multipliers(admm_state)
        qp_data = backward_qp.base_qp
        if qp_data.ineq.use_slack_variables and qp_data.ineq.G.shape[1] > 0:
            y_ineq_lin, dL_dgamma = self._recover_slack_active_ineq_gradient(
                backward_qp,
                forward_solution,
                dx_states,
                dx_controls,
                dl_dslack,
            )
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
        backward_qp: _BackwardQPContext,
        forward_solution: TurboMPCSolution,
        dl_dslack: jnp.ndarray,
    ) -> _BackwardSolveResult:
        """Solve the reduced adjoint system via a direct KKT solve."""
        if self._backward_backend == BackwardBackend.DIRECT_CUDSS_FFI:
            (dx_states, dx_controls), multipliers = _solve_backward_kkt_cudss_ffi(
                qp_data=backward_qp.solve_qp,
                zshape=self._zshape,
            )
        else:
            (dx_states, dx_controls), multipliers = solve_backward_kkt(
                qp_data=backward_qp.solve_qp, zshape=self._zshape
            )

        solve_qp = backward_qp.solve_qp
        base_qp = backward_qp.base_qp
        kkt_state = forward_solution.kkt_state
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

        if base_qp.ineq.use_slack_variables and base_qp.ineq.G.shape[1] > 0:
            y_ineq_lin, dL_dgamma = self._recover_slack_active_ineq_gradient(
                backward_qp,
                forward_solution,
                dx_states,
                dx_controls,
                dl_dslack,
            )
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
        active_lower,
        active_upper,
    ) -> _MixedDerivatives:
        def _cost_gradient(states, controls, params):
            def cost(states, controls):
                return self.program.cost(states, controls, params)

            return jax.grad(cost, argnums=(0, 1))(states, controls)

        def _eq_constraints(states, controls, params):
            return self.program.equality_constraints(states, controls, params)

        def _active_ineq(states, controls, params):
            g, l, u = self.program.inequality_constraints(states, controls, params)
            g = g.reshape((-1,))
            l = l.reshape((-1,))
            u = u.reshape((-1,))
            lower = active_lower.reshape((-1,)).astype(states.dtype)
            upper = active_upper.reshape((-1,)).astype(states.dtype)
            return lower * (g - l) + upper * (u - g)

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
            h_val = _active_ineq(states, controls, params)
            g_dx, g_du = _constraints_weighted_dz(
                states, controls, params, y_eq, _eq_constraints
            )
            h_dx, h_du = _constraints_weighted_dz(
                states, controls, params, y_ineq, _active_ineq
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

    def _weights_without_slack_penalty(self, weights):
        if "slack_penalization_weight" in weights:
            return {
                k: v for k, v in weights.items() if k != "slack_penalization_weight"
            }
        return weights

    def _forward_multipliers(
        self, forward_solution: TurboMPCSolution
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        kkt_state = forward_solution.kkt_state
        sign = kkt_state.ineq_active_lower_idx.astype(kkt_state.y_ineq.dtype) - kkt_state.ineq_active_upper_idx.astype(kkt_state.y_ineq.dtype)
        if forward_solution.admm_state is not None:
            y_ineq = forward_solution.admm_state.y_g
            if y_ineq.shape[1] > 0:
                # Map ADMM bound duals (normal cone of [l, u]) to multipliers for
                # active constraints h(x)=g-l (lower) / h(x)=u-g (upper), both >= 0.
                y_ineq = -sign * y_ineq
            return (
                self._pack_admm_eq_multipliers(forward_solution.admm_state),
                y_ineq.reshape(-1),
            )
        y_ineq = kkt_state.y_ineq
        if y_ineq.shape[1] > 0:
            y_ineq = -sign * y_ineq
        return kkt_state.y_f.reshape(-1), y_ineq.reshape(-1)

    def _compute_weight_gradient(
        self,
        forward_solution: TurboMPCSolution,
        problem_params: Dict[str, Any],
        weights: Any,
        backward_result: _BackwardSolveResult,
    ) -> jnp.ndarray:
        weights_for_mixed = self._weights_without_slack_penalty(weights)
        kkt_state = forward_solution.kkt_state
        y_eq_star, y_ineq_star = self._forward_multipliers(forward_solution)
        mixed = self._compute_mixed_derivatives(
            problem_params,
            weights_for_mixed,
            kkt_state.states,
            kkt_state.controls,
            y_eq_star,
            y_ineq_star,
            kkt_state.ineq_active_lower_idx,
            kkt_state.ineq_active_upper_idx,
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
        weights_no_slacks = self._weights_without_slack_penalty(weights)
        flat_weights, tree_def = jax.tree_util.tree_flatten(weights_no_slacks)
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
            problem_params = self.make_params_with_weights(weights, problem_params)
            return self._solve_impl(
                initial_guess.states,
                initial_guess.controls,
                problem_params,
                initial_guess.admm_state,
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
            problem_params = self.make_params_with_weights(weights, problem_params_orig)
            backward_qp = self._build_backward_qp(
                solution,
                cotangents.states,
                cotangents.controls,
                cotangents.slack,
                problem_params,
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

    def _build_qp_data(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        problem_params: Dict[str, Any],
        apply_variable_scaling: bool = True,
    ) -> QPData:
        """Assemble QP blocks.

        Rescaling: 
        - `apply_variable_scaling` controls only the solver-level column scaling
          (for the optimization variables) performed by `scale_qp_data(...)`.
        - Problem-level row scaling may already be present in the OCP constraints
          when `rescale_optimization_variables=True`.
        """
        D, E, q = self.program.get_cost_linearized_blocks(
            states, controls, problem_params
        )
        # D = jax.vmap(
        #    project_matrix_onto_positive_semidefinite_cone, in_axes=(0, None)
        # )(D, 1e-12)

        As_next, Bs_next, As, Bs, Cs = self.program.get_dynamics_linearized_matrices(
            states, controls, problem_params
        )

        ineq_blocks, ineq_l, ineq_u = self.program.get_inequalities_linearized_matrices(
            states, controls, problem_params
        )
        use_slack_variables = self.program.use_slack_variables
        slack_penalization_weight = jnp.asarray(
            problem_params.get("slack_penalization_weight", 0.0), dtype=states.dtype
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
            use_slack_variables=use_slack_variables,
            slack_penalization_weight=slack_penalization_weight,
        )
        if self.program.rescale_optimization_variables and apply_variable_scaling:
            _, _, _, _, state_diff, control_diff = self.program._get_rescaling_params(
                problem_params
            )
            qp_data = scale_qp_data(qp_data, state_diff, control_diff)
        return qp_data

    def _compute_first_order_convergence_error(
        self,
        qp_data: QPData,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        admm_state: ADMMState,
        problem_params: Dict[str, Any],
    ) -> Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        states_qp, controls_qp = self.program.scale_states_controls(
            states, controls, problem_params
        )
        x_blocks = pack_x(states_qp, controls_qp)

        stationarity = (
            _apply_P(qp_data, x_blocks)
            + qp_data.cost.q
            + _apply_Ct(qp_data, admm_state.y_f_0, admm_state.y_f_dyn)
            + _apply_Gt(qp_data, admm_state.y_g)
        )
        stationarity_error = _inf_norm(stationarity)

        c_stack = jnp.concatenate([qp_data.eq.c0, qp_data.eq.c.reshape(-1)], axis=0)
        eq_residual = _apply_C(qp_data, x_blocks) - c_stack
        eq_error = _inf_norm(eq_residual)

        ineq_values = _apply_G(qp_data, x_blocks)
        ineq_lower = qp_data.ineq.l
        ineq_upper = qp_data.ineq.u
        if qp_data.ineq.use_slack_variables:
            ineq_values_for_bounds = ineq_values + slacks
        else:
            ineq_values_for_bounds = ineq_values
        if qp_data.ineq.G.shape[1] > 0:
            ineq_violation = jnp.maximum(
                jnp.maximum(ineq_lower - ineq_values_for_bounds, 0.0),
                jnp.maximum(ineq_values_for_bounds - ineq_upper, 0.0),
            )
            ineq_error = _inf_norm(ineq_violation)
        else:
            ineq_error = jnp.array(0.0, dtype=states.dtype)

        if qp_data.ineq.use_slack_variables and qp_data.ineq.G.shape[1] > 0:
            slack_stationarity_error = _inf_norm(
                qp_data.ineq.slack_penalization_weight * slacks + admm_state.y_g
            )
        else:
            slack_stationarity_error = jnp.array(0.0, dtype=states.dtype)

        conv = jnp.maximum(
            stationarity_error,
            jnp.maximum(eq_error, jnp.maximum(ineq_error, slack_stationarity_error)),
        )
        return (
            conv,
            eq_error,
            ineq_error,
            ineq_values_for_bounds,
            ineq_lower,
            ineq_upper,
        )

    def _inequality_values_from_qp(
        self,
        qp_data: QPData,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        problem_params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        states_qp, controls_qp = self.program.scale_states_controls(
            states, controls, problem_params
        )
        values = _apply_G(qp_data, pack_x(states_qp, controls_qp))
        if qp_data.ineq.use_slack_variables:
            values = values + slacks
        return values, qp_data.ineq.l, qp_data.ineq.u

    def _skipped_first_order_check(
        self,
        qp_data: QPData,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        problem_params: Dict[str, Any],
        convergence_error: jnp.ndarray,
    ) -> TurboMPCConvergenceCheck:
        ineq_values, ineq_lower, ineq_upper = self._inequality_values_from_qp(
            qp_data, states, controls, slacks, problem_params
        )
        zero = jnp.array(0.0, dtype=states.dtype)
        return TurboMPCConvergenceCheck(
            qp_data=qp_data,
            convergence_error=convergence_error,
            eq_violation=zero,
            ineq_violation=zero,
            ineq_values=ineq_values,
            ineq_lower=ineq_lower,
            ineq_upper=ineq_upper,
        )

    def _first_order_check(
        self,
        qp_data: QPData,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        admm_state: ADMMState,
        problem_params: Dict[str, Any],
    ) -> TurboMPCConvergenceCheck:
        (
            conv,
            eq_violation,
            ineq_violation,
            ineq_values,
            ineq_lower,
            ineq_upper,
        ) = self._compute_first_order_convergence_error(
            qp_data, states, controls, slacks, admm_state, problem_params
        )
        return TurboMPCConvergenceCheck(
            qp_data=qp_data,
            convergence_error=conv,
            eq_violation=eq_violation,
            ineq_violation=ineq_violation,
            ineq_values=ineq_values,
            ineq_lower=ineq_lower,
            ineq_upper=ineq_upper,
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
        (
            eq_violation,
            ineq_violation,
            ineq_values,
            ineq_lower,
            ineq_upper,
        ) = self._evaluate_constraints_with_bounds(
            states, controls, slacks, problem_params
        )
        return TurboMPCConvergenceCheck(
            qp_data=qp_data,
            convergence_error=convergence_error,
            eq_violation=eq_violation,
            ineq_violation=ineq_violation,
            ineq_values=ineq_values,
            ineq_lower=ineq_lower,
            ineq_upper=ineq_upper,
        )

    def _augment_D_with_dynamics_hessian(
        self,
        qp_data,
        forward_solution,
        problem_params,
    ):
        """Add lambda^T nabla^2 f to D blocks for exact Lagrangian Hessian backward pass."""
        kkt_state = forward_solution.kkt_state
        # Get forward dynamics multipliers
        if forward_solution.admm_state is not None:
            lambdas = forward_solution.admm_state.y_f_dyn  # (N, nx)
        else:
            # Extract from packed kkt_state.y_f
            nx = self.program.num_state_variables
            m0 = qp_data.eq.A0.shape[0]
            y_f_flat = kkt_state.y_f.reshape(-1)
            lambdas = y_f_flat[m0 : m0 + self.program.horizon * nx].reshape(
                self.program.horizon, nx
            )

        dynamics_hessian = self.program.get_dynamics_lagrangian_hessian(
            kkt_state.states, kkt_state.controls, problem_params, lambdas
        )

        # The backward QP uses unscaled primal variables (`apply_variable_scaling=False`),
        # so the dynamics Hessian in true coordinates is added directly here.
        D_augmented = qp_data.cost.D + dynamics_hessian
        return QPData(
            cost=QPCostBlocks(D=D_augmented, E=qp_data.cost.E, q=qp_data.cost.q),
            eq=qp_data.eq,
            ineq=qp_data.ineq,
        )

    def _augment_D_with_inequality_hessian(
        self,
        qp_data,
        forward_solution,
        problem_params,
    ):
        """Add sum_i mu_i nabla^2 g_i to D blocks for the exact Lagrangian Hessian backward.
        """
        if qp_data.ineq.G.shape[1] == 0:
            return qp_data
        kkt_state = forward_solution.kkt_state
        if forward_solution.admm_state is not None:
            mus = forward_solution.admm_state.y_g  # (N+1, m), raw
        else:
            mus = kkt_state.y_ineq  # (N+1, m), raw

        inequality_hessian = self.program.get_inequality_lagrangian_hessian(
            kkt_state.states, kkt_state.controls, problem_params, mus
        )

        # Backward QP uses unscaled primal variables, so add in true coordinates.
        D_augmented = qp_data.cost.D + inequality_hessian
        return QPData(
            cost=QPCostBlocks(D=D_augmented, E=qp_data.cost.E, q=qp_data.cost.q),
            eq=qp_data.eq,
            ineq=qp_data.ineq,
        )

    def _build_kkt_state(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        admm_state: ADMMState,
        ineq_values: jnp.ndarray,
        ineq_lower: jnp.ndarray,
        ineq_upper: jnp.ndarray,
    ) -> KKTState:
        if admm_state is None:
            y_f = jnp.zeros_like(states)
            y_ineq = jnp.zeros((states.shape[0], 0), dtype=states.dtype)
            slack = jnp.zeros_like(y_ineq)
        else:
            nx = self.program.num_state_variables
            y_f0 = admm_state.y_f_0
            if self.program.constrain_initial_control:
                y_f0 = y_f0[:nx]
            y_f = jnp.concatenate([y_f0[jnp.newaxis], admm_state.y_f_dyn], axis=0)
            y_ineq = admm_state.y_g
            slack = admm_state.xi_g

        g, l, u = ineq_values, ineq_lower, ineq_upper
        g = g.reshape((states.shape[0], -1))
        l = l.reshape((states.shape[0], -1))
        u = u.reshape((states.shape[0], -1))
        if self.program.use_slack_variables and admm_state is not None:
            active_lower = slack > 1.0e-12
            active_upper = slack < -1.0e-12
        else:
            prox_lower, prox_upper = identify_active_inequalities(
                g, l, u, self.params["admm"]["eps_abs"], self.params["admm"]["eps_rel"]
            )
            # For hard constraints, disambiguate lower-vs-upper activity using
            # dual sign whenever ADMM duals are available.
            if admm_state is not None and y_ineq.shape[1] > 0:
                y_tol = self.params["admm"]["eps_abs"]
                dual_lower = y_ineq < -y_tol
                dual_upper = y_ineq > y_tol
                undecided = ~(dual_lower | dual_upper)
                active_lower = dual_lower | (undecided & prox_lower)
                active_upper = dual_upper | (undecided & prox_upper)
            else:
                active_lower, active_upper = prox_lower, prox_upper
        return KKTState(
            states=states,
            controls=controls,
            slack=slack,
            y_f=y_f,
            y_ineq=y_ineq,
            ineq_active_lower_idx=active_lower,
            ineq_active_upper_idx=active_upper,
        )

    def _init_kkt_state(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        admm_state: ADMMState,
        num_ineq: int,
    ) -> KKTState:
        if admm_state is None:
            y_f = jnp.zeros_like(states)
            slack = jnp.zeros((states.shape[0], num_ineq), dtype=states.dtype)
        else:
            nx = self.program.num_state_variables
            y_f0 = admm_state.y_f_0
            if self.program.constrain_initial_control:
                y_f0 = y_f0[:nx]
            y_f = jnp.concatenate([y_f0[jnp.newaxis], admm_state.y_f_dyn], axis=0)
            slack = admm_state.xi_g
        empty = jnp.zeros((states.shape[0], num_ineq), dtype=states.dtype)
        return KKTState(
            states=states,
            controls=controls,
            slack=slack,
            y_f=y_f,
            y_ineq=empty,
            ineq_active_lower_idx=empty.astype(bool),
            ineq_active_upper_idx=empty.astype(bool),
        )

    def _evaluate_constraints_with_bounds(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        problem_params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return evaluate_constraints_with_bounds(
            self.program, states, controls, slacks, problem_params
        )

    def linesearch(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        states_new: jnp.ndarray,
        controls_new: jnp.ndarray,
        slacks_new: jnp.ndarray,
        problem_params: Dict[str, Any],
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Applies a linesearch along the SQP step direction via shared helper."""
        return backtracking_linesearch(
            self.program,
            problem_params,
            self.params,
            states,
            controls,
            slacks,
            states_new,
            controls_new,
            slacks_new,
        )

    def solve(
        self,
        initial_guess: TurboMPCSolution,
        problem_params: Dict[str, Any],
        weights: Any = {},
    ) -> TurboMPCSolution:
        problem_params = jax.lax.stop_gradient(problem_params)
        problem_params = self.make_params_with_weights(weights, problem_params)

        return self._solve_impl(
            initial_guess.states,
            initial_guess.controls,
            problem_params,
            initial_guess.admm_state,
        )

    def _solve_impl(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        problem_params: Dict[str, Any],
        admm_state0: Optional[ADMMState] = None,
    ) -> TurboMPCSolution:
        max_iter = self.params["num_sqp_iteration_max"]
        admm_iters = jnp.zeros((max_iter,), dtype=jnp.int32)
        linesearch_alphas = None
        if self.params["linesearch"]:
            linesearch_alphas = jnp.zeros((max_iter,), dtype=states.dtype)

        z_prev = pack_z(states, controls)
        convergence_error = jnp.array(jnp.inf, dtype=states.dtype)

        # Initialize stat arrays
        admm_num_iters = jnp.zeros(max_iter, dtype=int)
        eq_constraints_violations = jnp.zeros(max_iter, dtype=float)
        ineq_constraints_violations = jnp.zeros(max_iter, dtype=float)
        convergence_errors = jnp.zeros(max_iter, dtype=float)

        admm_params = self.params["admm"]
        admm_solver = self._admm_solver_fwd
        qp_data = self._build_qp_data(states, controls, problem_params)

        if admm_state0 is None:
            admm_state0 = admm_solver.initial_state(
                qp_data=qp_data,
                rho_bar=admm_params["rho"],
                states0=jnp.zeros_like(states),
                controls0=jnp.zeros_like(controls),
            )

        class _SQPState(NamedTuple):
            it: jnp.ndarray
            states: jnp.ndarray
            controls: jnp.ndarray
            slacks: jnp.ndarray
            z_prev: jnp.ndarray
            conv: jnp.ndarray
            admm_iters: jnp.ndarray
            linesearch_alphas: Optional[jnp.ndarray]
            admm_state: ADMMState
            admm_num_iters: jnp.ndarray
            eq_constraints_violations: jnp.ndarray
            ineq_constraints_violations: jnp.ndarray
            convergence_errors: jnp.ndarray
            kkt_state: KKTState
            qp_data: QPData

        def body_fun(state: _SQPState) -> _SQPState:
            """Body function for SQP iteration."""
            qp_data = state.qp_data

            # Solve QP via ADMM
            (states_new, controls_new), admm_stats, admm_state_new = admm_solver.solve(
                qp_data=qp_data,
                admm_state0=state.admm_state,
                rho_bar=admm_params["rho"],
                alpha=self.params["admm"].get("relaxation_parameter", 1.0),
                slack_weight=qp_data.ineq.slack_penalization_weight,
            )
            if self.program.rescale_optimization_variables:
                states_new, controls_new = self.program.unscale_states_controls(
                    states_new, controls_new, problem_params
                )
            slacks_new = admm_state_new.xi_g
            admm_iters_next = state.admm_iters.at[state.it].set(
                admm_stats.num_iter.astype(jnp.int32)
            )

            # Line search
            linesearch_alphas_next = state.linesearch_alphas
            alpha = 1.0
            if self.params["linesearch"]:
                (states_new, controls_new, slacks_new), alpha = self.linesearch(
                    state.states,
                    state.controls,
                    state.slacks,
                    states_new,
                    controls_new,
                    slacks_new,
                    problem_params,
                )
                linesearch_alphas_next = linesearch_alphas_next.at[state.it].set(alpha)

            z_new = pack_z(states_new, controls_new)
            step_conv = jnp.max(jnp.abs(z_new - state.z_prev))
            if slacks_new.shape[1] > 0:
                # non-empty inequality constraints
                step_conv += jnp.max(jnp.abs(slacks_new - state.slacks))

            next_it = state.it + 1

            # Formulate QP approximation for the next iteration & convergence check
            def _build_next_qp():
                return self._build_qp_data(states_new, controls_new, problem_params)

            if self._convergence_criterion == "first_order":

                def _compute_first_order(_):
                    return self._first_order_check(
                        _build_next_qp(),
                        states_new,
                        controls_new,
                        slacks_new,
                        admm_state_new,
                        problem_params,
                    )

                def _skip_first_order(_):
                    conv_next = jnp.where(state.it == 0, step_conv, state.conv)
                    return self._skipped_first_order_check(
                        state.qp_data,
                        states_new,
                        controls_new,
                        slacks_new,
                        problem_params,
                        conv_next,
                    )

                if max_iter <= 1:
                    check = _skip_first_order(None)
                else:
                    check = jax.lax.cond(
                        next_it < max_iter,
                        _compute_first_order,
                        _skip_first_order,
                        operand=None,
                    )
            else:
                if max_iter <= 1:
                    qp_data_next = state.qp_data
                else:
                    should_relinearize = jnp.logical_and(
                        next_it < max_iter,
                        step_conv > self.params["tol_convergence"],
                    )
                    qp_data_next = jax.lax.cond(
                        should_relinearize,
                        lambda _: _build_next_qp(),
                        lambda _: state.qp_data,
                        operand=None,
                    )
                check = self._step_check(
                    qp_data_next,
                    states_new,
                    controls_new,
                    slacks_new,
                    problem_params,
                    step_conv,
                )

            # Store solver stat data
            admm_num_iters_next = state.admm_num_iters.at[state.it].set(
                admm_stats.num_iter.astype(jnp.int32)
            )
            eq_constraints_violations_next = state.eq_constraints_violations.at[
                state.it
            ].set(check.eq_violation)
            ineq_constraints_violations_next = state.ineq_constraints_violations.at[
                state.it
            ].set(check.ineq_violation)
            convergence_errors_next = state.convergence_errors.at[state.it].set(
                check.convergence_error
            )
            kkt_state_next = self._build_kkt_state(
                states_new,
                controls_new,
                admm_state_new,
                ineq_values=check.ineq_values,
                ineq_lower=check.ineq_lower,
                ineq_upper=check.ineq_upper,
            )

            return _SQPState(
                it=state.it + 1,
                states=states_new,
                controls=controls_new,
                slacks=slacks_new,
                z_prev=z_new,
                conv=check.convergence_error,
                admm_iters=admm_iters_next,
                linesearch_alphas=linesearch_alphas_next,
                admm_state=admm_state_new,
                admm_num_iters=admm_num_iters_next,
                eq_constraints_violations=eq_constraints_violations_next,
                ineq_constraints_violations=ineq_constraints_violations_next,
                convergence_errors=convergence_errors_next,
                kkt_state=kkt_state_next,
                qp_data=check.qp_data,
            )

        def cond_fun(state: _SQPState) -> jnp.ndarray:
            _continue = jnp.logical_and(
                state.it < max_iter, state.conv > self.params["tol_convergence"]
            )
            _continue = jnp.logical_or(_continue, state.it < 1)
            return _continue

        num_ineq = int(admm_state0.y_g.shape[-1])
        init_kkt_state = self._init_kkt_state(states, controls, admm_state0, num_ineq)
        init_state = _SQPState(
            it=jnp.array(0),
            states=states,
            controls=controls,
            slacks=admm_state0.xi_g,
            z_prev=z_prev,
            conv=convergence_error,
            admm_iters=admm_iters,
            linesearch_alphas=linesearch_alphas,
            admm_state=admm_state0,
            admm_num_iters=admm_num_iters,
            eq_constraints_violations=eq_constraints_violations,
            ineq_constraints_violations=ineq_constraints_violations,
            convergence_errors=convergence_errors,
            kkt_state=init_kkt_state,
            qp_data=qp_data,
        )

        out = jax.lax.while_loop(cond_fun, body_fun, init_state)

        solver_stats = SolverStats(
            admm_num_iters=out.admm_num_iters,
            eq_constraints_violations=out.eq_constraints_violations,
            ineq_constraints_violations=out.ineq_constraints_violations,
            convergence_errors=out.convergence_errors,
        )
        return TurboMPCSolution(
            states=out.states,
            controls=out.controls,
            slack=out.admm_state.xi_g,
            status=self.STATUS_SUCCESS,
            num_iter=out.it,
            convergence_error=out.conv,
            admm_iters=out.admm_iters,
            linesearch_alphas=out.linesearch_alphas,
            admm_state=out.admm_state,
            solver_stats=solver_stats,
            kkt_state=out.kkt_state,
        )
