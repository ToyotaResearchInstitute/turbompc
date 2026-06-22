"""
SQP solver using OSQP to solve the convex QP at each SQP iteration.

This is a CPU OSQP reference implementation used for tests and debugging. It is
not differentiable, not GPU accelerated, and not the optimized OSQP baseline
used for the paper's racing experiments.
"""
import copy
from dataclasses import dataclass
from time import time
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import osqp
import scipy.sparse as sp
from turbompc.problems.optimal_control_problem import (
    OptimalControlProblem,
    SlackProblemAdapter,
)
from turbompc.solvers.linesearch import (
    backtracking_linesearch,
    evaluate_constraints_with_bounds,
)
from turbompc.solvers.qp_data import QPData, qpdata_from_ocp_blocks, scale_qp_data
from turbompc.solvers.qp_utils import ZShape, pack_z, unpack_z
from turbompc.utils.jax_utils import project_matrix_onto_positive_semidefinite_cone
from turbompc.utils.load_params import load_solver_params

DEFAULT_SOLVER_PARAMS = load_solver_params("sqp_osqp.yaml")


@dataclass(frozen=True)
class SQPOSQPSolution:
    states: jnp.ndarray  # (N+1, nx)
    controls: jnp.ndarray  # (N+1, nu)
    slack: jnp.ndarray  # (N+1, m)
    status: int  # 0 success, negative failure
    num_iter: int
    convergence_error: float
    iters: np.ndarray  # (num_iter,) OSQP iterations each SQP iteration
    linesearch_alphas: Optional[np.ndarray] = None
    solver_stats: Optional[Dict[str, np.ndarray]] = None


class SQPOSQPSolver:
    """
    SQP solver that solves the inner QPs with OSQP.

    Decision variable:
        z = concat([states, controls], axis=-1).flatten()
        where states.shape=(N+1,nx), controls.shape=(N+1,nu)
    """

    STATUS_SUCCESS = 0
    STATUS_OSQP_FAILED = -1
    STATUS_NAN_DETECTED = -2

    _supported_program_types = [
        OptimalControlProblem,
        SlackProblemAdapter,
    ]

    def __init__(
        self,
        program: OptimalControlProblem,
        params: Optional[Dict[str, Any]] = None,
        name: str = "SQPOSQPSolver",
    ):
        self._program = program
        self._name = name
        self._params = DEFAULT_SOLVER_PARAMS if params is None else params

        if self.params["verbose"]:
            print("Initializing Solver with")
            print("> name    =", name)
            print("> program =", program)

        program_is_supported = False
        for supported_program_type in self._supported_program_types:
            if isinstance(program, supported_program_type):
                program_is_supported = True
        if not program_is_supported:
            raise NotImplementedError(str(program.name) + " is not supported.")

        self._zshape = ZShape(
            horizon=self.program.horizon,
            num_states=self.program.num_state_variables,
            num_controls=self.program.num_control_variables,
        )

        self._osqp_prob: Optional[osqp.OSQP] = None

        # Sparsity patterns
        self._P_pattern: Optional[sp.csc_matrix] = None
        self._Aeq_pattern: Optional[sp.csc_matrix] = None
        self._Aineq_pattern: Optional[sp.csc_matrix] = None
        self._A_full_pattern: Optional[sp.csc_matrix] = None

        # Counter for unit tests / debugging
        self._num_osqp_setups: int = 0

    @property
    def program(self) -> OptimalControlProblem:
        return self._program

    @property
    def params(self) -> Dict:
        return self._params

    @property
    def name(self) -> str:
        return self._name

    def _pack(self, states: jnp.ndarray, controls: jnp.ndarray) -> jnp.ndarray:
        return pack_z(states, controls)

    def _unpack(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return unpack_z(z, self._zshape)

    def _build_qp_data(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        problem_params: Dict[str, Any],
    ) -> QPData:
        D, E, q = self.program.get_cost_linearized_blocks(
            states, controls, problem_params
        )
        D = jax.vmap(project_matrix_onto_positive_semidefinite_cone, in_axes=(0, None))(
            D, 1e-12
        )

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
        if problem_params.get("rescale_optimization_variables", False):
            _, _, _, _, state_diff, control_diff = self.program._get_rescaling_params(
                problem_params
            )
            qp_data = scale_qp_data(qp_data, state_diff, control_diff)
        return qp_data

    def _evaluate_constraints_with_bounds(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slacks: jnp.ndarray,
        problem_params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute constraint violations and return inequality bounds."""
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
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]:
        """Wrap the shared backtracking linesearch helper."""
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

    def _build_qp_matrices_dense(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        problem_params: Dict[str, Any],
    ) -> Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        """
        Build dense QP:
            min 0.5 z^T P z + q^T z
            s.t. A z = b  (encoded as l=u=b for OSQP)

        The full OSQP constraints will be built as:
            A_full = [Aeq; Aineq]
            l_full = [beq; l_ineq]
            u_full = [beq; u_ineq]

        Returns:
            P: (nz,nz) (dense, symmetric)
            q: (nz,)
            Aeq:    (m_eq, nz)
            beq:    (m_eq,)
            Aineq:  (m_ineq, nz)
            l_ineq: (m_ineq,)
            u_ineq: (m_ineq,)
        """
        qp_data = self._build_qp_data(states, controls, problem_params)
        N = self.program.horizon
        nx = self.program.num_state_variables
        n = qp_data.cost.D.shape[1]
        nz = (N + 1) * n

        P = jnp.zeros((nz, nz), dtype=states.dtype)
        q = jnp.zeros((nz,), dtype=states.dtype)
        for t in range(N + 1):
            i0 = t * n
            P = P.at[i0 : i0 + n, i0 : i0 + n].set(qp_data.cost.D[t])
            q = q.at[i0 : i0 + n].set(qp_data.cost.q[t])
        for t in range(N):
            i0 = t * n
            i1 = (t + 1) * n
            P = P.at[i1 : i1 + n, i0 : i0 + n].set(qp_data.cost.E[t])
            P = P.at[i0 : i0 + n, i1 : i1 + n].set(qp_data.cost.E[t].T)

        n0 = qp_data.eq.A0.shape[0]
        m_eq = n0 + N * nx
        Aeq = jnp.zeros((m_eq, nz), dtype=states.dtype)
        beq = jnp.zeros((m_eq,), dtype=states.dtype)
        beq = beq.at[:n0].set(qp_data.eq.c0)
        beq = beq.at[n0:].set(qp_data.eq.c.reshape(-1))
        Aeq = Aeq.at[:n0, :n].set(qp_data.eq.A0)
        for t in range(N):
            row = n0 + t * nx
            col_prev = t * n
            col_next = (t + 1) * n
            Aeq = Aeq.at[row : row + nx, col_prev : col_prev + n].set(
                qp_data.eq.A_minus[t]
            )
            Aeq = Aeq.at[row : row + nx, col_next : col_next + n].set(
                qp_data.eq.A_plus[t]
            )

        use_slack_variables = problem_params.get("use_slack_variables", False)
        if use_slack_variables and not isinstance(self.program, SlackProblemAdapter):
            raise ValueError(
                "use_slack_variables=True requires a SlackProblemAdapter instance."
            )
        slack_penalization_weight = problem_params.get("slack_penalization_weight", 0.0)

        if qp_data.ineq.G.shape[0] > 0:
            m_ineq = qp_data.ineq.G.shape[1]
            Aineq = jnp.zeros(((N + 1) * m_ineq, nz), dtype=states.dtype)
            for t in range(N + 1):
                row = t * m_ineq
                col = t * n
                Aineq = Aineq.at[row : row + m_ineq, col : col + n].set(
                    qp_data.ineq.G[t]
                )
            l_ineq = qp_data.ineq.l.reshape(-1)
            u_ineq = qp_data.ineq.u.reshape(-1)
        else:
            Aineq = jnp.zeros((0, nz), dtype=states.dtype)
            l_ineq = jnp.zeros((0,), dtype=states.dtype)
            u_ineq = jnp.zeros((0,), dtype=states.dtype)

        if use_slack_variables and Aineq.shape[0] > 0:
            m_total = Aineq.shape[0]
            slack_dim = m_total
            P = jnp.block(
                [
                    [P, jnp.zeros((nz, slack_dim), dtype=states.dtype)],
                    [
                        jnp.zeros((slack_dim, nz), dtype=states.dtype),
                        slack_penalization_weight
                        * jnp.eye(slack_dim, dtype=states.dtype),
                    ],
                ]
            )
            q = jnp.concatenate(
                [q, jnp.zeros((slack_dim,), dtype=states.dtype)], axis=0
            )
            Aeq = jnp.concatenate(
                [Aeq, jnp.zeros((Aeq.shape[0], slack_dim), dtype=states.dtype)], axis=1
            )
            Aineq = jnp.concatenate(
                [Aineq, jnp.eye(slack_dim, dtype=states.dtype)], axis=1
            )
        return P, q, Aeq, beq, Aineq, l_ineq, u_ineq

    def _ensure_osqp_patterns(
        self,
        P_dense_np: np.ndarray,
        Aeq_dense_np: np.ndarray,
        Aineq_dense_np: np.ndarray,
    ) -> None:
        """
        Create and cache sparse patterns for P and A_full = [Aeq; Aineq] once.
        Subsequent iterations reuse the same CSC structure.
        """
        if self._P_pattern is None:
            self._P_pattern = sp.csc_matrix(P_dense_np)

        if self._Aeq_pattern is None:
            self._Aeq_pattern = sp.csc_matrix(Aeq_dense_np)

        if self._Aineq_pattern is None:
            self._Aineq_pattern = sp.csc_matrix(Aineq_dense_np)

        if self._A_full_pattern is None:
            self._A_full_pattern = sp.vstack(
                [self._Aeq_pattern, self._Aineq_pattern], format="csc"
            )

    def _update_sparse_data(
        self,
        P_dense_np: np.ndarray,
        q_np: np.ndarray,
        Aeq_dense_np: np.ndarray,
        beq_np: np.ndarray,
        Aineq_dense_np: np.ndarray,
        l_ineq_np: np.ndarray,
        u_ineq_np: np.ndarray,
    ) -> Tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray, np.ndarray]:
        """
        Given new values, updated the cached CSC arrays to match the
        cached sparsity pattern.

        Returns:
            P_csc, q, A_full_csc, l_full, u_full
        """
        assert self._P_pattern is not None
        assert self._Aeq_pattern is not None
        assert self._Aineq_pattern is not None
        assert self._A_full_pattern is not None

        def _fill_csc_data_from_dense(
            pattern: sp.csc_matrix, dense: np.ndarray
        ) -> np.ndarray:
            indptr = pattern.indptr
            rows = pattern.indices
            col_counts = np.diff(indptr)
            cols = np.repeat(np.arange(pattern.shape[1]), col_counts)
            return dense[rows, cols]

        self._P_pattern.data = _fill_csc_data_from_dense(self._P_pattern, P_dense_np)
        self._Aeq_pattern.data = _fill_csc_data_from_dense(
            self._Aeq_pattern, Aeq_dense_np
        )
        self._Aineq_pattern.data = _fill_csc_data_from_dense(
            self._Aineq_pattern, Aineq_dense_np
        )

        self._A_full_pattern = sp.vstack(
            [self._Aeq_pattern, self._Aineq_pattern], format="csc"
        )

        l_full = np.concatenate([beq_np, l_ineq_np])
        u_full = np.concatenate([beq_np, u_ineq_np])

        return self._P_pattern, q_np, self._A_full_pattern, l_full, u_full

    def _setup_or_update_osqp(
        self,
        P: sp.csc_matrix,
        q: np.ndarray,
        A: sp.csc_matrix,
        l: np.ndarray,
        u: np.ndarray,
        do_update: bool,
    ) -> None:
        osqp_params = self.params.get("osqp", {})
        if (self._osqp_prob is None) or (not do_update):
            self._num_osqp_setups += 1
            self._osqp_prob = osqp.OSQP()
            self._osqp_prob.setup(
                P,
                q,
                A,
                l,
                u,
                eps_abs=osqp_params.get("eps_abs", 1e-6),
                eps_rel=osqp_params.get("eps_rel", 1e-6),
                max_iter=osqp_params.get("max_iter", 10000),
                check_termination=osqp_params.get("check_termination_every", 25),
                verbose=osqp_params.get("verbose", False),
                polish=osqp_params.get("polish", False),
                warm_start=osqp_params.get("warm_start", True),
                scaling=osqp_params.get("scaling", 0),
                alpha=osqp_params.get("relaxation_parameter", 1.6),
            )
        else:
            # Update only data; keep sparsity pattern
            self._osqp_prob.update(
                Px=sp.triu(P).data,
                q=q,
                Ax=A.data,
                l=l,
                u=u,
            )

    def make_params_with_weights(self, weights, problem_params=None):
        if problem_params is None:
            new_params = copy.deepcopy(self.problem_params)
        else:
            new_params = copy.deepcopy(problem_params)

        for key, value in weights.items():
            new_params[key] = value
        return new_params

    def solve(
        self,
        initial_guess: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        problem_params: Optional[Dict[str, Any]] = None,
        weights: Any = {},
    ) -> SQPOSQPSolution:
        if problem_params is None:
            problem_params = self.program.params

        problem_params = jax.lax.stop_gradient(problem_params)
        problem_params = self.make_params_with_weights(weights, problem_params)

        if initial_guess is None:
            states, controls = self.program.initial_guess(problem_params)
        else:
            states, controls = initial_guess
        N = self.program.horizon
        n = states.shape[1] + controls.shape[1]
        nz = (N + 1) * n

        maxiter = self.params["num_sqp_iteration_max"]
        iters = np.zeros((maxiter,), dtype=int)

        linesearch_alphas = None
        if self.params["linesearch"]:
            linesearch_alphas = np.zeros(maxiter, dtype=float)

        z_prev = self._pack(states, controls)
        convergence_error = np.inf
        status = 0

        # Initialize stat arrays
        qp_define_times = np.zeros(maxiter, dtype=float)
        qp_solve_times = np.zeros(maxiter, dtype=float)
        admm_num_iters = np.zeros(maxiter, dtype=int)
        eq_constraints_violations = np.zeros(maxiter, dtype=float)
        ineq_constraints_violations = np.zeros(maxiter, dtype=float)
        convergence_errors = np.zeros(maxiter, dtype=float)
        total_times = np.zeros(maxiter, dtype=float)

        # Slack placeholder for linesearch/backtracking
        if problem_params.get("use_slack_variables", False):
            slack = jnp.zeros(
                (N + 1, self.program.num_inequality_constraints), dtype=states.dtype
            )
        else:
            slack = jnp.zeros((N + 1, 0), dtype=states.dtype)
        # First iteration: setup OSQP once (sparsity pattern fixed for fixed N,nx,nu)
        # Later iterations: update P,q,A,l,u only.
        for k in range(maxiter):
            tt0 = time()
            P, q, Aeq, beq, Aineq, l_ineq, u_ineq = self._build_qp_matrices_dense(
                states, controls, problem_params
            )
            P = np.array(P)
            q = np.array(q)
            Aeq = np.array(Aeq)
            beq = np.array(beq)
            Aineq = np.array(Aineq)
            l_ineq = np.array(l_ineq)
            u_ineq = np.array(u_ineq)
            self._ensure_osqp_patterns(P, Aeq, Aineq)
            # convert to CSC
            P, q, A, l, u = self._update_sparse_data(
                P, q, Aeq, beq, Aineq, l_ineq, u_ineq
            )

            do_update = k > 0
            self._setup_or_update_osqp(P, q, A, l, u, do_update=do_update)
            qp_define_times[k] = time() - tt0

            tt1 = time()
            result = self._osqp_prob.solve()
            qp_solve_times[k] = time() - tt1
            iters[k] = int(result.info.iter)
            admm_num_iters[k] = iters[k]

            if result.info.status != "solved":
                status = -1
                slack = jnp.zeros((N + 1, 0), dtype=states.dtype)
                break

            z_new = jnp.array(result.x, dtype=states.dtype)
            if problem_params.get("use_slack_variables", False) and Aineq.shape[0] > 0:
                n_base = nz
                z_primal = z_new[:n_base]
                slack_candidate = z_new[n_base:].reshape((N + 1, -1))
            else:
                z_primal = z_new
                slack_candidate = jnp.zeros((N + 1, 0), dtype=states.dtype)
            states_new, controls_new = self._unpack(z_primal)
            if problem_params.get("rescale_optimization_variables", False):
                states_new, controls_new = self.program.unscale_states_controls(
                    states_new, controls_new, problem_params
                )
            z_new = self._pack(states_new, controls_new)

            if self.params["linesearch"]:
                (states_new, controls_new, slack), alpha = self.linesearch(
                    states,
                    controls,
                    slack,
                    states_new,
                    controls_new,
                    slack_candidate,
                    problem_params,
                )
                linesearch_alphas[k] = alpha
                if self.params["verbose"]:
                    print(f"linesearch alpha = {alpha}")

                z_new = self._pack(states_new, controls_new)
            else:
                slack = slack_candidate
            total_times[k] = time() - tt0

            eq_constraints_violations[k] = jnp.linalg.norm(
                self.program.equality_constraints(
                    states_new, controls_new, problem_params
                ),
                ord=jnp.inf,
            )

            ineq_values, ineq_lower, ineq_upper = self.program.inequality_constraints(
                states_new, controls_new, problem_params
            )
            if problem_params.get("use_slack_variables", False):
                ineq_values, _, _ = self.program.inequality_constraints_with_slack(
                    states_new, controls_new, slack, problem_params
                )
            ineq_values = ineq_values.reshape(-1)
            ineq_lower = ineq_lower.reshape(-1)
            ineq_upper = ineq_upper.reshape(-1)
            ineq_violation = jnp.maximum(0.0, ineq_lower - ineq_values) + jnp.maximum(
                0.0, ineq_values - ineq_upper
            )
            ineq_constraints_violations[k] = jnp.linalg.norm(
                ineq_violation, ord=jnp.inf
            )

            convergence_error = float(jnp.max(jnp.abs(z_new - z_prev)))
            convergence_errors[k] = convergence_error
            z_prev = z_new
            states, controls = states_new, controls_new

            if convergence_error <= self.params["tol_convergence"]:
                num_completed = k + 1
                iters = iters[:num_completed]
                if linesearch_alphas is not None:
                    linesearch_alphas = linesearch_alphas[:num_completed]
                solver_stats = {
                    "qp_define_times": qp_define_times[:num_completed],
                    "qp_solve_times": qp_solve_times[:num_completed],
                    "admm_num_iters": admm_num_iters[:num_completed],
                    "eq_constraints_violations": eq_constraints_violations[
                        :num_completed
                    ],
                    "ineq_constraints_violations": ineq_constraints_violations[
                        :num_completed
                    ],
                    "convergence_errors": convergence_errors[:num_completed],
                    "total_times": total_times[:num_completed],
                }
                return SQPOSQPSolution(
                    states=states,
                    controls=controls,
                    slack=slack,
                    status=0,
                    num_iter=num_completed,
                    convergence_error=convergence_error,
                    iters=iters,
                    linesearch_alphas=linesearch_alphas,
                    solver_stats=solver_stats,
                )

        # If we exit loop without convergence
        num_completed = k + 1
        iters = iters[:num_completed]
        if linesearch_alphas is not None:
            linesearch_alphas = linesearch_alphas[:num_completed]
        solver_stats = {
            "qp_define_times": qp_define_times[:num_completed],
            "qp_solve_times": qp_solve_times[:num_completed],
            "admm_num_iters": admm_num_iters[:num_completed],
            "eq_constraints_violations": eq_constraints_violations[:num_completed],
            "ineq_constraints_violations": ineq_constraints_violations[:num_completed],
            "convergence_errors": convergence_errors[:num_completed],
            "total_times": total_times[:num_completed],
        }
        return SQPOSQPSolution(
            states=states,
            controls=controls,
            slack=slack,
            status=status,
            num_iter=num_completed,
            convergence_error=float(convergence_error),
            iters=iters,
            linesearch_alphas=linesearch_alphas,
            solver_stats=solver_stats,
        )
