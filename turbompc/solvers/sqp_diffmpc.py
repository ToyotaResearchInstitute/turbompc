"""
Implements SQP-PCG solver: https://github.com/ToyotaResearchInstitute/diffmpc.

It is slower than TurboMPC (since it is entirely written in JAX) and does not
support control rate costs, implicit integrators, or inequality constraints.
"""


import copy
import enum
from functools import partial
from typing import Any, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import grad, jacfwd
from turbompc.dynamics.integrators import DiscretizationScheme
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.linear_systems_solvers.pcg_dual import (
    PCGDualOptimalControl,
    QPDynamicsCostPremultipliedMatrices,
    QPSolution,
    SchurComplementMatrices,
)
from turbompc.solvers.qp_data import QPData, qpdata_from_ocp_blocks
from turbompc.utils.jax_utils import (
    project_matrix_onto_positive_semidefinite_cone,
    value_and_jacrev,
)
from turbompc.utils.load_params import load_solver_params

DEFAULT_SOLVER_PARAMS = load_solver_params("sqp.yaml")


class SolverReturnStatus(enum.IntEnum):
    """Status of result after solving the problem."""

    SUCCESS = 0
    ERROR = -100  # if nans are detected.


class SQPSolution(NamedTuple):
    # solution
    states: jnp.ndarray  # (horizon+1, nx) state trajectory
    controls: jnp.ndarray  # (horizon+1, nx) control trajectory
    dual: jnp.ndarray  # kkt multipliers for the equality constraints

    # QP parameters
    qp_data: QPData
    dynamics_cost_premultiplied_matrices: QPDynamicsCostPremultipliedMatrices
    schur_complement_matrices: SchurComplementMatrices

    status: SolverReturnStatus
    num_iter: int
    convergence_error: float

    convergence_errors: jnp.ndarray
    linesearch_alphas: jnp.ndarray
    num_pcg_iters: jnp.ndarray

    dual_backward_guess: jnp.ndarray = None


class SQPDiffMPCSolver:
    """SQP for optimal control (without inequality (box) constraints) solver."""

    _supported_program_types = [
        OptimalControlProblem,
    ]

    def __init__(
        self,
        program: OptimalControlProblem,
        name: str = "SQPDiffMPCSolver",
        params: Dict = DEFAULT_SOLVER_PARAMS,
    ):
        if params["verbose"]:
            print("Initializing Solver with")
            print("> name    =", name)
            print("> program =", program)

        program_is_supported = False
        for supported_program_type in self._supported_program_types:
            if isinstance(program, supported_program_type):
                program_is_supported = True
        if not program_is_supported:
            raise NotImplementedError(str(program.name) + " is not supported.")

        self._verbose = params["verbose"]
        self._params = params
        self._program = program
        self._name = name

        self._pcg = PCGDualOptimalControl(
            self.program.horizon,
            self.program.num_state_variables,
            self.program.num_control_variables,
            params["pcg"],
        )

        self.solve = self.get_differentiable_solve_function()
        self.solve(self.initial_guess(), problem_params=self.problem_params)

    @property
    def name(self) -> str:
        """Returns the name of the class."""
        return self._name

    @property
    def params(self) -> Dict:
        """Returns the parameters of the class."""
        return self._params

    @property
    def problem_params(self) -> Dict:
        """Returns the parameters of the optimization problem."""
        return self._program.params

    @property
    def maxiter(self) -> int:
        """Returns the maximum number of SQP iterations."""
        return self.params["num_sqp_iteration_max"]

    @property
    def program(self) -> OptimalControlProblem:
        """Returns the program class."""
        return self._program

    @property
    def verbose(self) -> bool:
        """Returns the verbosity level of the class."""
        return self._verbose

    @property
    def pcg(self) -> PCGDualOptimalControl:
        """Returns the PCG method of the class."""
        return self._pcg

    def initial_guess_primal(self, params: Dict[str, Any] = None) -> jnp.ndarray:
        """Returns primal initial guess."""
        return self.program.initial_guess(params)

    def initial_guess_dual(self) -> jnp.ndarray:
        """Returns KKT multipliers initial guess."""
        return jnp.zeros((self.program.horizon + 1, self.program.num_state_variables))

    def initial_guess(self, params: Dict[str, Any] = None) -> SQPSolution:
        return SQPSolution(
            self.initial_guess_primal(params)[0],
            self.initial_guess_primal(params)[1],
            self.initial_guess_dual(),
            self.pcg.zero_qp_data(),
            self.pcg.zero_dynamics_cost_premultiplied_matrices(),
            self.pcg.zero_schur_complement_matrices(),
            SolverReturnStatus.SUCCESS,
            int(0),  # num_iter
            float(0.0),  # convergence_error
            jnp.zeros(self.maxiter, dtype=float),  # convergence_errors,
            jnp.zeros(self.maxiter, dtype=float),  # linesearch_alphas,
            jnp.zeros(self.maxiter, dtype=int),  # num_pcg_iters
            self.initial_guess_dual(),  # dual_backward_guess
        )

    def get_cost_terms(
        self, states: jnp.ndarray, controls: jnp.ndarray, problem_params: Dict[str, Any]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns cost terms (Qs, qs, Rs, rs) for the SQP QP approximation.

        SQP-PCG only solves problems without off-diagonal stage cost coupling.
        Nonzero E blocks from get_cost_linearized_blocks are rejected because
        this solver cannot use them.

        Dimensions: (N, nx, nu) = (horizon, num_states, num_controls)

        Args:
            problem_params: dictionary of parameters of the optimal control problem,
                (key=string, value=Any)

        Returns:
            Qs: quadratic state cost matrices,
                (N + 1, nx, nx) array
            qs: state cost vectors,
                (N + 1, nx) array
            Rs: quadratic control cost matrices,
                (N + 1, nu, nu) array
            rs: control cost vectors,
                (N + 1, nu) array
        """
        D, E, q = self.program.get_cost_linearized_blocks(
            states, controls, problem_params
        )

        def _raise_cost_coupling_error():
            raise ValueError(
                "SQPDiffMPCSolver does not support nonzero off-diagonal cost blocks E "
                "from get_cost_linearized_blocks. Use TurboMPC for control-rate "
                "or other stage-coupled costs."
            )

        def _reject_cost_coupling(_):
            jax.debug.callback(_raise_cost_coupling_error)
            return jnp.array(0, dtype=jnp.int32)

        def _accept_uncoupled_cost(_):
            return jnp.array(0, dtype=jnp.int32)

        jax.lax.cond(
            jnp.any(jnp.abs(E) > 1e-12),
            _reject_cost_coupling,
            _accept_uncoupled_cost,
            operand=None,
        )

        nx = self.program.num_state_variables
        Qs = D[:, :nx, :nx]
        qs = q[:, :nx]
        Rs = D[:, nx:, nx:]
        rs = q[:, nx:]
        Qs = jax.vmap(
            project_matrix_onto_positive_semidefinite_cone, in_axes=(0, None)
        )(Qs, 1e-9)
        Rs = jax.vmap(
            project_matrix_onto_positive_semidefinite_cone, in_axes=(0, None)
        )(Rs, 1e-9)
        return Qs, qs, Rs, rs

    def get_dynamics_constraints_terms(
        self,
        states: jnp.array,
        controls: jnp.array,
        problem_params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns terms for the dynamics equality constraints
        linearized around a (states, controls) trajectory.

        Dimensions: (N, nx, nu) = (horizon, num_states, num_controls)

        Args:
            states: state trajectory,
                (N + 1, nx) array
            controls: control trajectory,
                (N + 1, nu) array
            problem_params: dictionary of parameters of the optimal control problem,
                (key=string, value=Any)

        Returns:
            Linearized dynamics blocks (As_next, Bs_next, As, Bs, Cs).
        """
        if self.program.discretization_scheme == DiscretizationScheme.IMPLICIT:
            raise ValueError(
                "SQPDiffMPCSolver does not support implicit integrators with control"
                " coupling. Use TurboMPC instead."
            )
        As_next, Bs_next, As, Bs, Cs = self.program.get_dynamics_linearized_matrices(
            states, controls, problem_params
        )
        return As_next, Bs_next, As, Bs, Cs

    def compute_qp_data(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        problem_params: Dict[str, jnp.ndarray],
    ) -> Tuple[QPData, QPDynamicsCostPremultipliedMatrices, SchurComplementMatrices]:
        """Computes the parameters of the QP approximation of the problem at (states,controls)."""
        Qs, qs, Rs, rs = self.get_cost_terms(states, controls, problem_params)
        As_next, Bs_next, As, Bs, Cs = self.get_dynamics_constraints_terms(
            states, controls, problem_params
        )
        if self.program.rescale_optimization_variables:
            _, _, _, _, state_scale, control_scale = self.program._get_rescaling_params(
                problem_params
            )
            Qs = (
                Qs
                * state_scale[jnp.newaxis, :, jnp.newaxis]
                * state_scale[jnp.newaxis, jnp.newaxis, :]
            )
            qs = qs * state_scale[jnp.newaxis, :]
            Rs = (
                Rs
                * control_scale[jnp.newaxis, :, jnp.newaxis]
                * control_scale[jnp.newaxis, jnp.newaxis, :]
            )
            rs = rs * control_scale[jnp.newaxis, :]
            As = As * state_scale[jnp.newaxis, jnp.newaxis, :]
            As_next = As_next * state_scale[jnp.newaxis, jnp.newaxis, :]
            Bs = Bs * control_scale[jnp.newaxis, jnp.newaxis, :]
            Bs_next = Bs_next * control_scale[jnp.newaxis, jnp.newaxis, :]
        Qinv = jnp.linalg.inv(Qs)
        Rinv = jnp.linalg.inv(Rs[:-1])
        n = self.program.num_state_variables + self.program.num_control_variables
        A0, c0 = self.program.get_initial_equality_linearized_matrices(
            problem_params, Qs.dtype
        )
        D = jnp.zeros((self.program.horizon + 1, n, n), dtype=Qs.dtype)
        D = D.at[
            :, : self.program.num_state_variables, : self.program.num_state_variables
        ].set(Qs)
        D = D.at[
            :, self.program.num_state_variables :, self.program.num_state_variables :
        ].set(Rs)
        E = jnp.zeros((self.program.horizon, n, n), dtype=Qs.dtype)
        q = jnp.concatenate([qs, rs], axis=-1)
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
            ineq_blocks=jnp.zeros(
                (
                    self.program.horizon + 1,
                    0,
                    self.program.num_state_variables
                    + self.program.num_control_variables,
                )
            ),
            ineq_l=jnp.zeros((self.program.horizon + 1, 0)),
            ineq_u=jnp.zeros((self.program.horizon + 1, 0)),
        )
        As_x_Qinv = jnp.matmul(As, Qinv[:-1])  # vmap(lambda A, B: A @ B)
        Asnext_x_Qinvnext = jnp.matmul(As_next, Qinv[1:])
        Bs_x_Rinv = jnp.matmul(Bs, Rinv)
        dynamics_cost_premultiplied_matrices = QPDynamicsCostPremultipliedMatrices(
            As_x_Qinv, Asnext_x_Qinvnext, Bs_x_Rinv
        )
        schur_complement_matrices = self.pcg.compute_S_Phiinv(qp_data)
        return (
            qp_data,
            dynamics_cost_premultiplied_matrices,
            schur_complement_matrices,
        )

    def solve_one_sqp_iteration(
        self,
        qp_solution: QPSolution,  # initial guess
        problem_params: Dict[str, Any],
    ) -> Tuple:
        # Setup QP problem
        (
            qp_data,
            dynamics_cost_premultiplied_matrices,
            schur_complement_matrices,
        ) = self.compute_qp_data(
            qp_solution.states,
            qp_solution.controls,
            problem_params,
        )
        nx = self.program.num_state_variables
        Q0inv = jnp.linalg.inv(qp_data.cost.D[0, :nx, :nx])
        qs = qp_data.cost.q[:, :nx]
        rs = qp_data.cost.q[:-1, nx:]
        c_stack = jnp.concatenate([qp_data.eq.c0[jnp.newaxis], qp_data.eq.c], axis=0)
        schur_complement_gammas = self.pcg.compute_gamma(
            dynamics_cost_premultiplied_matrices,
            c_stack,
            Q0inv,
            qs,
            rs,
        )

        # solve QP via PCG
        qp_solution, pcg_debug = self.pcg.solve_KKT_Schur(
            qp_data,
            schur_complement_matrices,
            schur_complement_gammas,
            qp_solution.kkt_multipliers,
        )
        if self.program.rescale_optimization_variables:
            _, _, _, _, state_scale, control_scale = self.program._get_rescaling_params(
                problem_params
            )
            qp_solution = qp_solution._replace(
                states=qp_solution.states * state_scale,
                controls=qp_solution.controls * control_scale,
            )
        return (
            qp_solution,
            qp_data,
            dynamics_cost_premultiplied_matrices,
            schur_complement_matrices,
            pcg_debug.num_iterations,
        )

    def linesearch(
        self,
        qp_solution: QPSolution,
        qp_solution_new: QPSolution,
        problem_params: Dict[str, Any],
    ) -> Tuple[QPSolution, float]:
        """Applies a linesearch for the solution to the QP approximation of problem.

        Reference: Algorithm 18.3 (Line Search SQP Algorithm) in
        [1] J. Nocedal and S. J. Wright, Numerical Optimization,
        Springer New York, second edition, 2006.

        Pseudocode:
        Set a = 1
        decrease_value = φ(x + a p; µ) - (φ(x; µ) + η a D(φ(x; µ) p))
        while decrease_value > 0
            Decrease a
        return x + a p

        Args:
            qp_solution: solution used to formulate the QP approximation
                of the (non-convex) program,
                (QPSolution)
            qp_solution_new: solution to the QP approximation,
                (QPSolution)
            problem_params: dictionary of parameters defining the problem,
                (key=string, value=Any)

        Returns:
            qp_solution_new: new solution of the QP approximation after the linesearch,
                (QPSolution)
        """
        horizon = self.program.horizon
        nx = self.program.num_state_variables
        eta = self.params["linesearch_eta"]
        alpha_candidates = jnp.array(self.params["linesearch_alphas"])

        # p
        states, controls = qp_solution.states, qp_solution.controls
        states_delta = qp_solution_new.states - states
        controls_delta = qp_solution_new.controls - controls
        z = jnp.concatenate([states, controls], axis=-1).flatten()
        z_delta = jnp.concatenate([states_delta, controls_delta], axis=-1).flatten()

        def cost_function(z):
            z = jnp.reshape(z, (horizon + 1, -1))
            return self.program.cost(z[:, :nx], z[:, nx:], problem_params) * jnp.ones(1)

        cost, cost_dz = value_and_jacrev(cost_function, z)
        cost, cost_dz = cost[0], cost_dz[0]
        cost_dz_times_delta_z = jnp.dot(cost_dz, z_delta)
        eq_constraints = self.program.equality_constraints(
            states, controls, problem_params
        )
        constraints_l1_norm = jnp.sum(jnp.abs(eq_constraints))

        # µ (see Eq. (18.36) in [1])
        linesearch_mu_min = cost_dz_times_delta_z / (0.5 * constraints_l1_norm)
        linesearch_mu = jnp.maximum(0.1, linesearch_mu_min)

        # φ(x; µ) and D(φ(x; µ) p) (see Eq. (18.27) & (18.29) in [1])
        merit_value = cost + linesearch_mu * constraints_l1_norm
        merit_derivative = cost_dz_times_delta_z + -linesearch_mu * constraints_l1_norm

        # Pre-compute decreases in the merit value for all candidate α's
        def decrease_function(alpha):
            # compute merit value φ(x + α p; µ) of candidate solution x + α p
            candidate_states = states + alpha * states_delta
            candidate_controls = controls + alpha * controls_delta
            cost = self.program.cost(
                candidate_states, candidate_controls, problem_params
            )
            eq_constraints = self.program.equality_constraints(
                candidate_states, candidate_controls, problem_params
            )
            merit_value_candidate = cost + linesearch_mu * jnp.sum(
                jnp.abs(eq_constraints)
            )

            # compute decrease of the merit function
            # φ(x + α p; µ) - (φ(x; µ) + η α D(φ(x; µ) p))
            decrease_value = merit_value_candidate - (
                merit_value + eta * alpha * merit_derivative
            )
            return decrease_value

        decrease_values = jax.vmap(decrease_function)(alpha_candidates)

        # Starting from α=1, decrease α until the decrease condition holds.
        def get_smaller_alpha(carry: Tuple[int, float]) -> Tuple[int, float]:
            i, _ = carry
            return (i - 1, alpha_candidates[i - 1])

        def continue_decreasing_alpha(carry: Tuple[int, float]) -> bool:
            i, _ = carry
            break_condition = decrease_values[i] < 0
            break_condition = jnp.logical_or(break_condition, i == 0)
            return ~break_condition

        index, alpha = len(alpha_candidates) + 1, jnp.max(alpha_candidates)
        alpha = jax.lax.while_loop(
            continue_decreasing_alpha, get_smaller_alpha, init_val=(index, alpha)
        )[1]

        # apply the linesearch
        qp_solution_new = QPSolution(
            states=states + alpha * states_delta,
            controls=controls + alpha * controls_delta,
            kkt_multipliers=qp_solution_new.kkt_multipliers,
        )
        return qp_solution_new, alpha

    def state_control_difference(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        new_states: jnp.ndarray,
        new_controls: jnp.ndarray,
    ) -> float:
        convergence_error = jnp.maximum(
            jnp.max(jnp.abs(new_states - states)),
            jnp.max(jnp.abs(new_controls - controls)),
        )
        return convergence_error

    def make_params_with_weights(self, weights, problem_params=None):
        if problem_params is None:
            new_params = copy.deepcopy(self.problem_params)
        else:
            new_params = copy.deepcopy(problem_params)

        for key, value in weights.items():
            new_params[key] = value
        return new_params

    def _solve(
        self,
        initial_guess: SQPSolution,
        problem_params: Dict[str, Any],
        weights: Any = {},
    ) -> SQPSolution:
        """Solves the optimal control problem.

        Args:
            initial_guess: initial guess for the solution of the program,
                SQPSolution
            problem_params: dictionary of parameters defining the program,
                (key=string, value=Any)
            weights: dictionary of parameters that might change and over-write problem_params,
                (key=string, value=jnp.array)

        Returns:
            solution: solution of the program,
                SQPSolution
        """
        initial_guess = jax.lax.stop_gradient(initial_guess)
        problem_params = jax.lax.stop_gradient(problem_params)
        problem_params = self.make_params_with_weights(weights, problem_params)

        def cond_fun(solution: SQPSolution):
            _continue = jnp.logical_and(
                solution.num_iter < self.maxiter,
                solution.convergence_error > self.params["tol_convergence"],
            )
            _continue = jnp.logical_or(_continue, solution.num_iter < 1)
            return _continue

        def sqp_iteration_body(previous_solution: SQPSolution):
            (
                states,
                controls,
                dual,
                qp_data,
                dynamics_cost_premultiplied_matrices,
                schur_complement_matrices,
                status,
                num_iter,
                convergence_error,
                convergence_errors,
                linesearch_alphas,
                num_pcg_iters,
                dual_backward_guess,
            ) = previous_solution

            qp_solution = QPSolution(states, controls, dual)
            (
                new_qp_solution,
                qp_data,
                dynamics_cost_premultiplied_matrices,
                schur_complement_matrices,
                num_pcg_iter,
            ) = self.solve_one_sqp_iteration(qp_solution, problem_params)
            if self.params["pcg"]["verbose"]:
                jax.debug.print(
                    "[solver::solve] (SQP iter={y}): PCG iter {i}",
                    y=num_iter,
                    i=num_pcg_iter,
                )
            if self.params["linesearch"]:
                new_qp_solution, alpha = self.linesearch(
                    qp_solution, new_qp_solution, problem_params
                )
                linesearch_alphas = linesearch_alphas.at[num_iter].set(alpha)
                if self.verbose:
                    jax.debug.print("linesearch alpha = {y}", y=alpha)
            convergence_error = self.state_control_difference(
                qp_solution.states,
                qp_solution.controls,
                new_qp_solution.states,
                new_qp_solution.controls,
            )
            return SQPSolution(
                new_qp_solution.states,
                new_qp_solution.controls,
                new_qp_solution.kkt_multipliers,
                qp_data,
                dynamics_cost_premultiplied_matrices,
                schur_complement_matrices,
                status,
                num_iter + 1,  # num_iter
                convergence_error,
                convergence_errors.at[num_iter].set(
                    convergence_error
                ),  # convergence_errors,
                linesearch_alphas,  # linesearch_alphas,
                num_pcg_iters.at[num_iter].set(
                    num_pcg_iter.astype(jnp.int32)
                ),  # num_pcg_iters
                dual_backward_guess,
            )

        initial_guess = initial_guess._replace(
            num_iter=0
        )  # reset num_iter so it runs at least once
        solution = jax.lax.while_loop(cond_fun, sqp_iteration_body, initial_guess)

        status = jnp.where(
            jnp.logical_or(
                jnp.logical_or(
                    jnp.isnan(solution.states).any(), jnp.isnan(solution.controls).any()
                ),
                solution.convergence_error > self.params["tol_convergence"],
            ),
            SolverReturnStatus.ERROR,
            SolverReturnStatus.SUCCESS,
        )
        solution = SQPSolution(
            states=jnp.nan_to_num(solution.states),
            controls=jnp.nan_to_num(solution.controls),
            dual=jnp.nan_to_num(solution.dual),
            qp_data=solution.qp_data,
            dynamics_cost_premultiplied_matrices=solution.dynamics_cost_premultiplied_matrices,
            schur_complement_matrices=solution.schur_complement_matrices,
            status=status,
            num_iter=solution.num_iter,
            convergence_error=solution.convergence_error,
            convergence_errors=solution.convergence_errors,
            linesearch_alphas=solution.linesearch_alphas,
            num_pcg_iters=solution.num_pcg_iters,
            dual_backward_guess=initial_guess.dual_backward_guess,
        )
        return solution

    def get_differentiable_solve_function(self):
        """Create a differentiable solve function with captured solver.

        Args:
            solver: Existing solver instance

        Returns:
            Differentiable solve function
        """

        @partial(jax.custom_vjp)
        def solve(
            initial_guess: SQPSolution,
            problem_params: Dict[str, Any],
            weights: Any = {},
        ):
            """Differentiable solve with captured solver. Problem params must be passed in explicitly.

            Args:
                ... (see solver.solve)
                weights: Dictionary of weights to differentiate with respect to

            Returns:
                solver solution
            """
            return self._solve(initial_guess, problem_params, weights)

        def _solve_fwd(
            initial_guess: SQPSolution,
            problem_params: Dict[str, Any],
            weights: Any,
        ):
            """VJP forward pass."""
            solution = self._solve(initial_guess, problem_params, weights)
            # add dual_backward_guess to solution
            solution = solution._replace(
                dual_backward_guess=initial_guess.dual_backward_guess
            )
            if self.verbose:
                jax.debug.print("forward pcg num iter = {y}", y=solution.num_pcg_iters)
            residual_for_backward_pass = (solution, weights)
            return solution, residual_for_backward_pass

        def _compute_mixed_derivatives(weights, states, controls, lambdas):
            def _cost_gradient(states, controls, params):
                def cost(states, controls):
                    return self.program.cost(states, controls, params)

                return grad(cost, argnums=(0, 1))(states, controls)

            def _constraints(states, controls, params):
                """(num_duals, )"""
                return self.program.equality_constraints(states, controls, params)

            def _constraints_weighted(states, controls, params, lambda_star):
                """Scalar-valued weighted constraints: λ*^T g(z)"""
                return lambda_star.flatten() @ _constraints(states, controls, params)

            def _constraints_weighted_dz(states, controls, params, lambda_star):
                """∇(λ*^T g(z)): (num_duals, num_primals)"""
                return jacfwd(
                    lambda x, u: _constraints_weighted(x, u, params, lambda_star),
                    argnums=(0, 1),
                )(states, controls)

            f_dx_theta, f_du_theta = jacfwd(
                lambda w: _cost_gradient(
                    states, controls, self.make_params_with_weights(w)
                )
            )(weights)
            g_theta = jacfwd(
                lambda w: _constraints(
                    states, controls, self.make_params_with_weights(w)
                )
            )(weights)
            g_dx_theta_weighted, g_du_theta_weighted = jacfwd(
                lambda w: _constraints_weighted_dz(
                    states, controls, self.make_params_with_weights(w), lambdas
                )
            )(weights)
            flat_f_dx_theta, _ = jax.tree_util.tree_flatten(f_dx_theta)
            flat_f_du_theta, _ = jax.tree_util.tree_flatten(f_du_theta)
            flat_g_dx_theta_weighted, _ = jax.tree_util.tree_flatten(
                g_dx_theta_weighted
            )
            flat_g_du_theta_weighted, _ = jax.tree_util.tree_flatten(
                g_du_theta_weighted
            )
            flat_g_theta, _ = jax.tree_util.tree_flatten(g_theta)
            flat_f_dx_theta = jnp.concatenate([arr for arr in flat_f_dx_theta], axis=-1)
            flat_f_du_theta = jnp.concatenate([arr for arr in flat_f_du_theta], axis=-1)
            flat_g_dx_theta_weighted = jnp.concatenate(
                [arr for arr in flat_g_dx_theta_weighted], axis=-1
            )
            flat_g_du_theta_weighted = jnp.concatenate(
                [arr for arr in flat_g_du_theta_weighted], axis=-1
            )
            flat_g_theta = jnp.concatenate([arr for arr in flat_g_theta], axis=-1)
            return (
                flat_f_dx_theta,
                flat_f_du_theta,
                flat_g_dx_theta_weighted,
                flat_g_du_theta_weighted,
                flat_g_theta,
            )

        def _solve_bwd_pcg(
            states,
            controls,
            kkt_multipliers,
            weights,
            dl_dstates,
            dl_dcontrols,
            qp_data,
            dynamics_cost_premultiplied_matrices,
            schur_complement_matrices,
            dual_backward_initial_guess,
        ):
            """PCG backward pass."""
            # update qs, rs
            qs = -dl_dstates
            rs = -dl_dcontrols[:-1]

            # update rest
            nx = self.program.num_state_variables
            rvec_full = jnp.concatenate([rs, jnp.zeros_like(rs[:1])], axis=0)
            q_full = jnp.concatenate([qs, rvec_full], axis=-1)
            qp_data = QPData(
                cost=qp_data.cost.__class__(
                    D=qp_data.cost.D,
                    E=qp_data.cost.E,
                    q=q_full,
                ),
                eq=qp_data.eq,
                ineq=qp_data.ineq,
            )
            Q0inv = jnp.linalg.inv(qp_data.cost.D[0, :nx, :nx])
            c_stack = jnp.concatenate(
                [
                    jnp.zeros_like(qp_data.eq.c0)[jnp.newaxis],
                    jnp.zeros_like(qp_data.eq.c),
                ],
                axis=0,
            )

            schur_complement_gammas = self.pcg.compute_gamma(
                dynamics_cost_premultiplied_matrices,
                c_stack,
                Q0inv,
                qs,
                rs,
            )

            # Step 3: Solve linear system - O(N^1.5)–O(N^2)
            linsys_sol, pcg_debug = self.pcg.solve_KKT_Schur(
                qp_data,
                schur_complement_matrices,
                schur_complement_gammas,
                dual_backward_initial_guess,
            )
            if self.verbose:
                jax.debug.print(
                    "backward pcg num iter = {y}", y=pcg_debug.num_iterations
                )

            # chain rule
            (
                f_dx_theta,
                f_du_theta,
                g_dx_theta_weighted,
                g_du_theta_weighted,
                g_theta,
            ) = _compute_mixed_derivatives(weights, states, controls, kkt_multipliers)

            # Compute dl/dθ using b - O(N n_θ)
            dL_dtheta = -(
                jnp.sum(
                    jnp.moveaxis(f_dx_theta + g_dx_theta_weighted, -1, 0)
                    * linsys_sol.states,
                    axis=(-2, -1),
                )
                + jnp.sum(
                    jnp.moveaxis(f_du_theta + g_du_theta_weighted, -1, 0)
                    * linsys_sol.controls,
                    axis=(-2, -1),
                )
                + jnp.sum(
                    jnp.moveaxis(g_theta, -1, 0) * linsys_sol.kkt_multipliers.flatten(),
                    axis=(-1),
                )
            )
            return dL_dtheta, linsys_sol.kkt_multipliers

        def _solve_bwd(residual_for_backward_pass, cotangents):
            """VJP backward pass with captured self."""
            solution, weights = residual_for_backward_pass

            dl_dstates = cotangents.states
            dl_dcontrols = cotangents.controls
            dual_backward_initial_guess = jnp.where(
                self.params["warm_start_backward"],
                cotangents.dual_backward_guess,
                jnp.zeros_like(solution.dual),
            )

            dL_dtheta_flat, dual_backward = _solve_bwd_pcg(
                solution.states,  # zstar_states,
                solution.controls,  # zstar_controls,
                solution.dual,  # lambda_star
                weights,
                dl_dstates,
                dl_dcontrols,
                solution.qp_data,
                solution.dynamics_cost_premultiplied_matrices,
                solution.schur_complement_matrices,
                dual_backward_initial_guess,
            )

            # get weights tree structure & reconstruct
            flat_weights, tree_def = jax.tree_util.tree_flatten(weights)
            flat_weight_lens = [arr.size for arr in flat_weights]
            start_idx = 0
            dL_dtheta_chunks = []
            for length in flat_weight_lens:
                dL_dtheta_chunks.append(dL_dtheta_flat[start_idx : start_idx + length])
                start_idx += length
            dL_dtheta_unflattened = jax.tree_util.tree_unflatten(
                tree_def, dL_dtheta_chunks
            )

            # Return gradients for each input arguments of differentiable_solve()
            return (
                SQPSolution(
                    states=None,
                    controls=None,
                    dual=None,
                    qp_data=None,
                    dynamics_cost_premultiplied_matrices=None,
                    schur_complement_matrices=None,
                    status=None,
                    num_iter=None,
                    convergence_error=None,
                    convergence_errors=None,
                    linesearch_alphas=None,
                    num_pcg_iters=None,
                    dual_backward_guess=dual_backward,
                ),  # initial_guess
                None,  # problem_params
                dL_dtheta_unflattened,
            )

        # Register the VJP
        solve.defvjp(_solve_fwd, _solve_bwd)
        return solve
