"""PCG method for optimal control QPs."""
from typing import Any, Dict, NamedTuple, Tuple

import jax.numpy as jnp
from jax import lax, vmap
from jax.lax import while_loop
from turbompc.utils.load_params import load_solver_params

DEFAULT_PCG_SOLVER_PARAMS = (
    load_solver_params("sqp_osqp.yaml")
    .get("admm", {})
    .get("pcg", {"max_iter": 200, "tol_epsilon": 1.0e-8})
)


class PCGDebugOutput(NamedTuple):
    """Debug output after running PCG."""

    num_iterations: int
    convergence_eta: float


class SchurComplementMatrices(NamedTuple):
    """Schur complement matrix parameters of the QP."""

    S: jnp.ndarray  # (N+1, nx+nu, 3*(nx+nu))
    preconditioner_Phiinv: jnp.ndarray  # (N+1, nx+nu, 3*(nx+nu))


class PCGPrimalOptimalControl:
    """
    Preconditioned conjugate gradient solver over primal state-control variables.

    The Schur system is represented in block-tridiagonal form.
    """

    def __init__(
        self,
        problem_horizon: int,
        problem_num_states: int,
        problem_num_controls: int,
        solver_params: Dict[str, Any] = DEFAULT_PCG_SOLVER_PARAMS,
    ):
        self._params = solver_params
        self._name = "PCGPrimalOptimalControl"
        self._problem_horizon = problem_horizon
        self._problem_num_states = problem_num_states
        self._problem_num_controls = problem_num_controls

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> Dict:
        return self._params

    @property
    def horizon(self) -> int:
        return self._problem_horizon

    @property
    def num_states(self) -> int:
        return self._problem_num_states

    @property
    def num_controls(self) -> int:
        return self._problem_num_controls

    def solve_linear_system(
        self,
        schur_complement_matrices: SchurComplementMatrices,
        schur_complement_gammas: jnp.ndarray,
        zs_guess: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, PCGDebugOutput]:
        """Solve the Schur system using iterative PCG (pure JAX)."""
        S = schur_complement_matrices.S
        Phiinv = schur_complement_matrices.preconditioner_Phiinv
        gammas = schur_complement_gammas
        pcg_max_iter = self.params["max_iter"]
        pcg_epsilon = self.params["tol_epsilon"]
        nx = self.num_states
        nu = self.num_controls
        nz = nx + nu

        def pcg_get_r_init(gammas, S, zs):
            def get_r_block(gamma, S_block, z_prev, z, z_next):
                return gamma - S_block @ jnp.concatenate([z_prev, z, z_next])

            zs_padded = jnp.concatenate(
                [jnp.zeros((1, nz)), zs, jnp.zeros((1, nz))], axis=0
            )
            return vmap(get_r_block)(
                gammas, S, zs_padded[:-2], zs_padded[1:-1], zs_padded[2:]
            )

        r = pcg_get_r_init(gammas, S, zs_guess)
        rs = jnp.concatenate(
            [
                jnp.concatenate([jnp.zeros_like(r[0])[jnp.newaxis], r[:-1]], axis=0),
                r,
                jnp.concatenate([r[1:], jnp.zeros_like(r[0])[jnp.newaxis]], axis=0),
            ],
            axis=-1,
        )
        rtilde = vmap(lambda A, v: A @ v)(Phiinv, rs)
        p = rtilde.copy()
        eta = jnp.sum(r * rtilde)
        eta_init_abs = jnp.abs(eta)

        # Convergence criterion matches fused CUDA kernel:
        #   |eta| < 1e-12 + pcg_epsilon * |eta_init|
        pcg_abs_tol = 1e-12

        def cond_fun(val: Tuple):
            it, _, _, eta_val, _, eta_init = val
            converged = jnp.abs(eta_val) < pcg_abs_tol + pcg_epsilon * eta_init
            _continue = jnp.logical_and(~converged, it <= pcg_max_iter - 1)
            _continue = jnp.logical_or(_continue, it < 1)
            return _continue

        def pcg_iterate_fun(val):
            it, r, p, eta_val, zs, eta_init = val
            ps = jnp.concatenate(
                [
                    jnp.concatenate(
                        [jnp.zeros_like(p[0])[jnp.newaxis], p[:-1]], axis=0
                    ),
                    p,
                    jnp.concatenate([p[1:], jnp.zeros_like(p[0])[jnp.newaxis]], axis=0),
                ],
                axis=-1,
            )
            Upsilon = vmap(lambda A, v: A @ v)(S, ps)
            v = jnp.sum(p * Upsilon)
            alpha = eta_val / v
            zs = zs + alpha * p
            r = r - alpha * Upsilon
            rs = jnp.concatenate(
                [
                    jnp.concatenate(
                        [jnp.zeros_like(r[0])[jnp.newaxis], r[:-1]], axis=0
                    ),
                    r,
                    jnp.concatenate([r[1:], jnp.zeros_like(r[0])[jnp.newaxis]], axis=0),
                ],
                axis=-1,
            )
            rtilde = vmap(lambda A, v: A @ v)(Phiinv, rs)
            etaprime = jnp.sum(r * rtilde)
            beta = etaprime / eta_val
            p = rtilde + beta * p
            eta_val = etaprime
            return (it + 1, r, p, eta_val, zs, eta_init)

        # If `zs_guess` already solves the system (e.g., `gammas==0` and `zs_guess==0`),
        # then r==0 => eta==0 and the first PCG iteration would compute 0/0, producing NaNs.
        def _already_converged(_):
            zs = zs_guess
            return zs, PCGDebugOutput(
                num_iterations=0, convergence_eta=jnp.asarray(0.0, dtype=eta.dtype)
            )

        def _run_pcg(_):
            val = while_loop(
                cond_fun,
                pcg_iterate_fun,
                init_val=(0, r, p, eta, zs_guess, eta_init_abs),
            )
            zs = val[4]
            return zs, PCGDebugOutput(num_iterations=val[0], convergence_eta=val[3])

        return lax.cond(
            eta_init_abs <= pcg_abs_tol, _already_converged, _run_pcg, operand=None
        )
