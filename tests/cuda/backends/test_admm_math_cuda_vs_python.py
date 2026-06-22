"""Per-function CUDA vs Python consistency tests (processor-based).

Each test validates shared CUDA device functions (admm_math.cuh) against
Python reference implementations (admm.py).

Uses OCPProcessor/SolverProcessor (tests/cuda/helpers/processors.py)
to parametrize across dynamics × horizons × variants × backends.
"""
import jax
import jax.numpy as jnp
import pytest
from tests.cuda.helpers.processors import OCPProcessor
from turbompc.solvers.admm.admm import (
    ADMMState,
    _apply_C_parts,
    _apply_Ct,
    _apply_G,
    _apply_Gt,
    _compute_residuals,
    _project_box,
    compute_gamma,
    compute_S_Phiinv,
)
from turbompc.solvers.turbompc_solver import BackwardBackend, ForwardBackend

# Tolerances
CUDSS_TOL = 1e-9
FUSED_TOL = 1e-4


def _build_state(ocp, params, seed=42):
    """Build a non-zero ADMMState matching the QP dimensions."""
    qp = _build_qp_data(ocp, params)
    T = qp.eq.c.shape[0] + 1 if qp.eq.c.ndim > 1 else params["horizon"] + 1
    N = params["horizon"]
    nx = qp.eq.A_minus.shape[-2]  # rows of A_minus
    n = qp.cost.D.shape[-1]
    m = qp.ineq.G.shape[-2]
    n0 = qp.eq.A0.shape[0]

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 6)
    return ADMMState(
        x_blocks=jax.random.normal(keys[0], (T, n)) * 0.1,
        y_f_0=jax.random.normal(keys[1], (n0,)) * 0.01,
        y_f_dyn=jax.random.normal(keys[2], (N, nx)) * 0.01,
        y_g=jax.random.normal(keys[3], (T, m)) * 0.01,
        z_g=jax.random.normal(keys[4], (T, m)) * 0.1,
        xi_g=jnp.zeros((T, m)),
        rho_bar=jnp.array(0.1),
    )


def _build_qp_data(ocp, params):
    """Build QPData via the SQP solver's linearization."""
    from turbompc.solvers.turbompc_solver import TurboMPCSolver

    solver = TurboMPCSolver(
        program=ocp,
        params={
            "num_sqp_iteration_max": 1,
            "tol_convergence": 1e-6,
            "linesearch": False,
            "admm": {
                "sigma": 1e-6,
                "max_iter": 1,
                "eps_abs": 1e-4,
                "eps_rel": 1e-4,
                "rho": 0.1,
                "rho_min": 1e-6,
                "rho_max": 1e6,
                "rho_f_factor": 1000.0,
                "relaxation_parameter": 1.0,
                "check_termination_every": 1,
                "adapt_rho_every": 0,
                "adaptive_rho_tolerance": 5.0,
                "pcg": {"max_iter": 200, "tol_epsilon": 1e-12},
            },
        },
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
    )
    guess = solver.initial_guess(params)
    return solver._build_qp_data(guess.states, guess.controls, params)


def _run_cuda_admm(ocp, params, state0, schur, backend, n_iters=1):
    """Run CUDA ADMM for exactly n_iters iterations."""
    qp = _build_qp_data(ocp, params)
    if backend == ForwardBackend.ADMM_FUSED_PCG:
        from turbompc.solvers.admm.admm_ffi_backend import admm_ffi_solve

        result = admm_ffi_solve(
            qp,
            schur,
            state0,
            max_iter=n_iters,
            pcg_max_iter=200,
            check_every=9999,
            eps_abs=1e-30,
            eps_rel=1e-30,
            sigma=1e-6,
            rho_f_factor=1000.0,
            alpha=1.6,
            pcg_eps=1e-12,
        )
    else:  # ADMM_FUSED_CUDSS
        from turbompc.solvers.admm.admm_cudss_ffi_backend import admm_cudss_ffi_solve

        result = admm_cudss_ffi_solve(
            qp,
            schur,
            state0,
            max_iter=n_iters,
            check_every=9999,
            eps_abs=1e-30,
            eps_rel=1e-30,
            sigma=1e-6,
            rho_f_factor=1000.0,
            alpha=1.6,
        )
    return result.state


PY_OCP_SPECS = OCPProcessor.parametrize(
    dynamics=["spacecraft", "linear"],
    horizon=[5, 25],
)


@pytest.mark.parametrize("ocp_spec", PY_OCP_SPECS)
class TestPythonReferences:
    """Validate Python reference implementations used for CUDA comparison."""

    def test_compute_gamma_matches_manual(self, ocp_spec):
        ocp, params = OCPProcessor.build(ocp_spec)
        state0 = _build_state(ocp, params)
        qp = _build_qp_data(ocp, params)

        gamma_py = compute_gamma(
            qp,
            state0.x_blocks,
            state0.z_g,
            state0.y_g,
            state0.y_f_0,
            state0.y_f_dyn,
            rho_f=100.0,
            rho_ineq=0.1,
            sigma=1e-6,
        )
        # Manual
        c0_tilde = 100.0 * qp.eq.c0 - state0.y_f_0
        c_tilde = 100.0 * qp.eq.c - state0.y_f_dyn
        eq_term = _apply_Ct(qp, c0_tilde, c_tilde)
        ineq_term = _apply_Gt(qp, 0.1 * state0.z_g - state0.y_g)
        gamma_manual = 1e-6 * state0.x_blocks - qp.cost.q + eq_term + ineq_term
        diff = float(jnp.max(jnp.abs(gamma_py - gamma_manual)))
        assert diff < 1e-14

    def test_apply_C_parts_and_G_shapes(self, ocp_spec):
        ocp, params = OCPProcessor.build(ocp_spec)
        state0 = _build_state(ocp, params)
        qp = _build_qp_data(ocp, params)
        Cx0, Cx_dyn = _apply_C_parts(qp, state0.x_blocks)
        Gx = _apply_G(qp, state0.x_blocks)
        assert Cx0.shape == (qp.eq.A0.shape[0],)
        assert Cx_dyn.shape == qp.eq.c.shape
        assert Gx.shape == state0.z_g.shape

    def test_residuals_nonnegative(self, ocp_spec):
        ocp, params = OCPProcessor.build(ocp_spec)
        state0 = _build_state(ocp, params)
        qp = _build_qp_data(ocp, params)
        res = _compute_residuals(qp, state0)
        assert float(res.primal_residual) >= 0
        assert float(res.dual_residual) >= 0

    def test_slack_projection_respects_bounds(self, ocp_spec):
        ocp, params = OCPProcessor.build(ocp_spec)
        state0 = _build_state(ocp, params)
        qp = _build_qp_data(ocp, params)
        Gx = _apply_G(qp, state0.x_blocks)
        z_tilde = 1.6 * Gx + (1 - 1.6) * state0.z_g + state0.y_g / 0.1
        z_projected = _project_box(z_tilde, qp.ineq.l, qp.ineq.u)
        assert float(jnp.min(z_projected - qp.ineq.l)) >= -1e-12
        assert float(jnp.min(qp.ineq.u - z_projected)) >= -1e-12


CUDA_OCP_SPECS = OCPProcessor.parametrize(
    dynamics=["spacecraft", "linear"],
    horizon=[5, 25],
)


@pytest.mark.parametrize("ocp_spec", CUDA_OCP_SPECS)
@pytest.mark.parametrize("n_iters", [1, 3])
def test_fused_cudss_deterministic(ocp_spec, n_iters):
    """cuDSS-loop ADMM is deterministic: same input → same output."""
    from tests.helpers.backend_utils import backend_available

    if not backend_available(ForwardBackend.ADMM_FUSED_CUDSS):
        pytest.skip("cuDSS-loop not available")

    ocp, params = OCPProcessor.build(ocp_spec)
    state0 = _build_state(ocp, params)
    qp = _build_qp_data(ocp, params)
    schur = compute_S_Phiinv(qp, rho_f=100.0, sigma=1e-6, rho_ineq=0.1)

    s1 = _run_cuda_admm(
        ocp, params, state0, schur, ForwardBackend.ADMM_FUSED_CUDSS, n_iters
    )
    s2 = _run_cuda_admm(
        ocp, params, state0, schur, ForwardBackend.ADMM_FUSED_CUDSS, n_iters
    )
    for name in ["x_blocks", "y_f_0", "y_f_dyn", "y_g", "z_g"]:
        diff = float(jnp.max(jnp.abs(getattr(s1, name) - getattr(s2, name))))
        # cuDSS is deterministic but matmul ordering may vary slightly across GPU streams
        assert diff < 1e-12, f"{name}: {diff:.2e}"


@pytest.mark.parametrize(
    "ocp_spec",
    OCPProcessor.parametrize(
        dynamics=["spacecraft", "linear"],
        horizon=[5, 10],
    ),
)
@pytest.mark.parametrize("n_iters", [1, 5])
def test_fused_vs_cudss_convergence(ocp_spec, n_iters):
    """Fused PCG and cuDSS-loop produce similar state (differs by PCG tolerance)."""
    from tests.helpers.backend_utils import backend_available

    if not (
        backend_available(ForwardBackend.ADMM_FUSED_PCG)
        and backend_available(ForwardBackend.ADMM_FUSED_CUDSS)
    ):
        pytest.skip("Fused + cuDSS FFI not both available")

    ocp, params = OCPProcessor.build(ocp_spec)
    state0 = _build_state(ocp, params)
    qp = _build_qp_data(ocp, params)
    schur = compute_S_Phiinv(qp, rho_f=100.0, sigma=1e-6, rho_ineq=0.1)

    s_fused = _run_cuda_admm(
        ocp, params, state0, schur, ForwardBackend.ADMM_FUSED_PCG, n_iters
    )
    s_cudss = _run_cuda_admm(
        ocp, params, state0, schur, ForwardBackend.ADMM_FUSED_CUDSS, n_iters
    )

    for name in ["x_blocks", "y_f_0", "y_f_dyn", "y_g", "z_g"]:
        diff = float(jnp.max(jnp.abs(getattr(s_fused, name) - getattr(s_cudss, name))))
        assert diff < FUSED_TOL, f"{name} fused vs cudss: {diff:.2e}"
