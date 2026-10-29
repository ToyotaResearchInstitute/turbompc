"""Per-function CUDA vs Python consistency tests (processor-based).

Each test validates shared CUDA device functions (admm_math.cuh) against
Python reference implementations (admm.py).

Uses OCPProcessor/SolverProcessor (tests/cuda/helpers/processors.py)
to parametrize across dynamics × horizons × variants × backends.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.cuda.helpers.processors import OCPProcessor
from tests.helpers.backend_utils import backend_param
from tests.helpers.constraint_handling_fixtures import (
    BARRIER_KAPPA,
    HANDLINGS,
    QP_PCG_PARAMS,
    SLACK_WEIGHT,
    make_qp,
)
from tests.helpers.solver_fixtures import build_gauss_newton_qp
from turbompc.solvers.admm import ADMMSolver
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
from turbompc.solvers.constraint_handling import ConstraintHandling
from turbompc.solvers.linear_systems_solvers.backends import (
    AdmmBackend,
    SchurSolverBackend,
)
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver
from turbompc.solvers.status import ADMMStatus
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
            "linesearch": {"method": "none"},
            "constraint_handling": "projection",
            "log_barrier_kappa": 0.0,
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
    return build_gauss_newton_qp(solver, guess.states, guess.controls, params)


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
            check_termination_every=9999,
            eps_abs=1e-30,
            eps_rel=1e-30,
            sigma=1e-6,
            rho_f_factor=1000.0,
            alpha=1.6,
            pcg_eps=1e-12,
        )
    else:  # ADMM_FUSED_CUDSS
        from turbompc.solvers.admm.admm_cudss_ffi_backend import (
            AdmmCudssWorkspace,
            admm_cudss_ffi_solve,
        )

        workspace = AdmmCudssWorkspace()

        result = admm_cudss_ffi_solve(
            qp,
            schur,
            state0,
            workspace=workspace,
            max_iter=n_iters,
            check_termination_every=9999,
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
    OCPProcessor.parametrize(dynamics=["spacecraft"], horizon=[5]),
)
@pytest.mark.parametrize("batch_shape", [(1,), (2,), (2, 1)])
def test_fused_pcg_custom_vmap_batches_natively(ocp_spec, batch_shape):
    """Nested vmap axes are flattened into one native fused-PCG call."""
    from tests.helpers.backend_utils import backend_available
    from turbompc.solvers.admm.admm_ffi_backend import admm_ffi_solve

    if not backend_available(ForwardBackend.ADMM_FUSED_PCG):
        pytest.skip("fused PCG FFI not available")

    ocp, params = OCPProcessor.build(ocp_spec)
    state0 = _build_state(ocp, params)
    qp = _build_qp_data(ocp, params)
    schur = compute_S_Phiinv(qp, rho_f=100.0, sigma=1e-6, rho_ineq=0.1)

    def solve_one(state):
        return admm_ffi_solve(
            qp,
            schur,
            state,
            max_iter=2,
            pcg_max_iter=200,
            check_termination_every=0,
            sigma=1e-6,
            rho_f_factor=1000.0,
            alpha=1.6,
            pcg_eps=1e-12,
        )

    expected = solve_one(state0)
    batched_state = jax.tree.map(
        lambda value: jnp.broadcast_to(value, batch_shape + value.shape), state0
    )
    solve_batched = solve_one
    for _ in batch_shape:
        solve_batched = jax.vmap(solve_batched)
    actual = jax.jit(solve_batched)(batched_state)
    jax.block_until_ready(actual)

    expected_x = jnp.broadcast_to(
        expected.state.x_blocks, batch_shape + expected.state.x_blocks.shape
    )
    assert actual.stats.num_iter.shape == batch_shape
    assert jnp.allclose(actual.state.x_blocks, expected_x, atol=FUSED_TOL)


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


# ---------------------------------------------------------------------------
# Fused ADMM vs the JAX loop through the public ADMMSolver, for both
# inequality handlings.
# ---------------------------------------------------------------------------


class _ParityConfig(NamedTuple):
    admm_backend: AdmmBackend
    pcg_params: dict
    # The fused-PCG residual floor limits its termination tolerance.
    eps: float
    # (state, residual) tolerances after one iteration and at convergence.
    one_iteration_atol: tuple
    converged_atol: tuple
    # PCG inexactness can shift the terminating iteration.
    same_num_iter: bool


_PARITY_CONFIGS = {
    ForwardBackend.ADMM_FUSED_CUDSS: _ParityConfig(
        AdmmBackend.FUSED_CUDSS, QP_PCG_PARAMS, 1e-10, (1e-9, 1e-9), (1e-9, 1e-9), True
    ),
    ForwardBackend.ADMM_FUSED_PCG: _ParityConfig(
        AdmmBackend.FUSED_PCG,
        {"max_iter": 500, "tol_epsilon": 1e-14},
        1e-6,
        (1e-6, 1e-7),
        (2e-5, 1e-5),
        False,
    ),
}
_FUSED_BACKENDS = [backend_param(backend) for backend in _PARITY_CONFIGS]


def _solve_both(
    config,
    qp_data,
    zshape,
    constraint_handling,
    log_barrier_kappa,
    *,
    use_slack=False,
    max_iter=300,
    admm_state0=None,
):
    """Return ``(jax_loop_result, fused_result)`` for the same QP.

    Rho is fixed: adaptive rho is a thresholded decision, so a roundoff-level
    difference between Schur solves can change its trajectory.
    """

    def solve(admm_backend):
        solver = ADMMSolver(
            zshape=zshape,
            schur_solver=make_schur_solver(
                SchurSolverBackend.JAX_DENSE,
                zshape.horizon,
                zshape.num_states,
                zshape.num_controls,
                pcg_params=config.pcg_params,
            ),
            pcg_params=config.pcg_params,
            sigma=1.0e-8,
            max_iter=max_iter,
            eps_abs=config.eps,
            eps_rel=config.eps * 10.0,
            adapt_rho_every=0,
            admm_backend=admm_backend,
            use_slack_variables=use_slack,
            constraint_handling=constraint_handling,
            log_barrier_kappa=log_barrier_kappa,
        )
        return solver.solve(qp_data=qp_data, admm_state0=admm_state0, rho_bar=1.0)

    return solve(AdmmBackend.JAX_LOOP), solve(config.admm_backend)


def _assert_results_match(expected, actual, atol, *, same_num_iter=True):
    state_atol, residual_atol = atol
    for field in expected.state._fields:
        np.testing.assert_allclose(
            getattr(actual.state, field),
            getattr(expected.state, field),
            atol=state_atol,
            rtol=0.0,
            err_msg=field,
        )
    assert int(actual.stats.status) == int(expected.stats.status)
    if same_num_iter:
        assert int(actual.stats.num_iter) == int(expected.stats.num_iter)
    # Residuals are differences of like-sized terms, so cancellation makes them
    # less accurate than the state they are computed from.
    for field in ("primal_residual", "dual_residual"):
        np.testing.assert_allclose(
            getattr(actual.stats, field),
            getattr(expected.stats, field),
            atol=residual_atol,
            rtol=0.0,
            err_msg=field,
        )


def _parity_qp(bounds, use_slack, **kwargs):
    return make_qp(
        bounds, slack_penalization_weight=SLACK_WEIGHT if use_slack else 0.0, **kwargs
    )


# The "mixed" QP has two-sided, lower-only, upper-only and unbounded rows.


def _warm_iteration(
    config, constraint_handling, log_barrier_kappa, use_slack, dtype=jnp.float64
):
    """One fused and one JAX-loop iteration from the same nonzero state.

    Starting from a partial JAX-loop solve makes the kernels read a nonzero
    ``z_g``, ``xi_g`` and ``y_g``, which a cold start would not exercise.
    """
    qp_data, zshape = _parity_qp("mixed", use_slack, dtype=dtype)
    kwargs = dict(use_slack=use_slack)
    partial, _ = _solve_both(
        config,
        qp_data,
        zshape,
        constraint_handling,
        log_barrier_kappa,
        max_iter=5,
        **kwargs,
    )
    return _solve_both(
        config,
        qp_data,
        zshape,
        constraint_handling,
        log_barrier_kappa,
        max_iter=1,
        admm_state0=partial.state,
        **kwargs,
    )


@pytest.mark.parametrize("backend", _FUSED_BACKENDS)
@pytest.mark.parametrize(("constraint_handling", "log_barrier_kappa"), HANDLINGS)
@pytest.mark.parametrize("use_slack", [False, True])
def test_fused_iteration_matches_the_jax_loop(
    backend, constraint_handling, log_barrier_kappa, use_slack
):
    config = _PARITY_CONFIGS[backend]
    expected, actual = _warm_iteration(
        config, constraint_handling, log_barrier_kappa, use_slack
    )

    assert int(expected.stats.status) == int(ADMMStatus.MAX_ITER)
    _assert_results_match(expected, actual, config.one_iteration_atol)


@pytest.mark.parametrize("backend", _FUSED_BACKENDS)
@pytest.mark.parametrize(("constraint_handling", "log_barrier_kappa"), HANDLINGS)
def test_fused_converged_solution_matches_the_jax_loop(
    backend, constraint_handling, log_barrier_kappa
):
    """Termination and residuals agree, not only the update."""
    config = _PARITY_CONFIGS[backend]
    qp_data, zshape = _parity_qp("mixed", use_slack=True)
    expected, actual = _solve_both(
        config, qp_data, zshape, constraint_handling, log_barrier_kappa, use_slack=True
    )

    assert int(expected.stats.status) == int(ADMMStatus.SOLVED)
    _assert_results_match(
        expected, actual, config.converged_atol, same_num_iter=config.same_num_iter
    )


@pytest.mark.parametrize("backend", _FUSED_BACKENDS)
def test_fused_log_barrier_float32_matches_the_jax_loop(backend):
    """The device-side barrier Newton iteration in single precision."""
    config = _PARITY_CONFIGS[backend]
    expected, actual = _warm_iteration(
        config, ConstraintHandling.LOG_BARRIER, BARRIER_KAPPA, True, dtype=jnp.float32
    )

    for field in expected.state._fields:
        assert getattr(actual.state, field).dtype == jnp.float32, field
    _assert_results_match(expected, actual, (5.0e-4, 1.0e-4))
