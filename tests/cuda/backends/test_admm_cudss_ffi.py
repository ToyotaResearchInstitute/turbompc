"""Tests for the cuDSS-loop ADMM FFI backend (admm_cudss.cu / admm_cudss_ffi.cc)."""
from __future__ import annotations

import ctypes
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.cuda.helpers.processors import OCPProcessor, OCPSpec
from tests.helpers.backend_utils import backend_available
from turbompc.solvers.admm.admm import ADMMState, compute_S_Phiinv
from turbompc.solvers.linear_systems_solvers.pcg_primal import SchurComplementMatrices
from turbompc.solvers.qp_data import (
    QPCostBlocks,
    QPData,
    QPEqualityBlocks,
    QPInequalityBlocks,
)
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)

pytestmark = pytest.mark.skipif(
    not backend_available(ForwardBackend.ADMM_FUSED_CUDSS),
    reason="cuDSS-loop FFI not available",
)


_BASE_KW = dict(
    sigma=1e-6,
    rho_f_factor=1000.0,
    alpha=1.6,
    slack_weight=0.0,
    use_slack=False,
)


def _admm_cudss_backend():
    from turbompc.solvers.admm import admm_cudss_ffi_backend

    return admm_cudss_ffi_backend


def _build_qp(horizon: int):
    spec = OCPSpec(
        ocp_variant="base",
        dynamics="spacecraft",
        horizon=horizon,
        discretization="euler",
        bounds_mode="both",
        state_bound=10.0,
        control_bound=10.0,
    )
    ocp, params = OCPProcessor.build(spec)
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


def _zero_state(qp, rho_bar, batched_Nb=None):
    T = qp.eq.c.shape[0] + 1
    N = T - 1
    n = qp.cost.D.shape[-1]
    m = qp.ineq.G.shape[-2]
    nx = qp.eq.A_minus.shape[-2]
    n0 = qp.eq.A0.shape[0]
    dtype = qp.cost.q.dtype
    if batched_Nb is None:
        return ADMMState(
            x_blocks=jnp.zeros((T, n), dtype=dtype),
            y_g=jnp.zeros((T, m), dtype=dtype),
            y_f_0=jnp.zeros((n0,), dtype=dtype),
            y_f_dyn=jnp.zeros((N, nx), dtype=dtype),
            z_g=jnp.zeros((T, m), dtype=dtype),
            xi_g=jnp.zeros((T, m), dtype=dtype),
            rho_bar=jnp.asarray(rho_bar, dtype=dtype),
        )
    Nb = batched_Nb
    return ADMMState(
        x_blocks=jnp.zeros((Nb, T, n), dtype=dtype),
        y_g=jnp.zeros((Nb, T, m), dtype=dtype),
        y_f_0=jnp.zeros((Nb, n0), dtype=dtype),
        y_f_dyn=jnp.zeros((Nb, N, nx), dtype=dtype),
        z_g=jnp.zeros((Nb, T, m), dtype=dtype),
        xi_g=jnp.zeros((Nb, T, m), dtype=dtype),
        rho_bar=jnp.asarray(rho_bar, dtype=dtype),
    )


def _broadcast_qp(qp, Nb):
    def tile(a):
        a = jnp.asarray(a)
        return jnp.broadcast_to(a[None], (Nb,) + a.shape)

    return QPData(
        cost=QPCostBlocks(D=tile(qp.cost.D), E=tile(qp.cost.E), q=tile(qp.cost.q)),
        eq=QPEqualityBlocks(
            A0=tile(qp.eq.A0),
            A_minus=tile(qp.eq.A_minus),
            A_plus=tile(qp.eq.A_plus),
            c0=tile(qp.eq.c0),
            c=tile(qp.eq.c),
        ),
        ineq=QPInequalityBlocks(
            G=tile(qp.ineq.G),
            l=tile(qp.ineq.l),
            u=tile(qp.ineq.u),
            slack_penalization_weight=qp.ineq.slack_penalization_weight,
            use_slack_variables=qp.ineq.use_slack_variables,
        ),
    )


def _solve(qp, rho_bar, *, max_iter, check_every, adapt_rho_every, eps=1e-4):
    backend = _admm_cudss_backend()
    schur = compute_S_Phiinv(
        qp,
        jnp.asarray(rho_bar * _BASE_KW["rho_f_factor"], dtype=qp.cost.q.dtype),
        jnp.asarray(_BASE_KW["sigma"], dtype=qp.cost.q.dtype),
        rho_ineq=jnp.asarray(rho_bar, dtype=qp.cost.q.dtype),
    )
    state0 = _zero_state(qp, rho_bar)
    return backend.admm_cudss_ffi_solve_single(
        qp,
        schur,
        state0,
        max_iter=max_iter,
        check_every=check_every,
        adapt_rho_every=adapt_rho_every,
        eps_abs=eps,
        eps_rel=eps,
        adaptive_rho_tolerance=5.0,
        rho_min=1e-6,
        rho_max=1e6,
        **_BASE_KW,
    )


def _cache_size() -> int:
    fn = _admm_cudss_backend()._LIB.ADMMCudssCacheSize
    fn.restype = ctypes.c_int
    return int(fn())


def _clear_cache() -> None:
    _admm_cudss_backend()._LIB.ClearADMMCudssCacheImpl()


def test_adapt_rho_disabled_is_identical_to_fixed():
    # cuDSS is not bit-deterministic across calls; 1e-12, not bit-equality.
    qp = _build_qp(horizon=20)
    rho_bar = 0.1

    a = _solve(qp, rho_bar, max_iter=200, check_every=5, adapt_rho_every=0)
    b = _solve(qp, rho_bar, max_iter=200, check_every=5, adapt_rho_every=0)

    np.testing.assert_allclose(np.asarray(a.x_out), np.asarray(b.x_out), atol=1e-12)
    assert int(a.iters_out) == int(b.iters_out)
    np.testing.assert_allclose(
        np.asarray(a.state.rho_bar), np.asarray(b.state.rho_bar), atol=1e-12
    )


def test_adapt_rho_actually_updates_rho():
    qp = _build_qp(horizon=20)
    rho_bar_init = 1e-4  # tiny → primal residuals dominate, rho must grow

    fixed = _solve(qp, rho_bar_init, max_iter=200, check_every=5, adapt_rho_every=0)
    adapt = _solve(qp, rho_bar_init, max_iter=200, check_every=5, adapt_rho_every=10)

    rho_fixed = float(fixed.state.rho_bar)
    rho_adapt = float(adapt.state.rho_bar)

    assert (
        abs(rho_fixed - rho_bar_init) < 1e-12
    ), f"fixed-rho path mutated rho: init={rho_bar_init}, out={rho_fixed}"
    rho_ratio = max(rho_adapt / rho_bar_init, rho_bar_init / rho_adapt)
    assert rho_ratio >= 5.0, (
        f"adaptive rho did not move: init={rho_bar_init}, out={rho_adapt}, "
        f"ratio={rho_ratio:.2f}"
    )


def test_adapt_rho_speeds_up_convergence_when_misturned():
    # Empirical: fixed ~1150 iters, adaptive ~20. eps≤1e-4 required —
    # at 1e-3 fixed-rho can pass tolerance while still ~7% off the minimum.
    qp = _build_qp(horizon=20)
    rho_bar_init = 1e-3  # well-tuned is ~0.1 here

    fixed = _solve(
        qp, rho_bar_init, max_iter=2000, check_every=5, adapt_rho_every=0, eps=1e-4
    )
    adapt = _solve(
        qp, rho_bar_init, max_iter=2000, check_every=5, adapt_rho_every=10, eps=1e-4
    )

    iters_fixed = int(fixed.iters_out)
    iters_adapt = int(adapt.iters_out)
    assert iters_fixed < 2000, f"fixed-rho path did not converge: {iters_fixed} iters"
    assert iters_adapt < 2000, f"adapt-rho path did not converge: {iters_adapt} iters"

    assert iters_adapt * 3 < iters_fixed, (
        f"adaptive rho did not save iterations: fixed={iters_fixed}, "
        f"adaptive={iters_adapt}"
    )

    diff = float(jnp.max(jnp.abs(fixed.x_out - adapt.x_out)))
    norm = float(jnp.max(jnp.abs(fixed.x_out)))
    rel = diff / max(norm, 1e-30)
    assert rel < 5e-3, f"fixed and adaptive solutions disagree: rel diff = {rel:.3e}"


@pytest.mark.parametrize("check_every", [5, 10, 25])
def test_iters_out_matches_buffer_state(check_every):
    backend = _admm_cudss_backend()
    qp = _build_qp(horizon=30)
    rho_bar = 0.1
    schur = compute_S_Phiinv(
        qp,
        jnp.asarray(rho_bar * _BASE_KW["rho_f_factor"], dtype=qp.cost.q.dtype),
        jnp.asarray(_BASE_KW["sigma"], dtype=qp.cost.q.dtype),
        rho_ineq=jnp.asarray(rho_bar, dtype=qp.cost.q.dtype),
    )
    state0 = _zero_state(qp, rho_bar)

    solved = backend.admm_cudss_ffi_solve_single(
        qp,
        schur,
        state0,
        max_iter=200,
        check_every=check_every,
        eps_abs=1e-4,
        eps_rel=1e-3,
        **_BASE_KW,
    )
    K = int(solved.iters_out)
    assert K < 200, "test problem should converge within max_iter"

    oracle = backend.admm_cudss_ffi_solve_single(
        qp,
        schur,
        state0,
        max_iter=K,
        check_every=0,
        eps_abs=1e-30,
        eps_rel=1e-30,
        **_BASE_KW,
    )

    diff_x = float(jnp.max(jnp.abs(solved.x_out - oracle.x_out)))
    diff_z = float(jnp.max(jnp.abs(solved.state.z_g - oracle.state.z_g)))
    diff_y = float(jnp.max(jnp.abs(solved.state.y_g - oracle.state.y_g)))

    norm_x = float(jnp.max(jnp.abs(oracle.x_out)))
    rel = diff_x / max(norm_x, 1e-30)

    assert rel < 1e-4, (
        f"check_every={check_every}: solved.x_out is not the iter-{K} state "
        f"(rel {rel:.3e}, abs {diff_x:.3e}; z_g {diff_z:.3e}; y_g {diff_y:.3e})"
    )


@pytest.mark.parametrize("rhos", [(0.01, 100.0), (0.001, 1000.0)])
def test_batched_rho_applied_per_element(rhos):
    backend = _admm_cudss_backend()
    # Wide rho spread + impossible tolerance + few iters → iterate stays off
    # the (rho-invariant) QP minimum so per-element divergence is detectable.
    qp = _build_qp(horizon=30)
    rho_a, rho_b = rhos
    dtype = qp.cost.q.dtype

    kw_no_conv = dict(
        max_iter=20,
        check_every=0,
        eps_abs=1e-30,
        eps_rel=1e-30,
        **_BASE_KW,
    )

    def run_single(rho):
        schur = compute_S_Phiinv(
            qp,
            jnp.asarray(rho * _BASE_KW["rho_f_factor"], dtype=dtype),
            jnp.asarray(_BASE_KW["sigma"], dtype=dtype),
            rho_ineq=jnp.asarray(rho, dtype=dtype),
        )
        state0 = _zero_state(qp, rho)
        return backend.admm_cudss_ffi_solve_single(qp, schur, state0, **kw_no_conv)

    r_a = run_single(rho_a)
    r_b = run_single(rho_b)

    diff_per_rho = float(jnp.max(jnp.abs(r_a.x_out - r_b.x_out)))
    assert diff_per_rho > 1e-4, (
        "test problem is degenerate: x_out is rho-invariant (||x_a -"
        f" x_b||={diff_per_rho:.2e})"
    )

    rhos_arr = jnp.asarray([rho_a, rho_b], dtype=dtype)

    def per_rho_S(rho):
        s = compute_S_Phiinv(
            qp,
            rho * jnp.asarray(_BASE_KW["rho_f_factor"], dtype=dtype),
            jnp.asarray(_BASE_KW["sigma"], dtype=dtype),
            rho_ineq=rho,
        )
        return s.S, s.preconditioner_Phiinv

    S_b, Phi_b = jax.vmap(per_rho_S)(rhos_arr)
    schur_batched = SchurComplementMatrices(S=S_b, preconditioner_Phiinv=Phi_b)
    qp_batched = _broadcast_qp(qp, Nb=2)
    state0_batched = _zero_state(qp, rhos_arr, batched_Nb=2)

    r_batched = backend.admm_cudss_ffi_solve(
        qp_batched, schur_batched, state0_batched, **kw_no_conv
    )

    diff_0 = float(jnp.max(jnp.abs(r_batched.x_out[0] - r_a.x_out)))
    diff_1 = float(jnp.max(jnp.abs(r_batched.x_out[1] - r_b.x_out)))

    tol = 1e-5
    assert (
        diff_0 < tol
    ), f"rhos={rhos}: bid=0 batched vs single-call diff = {diff_0:.3e}"
    assert diff_1 < tol, (
        f"rhos={rhos}: bid=1 batched vs single-call diff = {diff_1:.3e} — "
        f"kernel likely used rho_bar_init[0]={rho_a} for bid=1"
    )


def test_all_cudss_calls_are_status_checked():
    # Source-grep lint: cuDSS GENERAL silently produces NaN/garbage on bad
    # numerics rather than returning a status, so a runtime test can't catch
    # an unwrapped call — only a structural check can.
    cu_path = (
        Path(__file__).resolve().parents[3] / "turbompc/solvers/admm/csrc/admm_cudss.cu"
    )
    src = cu_path.read_text()

    cudss_call_pattern = re.compile(r"\bcudss[A-Z][a-zA-Z]*\s*\(")
    cudss_destroy_pattern = re.compile(
        r"\bcudss(?:Destroy|MatrixDestroy|DataDestroy|ConfigDestroy)\s*\("
    )
    unwrapped: list[tuple[int, str]] = []
    for lineno, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        if (
            stripped.startswith("//")
            or stripped.startswith("*")
            or stripped.startswith("#")
        ):
            continue
        if "cudssStatus_t" in line or "cudssHandle_t" in line:
            continue
        if not cudss_call_pattern.search(line):
            continue
        if "CUDSS_CHECK" in line:
            continue
        # Destroy* calls live in the noexcept destructor; intentional.
        if cudss_destroy_pattern.search(line):
            continue
        unwrapped.append((lineno, line.rstrip()))

    assert not unwrapped, (
        "cuDSS calls must be wrapped in CUDSS_CHECK(...). Unwrapped calls:\n"
        + "\n".join(f"  {cu_path}:{n}: {ln}" for n, ln in unwrapped)
    )

    cc_path = (
        Path(__file__).resolve().parents[3]
        / "turbompc/solvers/admm/csrc/admm_cudss_ffi.cc"
    )
    cc_src = cc_path.read_text()
    assert "try {" in cc_src and "catch (const std::exception" in cc_src, (
        f"Expected try/catch around the Launch* call in {cc_path}; without it "
        "a CUDSS_CHECK throw would crash the process instead of becoming an "
        "FFI-level error."
    )
    assert "ffi::Error::Internal(e.what())" in cc_src, (
        f"Expected the catch handler in {cc_path} to translate exceptions to "
        "ffi::Error::Internal."
    )


def test_clear_cache_releases_cuDSS_entries():
    _clear_cache()  # other tests may have populated entries
    assert _cache_size() == 0, "clear_cache did not empty the map at test entry"

    horizons = [12, 24, 36]
    for T in horizons:
        qp = _build_qp(horizon=T)
        out = _solve(qp, 0.1, max_iter=5, check_every=5, adapt_rho_every=0)
        out.x_out.block_until_ready()

    populated = _cache_size()
    assert populated == len(horizons), (
        f"expected {len(horizons)} cache entries after solves at "
        f"{horizons}, got {populated}"
    )

    _clear_cache()
    assert _cache_size() == 0, "clear_cache failed to remove entries"


def test_no_rho_update_on_converged_iteration():
    backend = _admm_cudss_backend()
    # eps=1e10 forces convergence on iter 1; mis-tuned rho + tight gate forces
    # the rho-update to fire — so both events collide on the break iteration.
    qp = _build_qp(horizon=20)
    rho_bar = 1e-6  # far from optimal (~0.1)

    schur = compute_S_Phiinv(
        qp,
        jnp.asarray(rho_bar * _BASE_KW["rho_f_factor"], dtype=qp.cost.q.dtype),
        jnp.asarray(_BASE_KW["sigma"], dtype=qp.cost.q.dtype),
        rho_ineq=jnp.asarray(rho_bar, dtype=qp.cost.q.dtype),
    )
    state0 = _zero_state(qp, rho_bar)

    out = backend.admm_cudss_ffi_solve_single(
        qp,
        schur,
        state0,
        max_iter=10,
        check_every=1,
        adapt_rho_every=1,
        eps_abs=1e10,
        eps_rel=1e10,
        adaptive_rho_tolerance=1.5,
        rho_min=1e-12,
        rho_max=1e12,
        **_BASE_KW,
    )

    iters = int(out.iters_out)
    assert (
        1 <= iters <= 5
    ), f"expected near-immediate convergence (iters in [1,5]); got iters={iters}"

    rho_out = float(out.state.rho_bar)
    assert rho_out == rho_bar, (
        f"rho_bar_out={rho_out} != rho_bar_init={rho_bar}: rho-update fired "
        "on the converged iteration and leaked into the output"
    )
