"""Backward cuDSS sparse-KKT analysis-cache eviction.

`g_analysis_cache` (cudss_sparse_kkt.cu) is a static map keyed by
(n, nnz, dtype): one cudssData_t (symbolic factorization + GPU workspace)
per sparsity pattern, with no destructor/eviction. A long-lived process that
solves many distinct shapes accumulates GPU memory forever.

`clear_analysis_cache()` (added alongside `analysis_cache_size()`) must
destroy every cached object and empty the map. Mirror of the forward
`test_clear_cache_releases_cuDSS_entries` for `g_admm_cache`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from tests.cuda.helpers.processors import OCPProcessor, OCPSpec
from tests.helpers.backend_utils import backend_available
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)

pytestmark = pytest.mark.skipif(
    not (
        backend_available(ForwardBackend.ADMM_FUSED_CUDSS)
        and backend_available(BackwardBackend.DIRECT_CUDSS_FFI)
    ),
    reason="cuDSS-loop and DIRECT_CUDSS_FFI backends are required",
)


def _cache_helpers():
    from turbompc.solvers.backward.backward_kkt_cudss_ffi import (
        analysis_cache_size,
        clear_analysis_cache,
    )

    return analysis_cache_size, clear_analysis_cache


_SOLVER_PARAMS = {
    "num_sqp_iteration_max": 1,
    "tol_convergence": 1e-6,
    "linesearch": False,
    "admm": {
        "sigma": 1e-6,
        "max_iter": 50,
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
}


def _run_backward_grad(horizon: int) -> None:
    """One gradient through the cuDSS backward sparse-KKT path → populates
    g_analysis_cache for this problem shape."""
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
        params=_SOLVER_PARAMS,
        forward_backend=ForwardBackend.ADMM_FUSED_CUDSS,
        backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
    )
    guess = solver.initial_guess(params)
    diff_fn = solver.get_differentiable_solve_function()
    weights = {
        "weights_penalization_reference_state_trajectory": jnp.asarray(
            params["weights_penalization_reference_state_trajectory"]
        ),
    }

    def loss(w):
        return jnp.sum(diff_fn(guess, params, w).states ** 2)

    g = jax.grad(loss)(weights)
    jax.block_until_ready(g)


def test_clear_analysis_cache_releases_cudss_entries():
    analysis_cache_size, clear_analysis_cache = _cache_helpers()

    # Clean slate (other tests may have populated entries).
    clear_analysis_cache()
    assert analysis_cache_size() == 0, "clear_analysis_cache did not empty the map"

    for h in (5, 8):
        _run_backward_grad(h)

    populated = analysis_cache_size()
    assert (
        populated >= 1
    ), f"backward cuDSS solves did not populate g_analysis_cache (size={populated})"

    clear_analysis_cache()
    assert analysis_cache_size() == 0, "clear_analysis_cache failed to remove entries"
