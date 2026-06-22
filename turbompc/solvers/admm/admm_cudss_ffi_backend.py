"""ADMM solver with cuDSS direct solve via CUDA FFI (host-side ADMM loop)."""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
from turbompc.solvers.admm.admm import ADMMSolveResult, ADMMState
from turbompc.solvers.linear_systems_solvers.pcg_primal import SchurComplementMatrices
from turbompc.solvers.qp_data import QPData


def _find_lib() -> Path:
    turbompc_root = Path(__file__).resolve().parents[2]
    candidates = [
        turbompc_root / "solvers" / "csrc" / "build" / "libadmm_cudss_ffi.so",
        turbompc_root.parent / "build" / "ffi" / "libadmm_cudss_ffi.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find libadmm_cudss_ffi.so. Build first:\n"
        "  cmake -S turbompc/solvers/csrc -B build/ffi -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build build/ffi -j\n"
    )


_LIB = ctypes.cdll.LoadLibrary(str(_find_lib()))

jax.ffi.register_ffi_target(
    "admm_cudss_cuda",
    jax.ffi.pycapsule(_LIB.AdmmCudssCuda),
    platform="CUDA",
)
jax.ffi.register_ffi_target(
    "admm_cudss_cuda_f64",
    jax.ffi.pycapsule(_LIB.AdmmCudssCudaF64),
    platform="CUDA",
)

# Configure the cache-control entry points (extern "C" — not part of the
# JAX FFI traced graph; called directly via ctypes for cleanup/diagnostics).
_LIB.ClearADMMCudssCacheImpl.restype = None
_LIB.ADMMCudssCacheSize.restype = ctypes.c_int


def clear_cache() -> None:
    """Drop every cached cuDSS handle + GPU buffer held by the cuDSS-loop
    ADMM kernel.

    The kernel keeps a process-global ``unordered_map`` of resources keyed by
    problem dimensions; without this call, sweeping multiple horizons in one
    process accumulates entries forever. Safe to call between solves; the
    next solve re-creates whatever it needs.
    """
    _LIB.ClearADMMCudssCacheImpl()


def admm_cudss_ffi_solve(
    qp_data: QPData,
    schur: SchurComplementMatrices,
    state0: ADMMState,
    *,
    max_iter: int = 100,
    check_every: int = 5,
    adapt_rho_every: int = 0,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-3,
    sigma: float = 1e-6,
    rho_f_factor: float = 1000.0,
    alpha: float = 1.6,
    adaptive_rho_tolerance: float = 5.0,
    rho_min: float = 1e-6,
    rho_max: float = 1e6,
    slack_weight: float = 0.0,
    use_slack: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, ADMMState]:
    """Solve ADMM QP via host-side loop + cuDSS direct solve.

    Args:
        qp_data: QP problem data (cost, equality, inequality blocks)
        schur: Precomputed Schur complement matrices (S used for cuDSS)
        state0: ADMM warm-start state
        max_iter, check_every, etc: ADMM configuration

    Returns:
        x_out: primal solution (Nb, T, n) or (T, n)
        iters_out: iteration counts (Nb,) or ()
        state_out: final ADMM state for warm-starting
    """
    dtype = qp_data.cost.q.dtype
    S = schur.S

    # Detect batched vs unbatched from S shape
    batched = S.ndim == 4

    if batched:
        Nb, T, n, _ = S.shape
    else:
        T, n, _ = S.shape
        Nb = 1
        S = S[None]

    # Extract dimensions
    n0 = qp_data.eq.A0.shape[
        -2
    ]  # initial constraint rows (nx+nu when constrain_initial_control)
    nx_dyn = qp_data.eq.A_minus.shape[-2]  # dynamics rows (always true nx)
    m = qp_data.ineq.G.shape[-2]

    # Determine FFI target
    if dtype == jnp.float64:
        kernel_dtype = jnp.float64
        ffi_target = "admm_cudss_cuda_f64"
    else:
        kernel_dtype = jnp.float32
        ffi_target = "admm_cudss_cuda"

    N = T - 1

    def _ensure(arr, shape, name=""):
        """Ensure array has correct dtype and batch dimension."""
        a = jnp.asarray(arr, dtype=kernel_dtype)
        if not batched and a.ndim == len(shape) - 1:
            a = a[None]
        return a

    # QP data arrays — add batch dim if needed (no Phiinv)
    S_k = jnp.asarray(S, dtype=kernel_dtype)
    D_k = _ensure(qp_data.cost.D, (Nb, T, n, n), "D")
    E_k = _ensure(qp_data.cost.E, (Nb, N, n, n), "E")
    q_k = _ensure(qp_data.cost.q, (Nb, T, n), "q")
    A0_k = _ensure(qp_data.eq.A0, (Nb, n0, n), "A0")
    Am_k = _ensure(qp_data.eq.A_minus, (Nb, N, nx_dyn, n), "A_minus")
    Ap_k = _ensure(qp_data.eq.A_plus, (Nb, N, nx_dyn, n), "A_plus")
    G_k = _ensure(qp_data.ineq.G, (Nb, T, m, n), "G")
    l_k = _ensure(qp_data.ineq.l, (Nb, T, m), "l")
    u_k = _ensure(qp_data.ineq.u, (Nb, T, m), "u")
    c0_k = _ensure(qp_data.eq.c0, (Nb, n0), "c0")
    c_k = _ensure(qp_data.eq.c, (Nb, N, nx_dyn), "c")

    # Warm-start state — add batch dim if needed
    x0_k = _ensure(state0.x_blocks, (Nb, T, n))
    z_g0_k = _ensure(state0.z_g, (Nb, T, m))
    y_g0_k = _ensure(state0.y_g, (Nb, T, m))
    y_f_0_k = _ensure(state0.y_f_0, (Nb, n0))
    y_f_dyn_k = _ensure(state0.y_f_dyn, (Nb, N, nx_dyn))
    xi_g0_k = _ensure(state0.xi_g, (Nb, T, m))
    rho_bar_k = jnp.asarray(state0.rho_bar, dtype=kernel_dtype)
    if rho_bar_k.ndim == 0:
        rho_bar_k = rho_bar_k[None]
    slack_weight_k = jnp.asarray(slack_weight, dtype=kernel_dtype)

    # Output shapes
    out_shapes = [
        jax.ShapeDtypeStruct((Nb, T, n), kernel_dtype),  # x_out
        jax.ShapeDtypeStruct((Nb,), jnp.uint32),  # iters_out
        jax.ShapeDtypeStruct((Nb, T, n), kernel_dtype),  # x_blocks_out
        jax.ShapeDtypeStruct((Nb, T, m), kernel_dtype),  # z_g_out
        jax.ShapeDtypeStruct((Nb, T, m), kernel_dtype),  # y_g_out
        jax.ShapeDtypeStruct((Nb, n0), kernel_dtype),  # y_f_0_out
        jax.ShapeDtypeStruct((Nb, N, nx_dyn), kernel_dtype),  # y_f_dyn_out
        jax.ShapeDtypeStruct((Nb, T, m), kernel_dtype),  # xi_g_out
        jax.ShapeDtypeStruct((Nb,), kernel_dtype),  # rho_bar_out
        jax.ShapeDtypeStruct((Nb,), kernel_dtype),  # kernel_ns_out
    ]

    call = jax.ffi.ffi_call(
        ffi_target,
        out_shapes,
        vmap_method="broadcast_all",
    )

    results = call(
        # QP data (12 args — no Phiinv)
        S_k,
        D_k,
        E_k,
        q_k,
        A0_k,
        Am_k,
        Ap_k,
        G_k,
        l_k,
        u_k,
        c0_k,
        c_k,
        # Warm-start (7 args)
        x0_k,
        z_g0_k,
        y_g0_k,
        y_f_0_k,
        y_f_dyn_k,
        xi_g0_k,
        rho_bar_k,
        # slack_weight as buffer input
        slack_weight_k,
        # Config attrs (no pcg_max_iter, pcg_eps)
        max_iter=int(max_iter),
        check_every=int(check_every),
        adapt_rho_every=int(adapt_rho_every),
        eps_abs=float(eps_abs),
        eps_rel=float(eps_rel),
        sigma=float(sigma),
        rho_f_factor=float(rho_f_factor),
        alpha=float(alpha),
        adaptive_rho_tolerance=float(adaptive_rho_tolerance),
        rho_min=float(rho_min),
        rho_max=float(rho_max),
        nx=int(nx_dyn),
        n0=int(n0),
        m=int(m),
        use_slack=int(use_slack),
    )

    (
        x_out,
        iters_out,
        x_blocks_out,
        z_g_out,
        y_g_out,
        y_f_0_out,
        y_f_dyn_out,
        xi_g_out,
        rho_bar_out,
        kernel_ns_out,
    ) = results

    # Remove batch dim if input was unbatched
    if not batched:
        x_out = x_out[0]
        iters_out = iters_out[0]
        x_blocks_out = x_blocks_out[0]
        z_g_out = z_g_out[0]
        y_g_out = y_g_out[0]
        y_f_0_out = y_f_0_out[0]
        y_f_dyn_out = y_f_dyn_out[0]
        xi_g_out = xi_g_out[0]
        rho_bar_out = rho_bar_out[0]
        kernel_ns_out = kernel_ns_out[0]

    # Cast back to original dtype
    x_out = jnp.asarray(x_out, dtype=dtype)

    state_out = ADMMState(
        x_blocks=jnp.asarray(x_blocks_out, dtype=dtype),
        y_g=jnp.asarray(y_g_out, dtype=dtype),
        y_f_0=jnp.asarray(y_f_0_out, dtype=dtype),
        y_f_dyn=jnp.asarray(y_f_dyn_out, dtype=dtype),
        z_g=jnp.asarray(z_g_out, dtype=dtype),
        xi_g=jnp.asarray(xi_g_out, dtype=dtype),
        rho_bar=jnp.asarray(rho_bar_out, dtype=dtype),
    )

    return ADMMSolveResult(x_out, iters_out, state_out, kernel_ns_out)


def admm_cudss_ffi_solve_single(
    qp_data: QPData,
    schur: SchurComplementMatrices,
    state0: ADMMState,
    *,
    max_iter: int = 100,
    check_every: int = 5,
    adapt_rho_every: int = 0,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-3,
    sigma: float = 1e-6,
    rho_f_factor: float = 1000.0,
    alpha: float = 1.6,
    adaptive_rho_tolerance: float = 5.0,
    rho_min: float = 1e-6,
    rho_max: float = 1e6,
    slack_weight: float = 0.0,
    use_slack: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, ADMMState]:
    """ADMM-cuDSS solve for a single (unbatched) problem, vmap-compatible."""
    dtype = qp_data.cost.q.dtype
    S = schur.S  # (T, n, 3n)

    T, n, _ = S.shape
    N = T - 1
    n0 = qp_data.eq.A0.shape[
        -2
    ]  # initial constraint rows (nx+nu when constrain_initial_control)
    nx_dyn = qp_data.eq.A_minus.shape[-2]  # dynamics rows (always true nx)
    m = qp_data.ineq.G.shape[-2]

    if dtype == jnp.float64:
        kernel_dtype = jnp.float64
        ffi_target = "admm_cudss_cuda_f64"
    else:
        kernel_dtype = jnp.float32
        ffi_target = "admm_cudss_cuda"

    def _cast(arr):
        return jnp.asarray(arr, dtype=kernel_dtype)

    # Pass unbatched (rank-3) arrays; vmap+broadcast_all adds batch dim -> rank-4
    S_k = _cast(S)
    D_k = _cast(qp_data.cost.D)
    E_k = _cast(qp_data.cost.E)
    q_k = _cast(qp_data.cost.q)
    A0_k = _cast(qp_data.eq.A0)
    Am_k = _cast(qp_data.eq.A_minus)
    Ap_k = _cast(qp_data.eq.A_plus)
    G_k = _cast(qp_data.ineq.G)
    l_k = _cast(qp_data.ineq.l)
    u_k = _cast(qp_data.ineq.u)
    c0_k = _cast(qp_data.eq.c0)
    c_k = _cast(qp_data.eq.c)

    x0_k = _cast(state0.x_blocks)
    z_g0_k = _cast(state0.z_g)
    y_g0_k = _cast(state0.y_g)
    y_f_0_k = _cast(state0.y_f_0)
    y_f_dyn_k = _cast(state0.y_f_dyn)
    xi_g0_k = _cast(state0.xi_g)
    rho_bar_k = _cast(state0.rho_bar)
    slack_weight_k = jnp.asarray(slack_weight, dtype=kernel_dtype)

    # Unbatched output shapes; C++ sees rank-3 (Nb=1), vmap maps to rank-4
    out_shapes = [
        jax.ShapeDtypeStruct((T, n), kernel_dtype),  # x_out
        jax.ShapeDtypeStruct((), jnp.uint32),  # iters_out
        jax.ShapeDtypeStruct((T, n), kernel_dtype),  # x_blocks_out
        jax.ShapeDtypeStruct((T, m), kernel_dtype),  # z_g_out
        jax.ShapeDtypeStruct((T, m), kernel_dtype),  # y_g_out
        jax.ShapeDtypeStruct((n0,), kernel_dtype),  # y_f_0_out
        jax.ShapeDtypeStruct((N, nx_dyn), kernel_dtype),  # y_f_dyn_out
        jax.ShapeDtypeStruct((T, m), kernel_dtype),  # xi_g_out
        jax.ShapeDtypeStruct((), kernel_dtype),  # rho_bar_out
        jax.ShapeDtypeStruct((), kernel_dtype),  # kernel_ns_out
    ]

    call = jax.ffi.ffi_call(
        ffi_target,
        out_shapes,
        vmap_method="broadcast_all",
    )

    results = call(
        S_k,
        D_k,
        E_k,
        q_k,
        A0_k,
        Am_k,
        Ap_k,
        G_k,
        l_k,
        u_k,
        c0_k,
        c_k,
        x0_k,
        z_g0_k,
        y_g0_k,
        y_f_0_k,
        y_f_dyn_k,
        xi_g0_k,
        rho_bar_k,
        slack_weight_k,
        max_iter=int(max_iter),
        check_every=int(check_every),
        adapt_rho_every=int(adapt_rho_every),
        eps_abs=float(eps_abs),
        eps_rel=float(eps_rel),
        sigma=float(sigma),
        rho_f_factor=float(rho_f_factor),
        alpha=float(alpha),
        adaptive_rho_tolerance=float(adaptive_rho_tolerance),
        rho_min=float(rho_min),
        rho_max=float(rho_max),
        nx=int(nx_dyn),
        n0=int(n0),
        m=int(m),
        use_slack=int(use_slack),
    )

    (
        x_out,
        iters_out,
        x_blocks_out,
        z_g_out,
        y_g_out,
        y_f_0_out,
        y_f_dyn_out,
        xi_g_out,
        rho_bar_out,
        kernel_ns_out,
    ) = results

    x_out = jnp.asarray(x_out, dtype=dtype)
    state_out = ADMMState(
        x_blocks=jnp.asarray(x_blocks_out, dtype=dtype),
        y_g=jnp.asarray(y_g_out, dtype=dtype),
        y_f_0=jnp.asarray(y_f_0_out, dtype=dtype),
        y_f_dyn=jnp.asarray(y_f_dyn_out, dtype=dtype),
        z_g=jnp.asarray(z_g_out, dtype=dtype),
        xi_g=jnp.asarray(xi_g_out, dtype=dtype),
        rho_bar=jnp.asarray(rho_bar_out, dtype=dtype),
    )
    return ADMMSolveResult(x_out, iters_out, state_out, kernel_ns_out)
