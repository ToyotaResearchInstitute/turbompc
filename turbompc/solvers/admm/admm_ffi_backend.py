"""Fused ADMM solver via CUDA FFI mega-kernel."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from turbompc.solvers.admm.admm import ADMMState, ADMMStats, FusedADMMSolveResult
from turbompc.solvers.constraint_handling import ConstraintHandling
from turbompc.solvers.ffi_loader import load_ffi_lib
from turbompc.solvers.linear_systems_solvers.pcg_primal import SchurComplementMatrices
from turbompc.solvers.qp_data import QPData

# ADMM_FFI_LIB allows overriding with the profiling variant:
#   ADMM_FFI_LIB=build/ffi/libadmm_fused_profile_ffi.so python ...
_LIB_NAME = "libadmm_fused_ffi.so"
_LIB = load_ffi_lib(_LIB_NAME, env_var="ADMM_FFI_LIB")

jax.ffi.register_ffi_target(
    "admm_fused_cuda",
    jax.ffi.pycapsule(_LIB.AdmmFusedCuda),
    platform="CUDA",
)
jax.ffi.register_ffi_target(
    "admm_fused_cuda_f64",
    jax.ffi.pycapsule(_LIB.AdmmFusedCudaF64),
    platform="CUDA",
)


def _admm_ffi_solve_impl(
    qp_data: QPData,
    schur: SchurComplementMatrices,
    state0: ADMMState,
    *,
    max_iter: int = 100,
    pcg_max_iter: int = 200,
    check_termination_every: int = 5,
    check_infeasibility_every: int = 25,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-3,
    eps_prim_inf: float = 1e-4,
    eps_dual_inf: float = 1e-4,
    sigma: float = 1e-6,
    rho_f_factor: float = 1000.0,
    alpha: float = 1.6,
    pcg_eps: float = 1e-8,
    adapt_rho_every: int = 0,
    adaptive_rho_tolerance: float = 5.0,
    rho_update_max_factor: float = 1.0e6,
    rho_min: float = 1e-6,
    rho_max: float = 1e6,
    slack_weight: float = 0.0,
    use_slack: bool = False,
    constraint_handling: int = ConstraintHandling.PROJECTION,
    log_barrier_kappa: float = 0.0,
) -> FusedADMMSolveResult:
    """Solve ADMM QP via fused CUDA kernel.

    Args:
        qp_data: QP problem data (cost, equality, inequality blocks)
        schur: Precomputed Schur complement matrices (S, Phiinv)
        state0: ADMM warm-start state
        max_iter, pcg_max_iter, etc: ADMM configuration

    Returns:
        Iteration counts, final ADMM state, and kernel time.
    """
    dtype = qp_data.cost.q.dtype
    S = schur.S
    Phiinv = schur.preconditioner_Phiinv

    # Detect batched vs unbatched from S shape
    batched = S.ndim == 4

    if batched:
        Nb, T, n, _ = S.shape
    else:
        T, n, _ = S.shape
        Nb = 1
        # Add batch dim for the kernel (always batched)
        S = S[None]
        Phiinv = Phiinv[None]

    # Extract dimensions
    n0 = qp_data.eq.A0.shape[
        -2
    ]  # initial constraint rows (nx+nu when constrain_initial_control)
    nx_dyn = qp_data.eq.A_minus.shape[-2]  # dynamics rows (always true nx)
    m = qp_data.ineq.G.shape[-2]  # (T, m, n) or (Nb, T, m, n)

    # Determine FFI target
    if dtype == jnp.float64:
        kernel_dtype = jnp.float64
        ffi_target = "admm_fused_cuda_f64"
    else:
        kernel_dtype = jnp.float32
        ffi_target = "admm_fused_cuda"

    N = T - 1
    n3 = 3 * n
    certificate_workspace_size = 7 + T * n + 2 * T * m + n0 + N * nx_dyn

    def _ensure(arr, shape, name=""):
        """Ensure array has correct dtype and batch dimension."""
        a = jnp.asarray(arr, dtype=kernel_dtype)
        if not batched and a.ndim == len(shape) - 1:
            a = a[None]
        return a

    # QP data arrays — add batch dim if needed
    S_k = jnp.asarray(S, dtype=kernel_dtype)
    Phiinv_k = jnp.asarray(Phiinv, dtype=kernel_dtype)
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
        rho_bar_k = rho_bar_k[None]  # (1,) for single problem
    slack_weight_k = jnp.broadcast_to(
        jnp.asarray(slack_weight, dtype=kernel_dtype), rho_bar_k.shape
    )

    # Output shapes
    out_shapes = [
        jax.ShapeDtypeStruct((Nb,), jnp.uint32),  # iters_out
        jax.ShapeDtypeStruct((Nb, T, n), kernel_dtype),  # x_blocks_out
        jax.ShapeDtypeStruct((Nb, T, m), kernel_dtype),  # z_g_out
        jax.ShapeDtypeStruct((Nb, T, m), kernel_dtype),  # y_g_out
        jax.ShapeDtypeStruct((Nb, n0), kernel_dtype),  # y_f_0_out
        jax.ShapeDtypeStruct((Nb, N, nx_dyn), kernel_dtype),  # y_f_dyn_out
        jax.ShapeDtypeStruct((Nb, T, m), kernel_dtype),  # xi_g_out
        jax.ShapeDtypeStruct((Nb,), kernel_dtype),  # rho_bar_out
        jax.ShapeDtypeStruct((Nb,), kernel_dtype),  # kernel_ns_out
        jax.ShapeDtypeStruct((Nb,), jnp.uint32),  # status_out
        jax.ShapeDtypeStruct((Nb, 4), kernel_dtype),  # residuals and limits
        jax.ShapeDtypeStruct(
            (Nb, certificate_workspace_size), kernel_dtype
        ),  # private certificate workspace
        jax.ShapeDtypeStruct(
            (Nb, T, n, n3 + n), kernel_dtype
        ),  # S_work + theta_inv temp
        jax.ShapeDtypeStruct((Nb, T, n, n3), kernel_dtype),  # Phiinv_work
    ]

    call = jax.ffi.ffi_call(
        ffi_target,
        out_shapes,
        vmap_method="broadcast_all",
    )

    results = call(
        # QP data (13 args)
        S_k,
        Phiinv_k,
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
        # Warm-start (7 args) + runtime params (1 arg)
        x0_k,
        z_g0_k,
        y_g0_k,
        y_f_0_k,
        y_f_dyn_k,
        xi_g0_k,
        rho_bar_k,
        slack_weight_k,
        # Config attrs
        max_iter=int(max_iter),
        pcg_max_iter=int(pcg_max_iter),
        check_termination_every=int(check_termination_every),
        check_infeasibility_every=int(check_infeasibility_every),
        eps_abs=float(eps_abs),
        eps_rel=float(eps_rel),
        eps_prim_inf=float(eps_prim_inf),
        eps_dual_inf=float(eps_dual_inf),
        sigma=float(sigma),
        rho_f_factor=float(rho_f_factor),
        alpha=float(alpha),
        pcg_eps=float(pcg_eps),
        nx=int(nx_dyn),
        n0=int(n0),
        m=int(m),
        use_slack=int(use_slack),
        adapt_rho_every=int(adapt_rho_every),
        adaptive_rho_tolerance=float(adaptive_rho_tolerance),
        rho_update_max_factor=float(rho_update_max_factor),
        rho_min=float(rho_min),
        rho_max=float(rho_max),
        constraint_handling=int(constraint_handling),
        log_barrier_kappa=float(log_barrier_kappa),
    )

    (
        iters_out,
        x_blocks_out,
        z_g_out,
        y_g_out,
        y_f_0_out,
        y_f_dyn_out,
        xi_g_out,
        rho_bar_out,
        kernel_ns_out,
        status_out,
        residuals_out,
        _certificate_work,
        _S_work,
        _Phiinv_work,
    ) = results

    # Remove batch dim if input was unbatched
    if not batched:
        iters_out = iters_out[0]
        x_blocks_out = x_blocks_out[0]
        z_g_out = z_g_out[0]
        y_g_out = y_g_out[0]
        y_f_0_out = y_f_0_out[0]
        y_f_dyn_out = y_f_dyn_out[0]
        xi_g_out = xi_g_out[0]
        rho_bar_out = rho_bar_out[0]
        kernel_ns_out = kernel_ns_out[0]
        status_out = status_out[0]
        residuals_out = residuals_out[0]

    state_out = ADMMState(
        x_blocks=jnp.asarray(x_blocks_out, dtype=dtype),
        y_g=jnp.asarray(y_g_out, dtype=dtype),
        y_f_0=jnp.asarray(y_f_0_out, dtype=dtype),
        y_f_dyn=jnp.asarray(y_f_dyn_out, dtype=dtype),
        z_g=jnp.asarray(z_g_out, dtype=dtype),
        xi_g=jnp.asarray(xi_g_out, dtype=dtype),
        rho_bar=jnp.asarray(rho_bar_out, dtype=dtype),
    )

    stats = ADMMStats(
        status=jnp.asarray(status_out, dtype=jnp.int32),
        num_iter=jnp.asarray(iters_out, dtype=jnp.int32),
        primal_residual=residuals_out[..., 0],
        dual_residual=residuals_out[..., 1],
        primal_residual_limit=residuals_out[..., 2],
        dual_residual_limit=residuals_out[..., 3],
    )
    return FusedADMMSolveResult(state_out, stats, kernel_ns_out)


def _admm_ffi_solve_unbatched_impl(
    qp_data: QPData,
    schur: SchurComplementMatrices,
    state0: ADMMState,
    *,
    max_iter: int = 100,
    pcg_max_iter: int = 200,
    check_termination_every: int = 5,
    check_infeasibility_every: int = 25,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-3,
    eps_prim_inf: float = 1e-4,
    eps_dual_inf: float = 1e-4,
    sigma: float = 1e-6,
    rho_f_factor: float = 1000.0,
    alpha: float = 1.6,
    pcg_eps: float = 1e-8,
    adapt_rho_every: int = 0,
    adaptive_rho_tolerance: float = 5.0,
    rho_update_max_factor: float = 1.0e6,
    rho_min: float = 1e-6,
    rho_max: float = 1e6,
    slack_weight: float = 0.0,
    use_slack: bool = False,
    constraint_handling: int = ConstraintHandling.PROJECTION,
    log_barrier_kappa: float = 0.0,
) -> FusedADMMSolveResult:
    """Fused ADMM solve for a single (unbatched) problem, vmap-compatible.

    All inputs must be unbatched (no leading batch dim).
    Outputs are unbatched. Designed to be called inside vmap.
    """
    dtype = qp_data.cost.q.dtype
    S = schur.S  # (T, n, 3n)
    Phiinv = schur.preconditioner_Phiinv  # (T, n, 3n)

    T, n, _ = S.shape
    n3 = 3 * n
    N = T - 1
    n0 = qp_data.eq.A0.shape[-2]  # initial constraint rows
    nx_dyn = qp_data.eq.A_minus.shape[-2]  # dynamics rows (always true nx)
    m = qp_data.ineq.G.shape[-2]
    certificate_workspace_size = 7 + T * n + 2 * T * m + n0 + N * nx_dyn

    if dtype == jnp.float64:
        kernel_dtype = jnp.float64
        ffi_target = "admm_fused_cuda_f64"
    else:
        kernel_dtype = jnp.float32
        ffi_target = "admm_fused_cuda"

    def _cast(arr):
        return jnp.asarray(arr, dtype=kernel_dtype)

    # Pass unbatched (rank-3) arrays; vmap+broadcast_all adds batch dim -> rank-4
    S_k = _cast(S)
    Phiinv_k = _cast(Phiinv)
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
        jax.ShapeDtypeStruct((), jnp.uint32),  # iters_out
        jax.ShapeDtypeStruct((T, n), kernel_dtype),  # x_blocks_out
        jax.ShapeDtypeStruct((T, m), kernel_dtype),  # z_g_out
        jax.ShapeDtypeStruct((T, m), kernel_dtype),  # y_g_out
        jax.ShapeDtypeStruct((n0,), kernel_dtype),  # y_f_0_out
        jax.ShapeDtypeStruct((N, nx_dyn), kernel_dtype),  # y_f_dyn_out
        jax.ShapeDtypeStruct((T, m), kernel_dtype),  # xi_g_out
        jax.ShapeDtypeStruct((), kernel_dtype),  # rho_bar_out
        jax.ShapeDtypeStruct((), kernel_dtype),  # kernel_ns_out
        jax.ShapeDtypeStruct((), jnp.uint32),  # status_out
        jax.ShapeDtypeStruct((4,), kernel_dtype),  # residuals and limits
        jax.ShapeDtypeStruct(
            (certificate_workspace_size,), kernel_dtype
        ),  # private certificate workspace
        jax.ShapeDtypeStruct((T, n, n3 + n), kernel_dtype),  # S_work + theta_inv temp
        jax.ShapeDtypeStruct((T, n, n3), kernel_dtype),  # Phiinv_work
    ]

    call = jax.ffi.ffi_call(
        ffi_target,
        out_shapes,
        vmap_method="broadcast_all",
    )

    results = call(
        S_k,
        Phiinv_k,
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
        pcg_max_iter=int(pcg_max_iter),
        check_termination_every=int(check_termination_every),
        check_infeasibility_every=int(check_infeasibility_every),
        eps_abs=float(eps_abs),
        eps_rel=float(eps_rel),
        eps_prim_inf=float(eps_prim_inf),
        eps_dual_inf=float(eps_dual_inf),
        sigma=float(sigma),
        rho_f_factor=float(rho_f_factor),
        alpha=float(alpha),
        pcg_eps=float(pcg_eps),
        nx=int(nx_dyn),
        n0=int(n0),
        m=int(m),
        use_slack=int(use_slack),
        adapt_rho_every=int(adapt_rho_every),
        adaptive_rho_tolerance=float(adaptive_rho_tolerance),
        rho_update_max_factor=float(rho_update_max_factor),
        rho_min=float(rho_min),
        rho_max=float(rho_max),
        constraint_handling=int(constraint_handling),
        log_barrier_kappa=float(log_barrier_kappa),
    )

    (
        iters_out,
        x_blocks_out,
        z_g_out,
        y_g_out,
        y_f_0_out,
        y_f_dyn_out,
        xi_g_out,
        rho_bar_out,
        kernel_ns_out,
        status_out,
        residuals_out,
        _certificate_work,
        _S_work,
        _Phiinv_work,
    ) = results

    state_out = ADMMState(
        x_blocks=jnp.asarray(x_blocks_out, dtype=dtype),
        y_g=jnp.asarray(y_g_out, dtype=dtype),
        y_f_0=jnp.asarray(y_f_0_out, dtype=dtype),
        y_f_dyn=jnp.asarray(y_f_dyn_out, dtype=dtype),
        z_g=jnp.asarray(z_g_out, dtype=dtype),
        xi_g=jnp.asarray(xi_g_out, dtype=dtype),
        rho_bar=jnp.asarray(rho_bar_out, dtype=dtype),
    )
    stats = ADMMStats(
        status=jnp.asarray(status_out, dtype=jnp.int32),
        num_iter=jnp.asarray(iters_out, dtype=jnp.int32),
        primal_residual=residuals_out[0],
        dual_residual=residuals_out[1],
        primal_residual_limit=residuals_out[2],
        dual_residual_limit=residuals_out[3],
    )
    return FusedADMMSolveResult(state_out, stats, kernel_ns_out)


def admm_ffi_solve(
    qp_data: QPData,
    schur: SchurComplementMatrices,
    state0: ADMMState,
    *,
    max_iter: int = 100,
    pcg_max_iter: int = 200,
    check_termination_every: int = 5,
    check_infeasibility_every: int = 25,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-3,
    eps_prim_inf: float = 1e-4,
    eps_dual_inf: float = 1e-4,
    sigma: float = 1e-6,
    rho_f_factor: float = 1000.0,
    alpha: float = 1.6,
    pcg_eps: float = 1e-8,
    adapt_rho_every: int = 0,
    adaptive_rho_tolerance: float = 5.0,
    rho_update_max_factor: float = 1.0e6,
    rho_min: float = 1e-6,
    rho_max: float = 1e6,
    slack_weight: float = 0.0,
    use_slack: bool = False,
    constraint_handling: int = ConstraintHandling.PROJECTION,
    log_barrier_kappa: float = 0.0,
) -> FusedADMMSolveResult:
    """Solve one QP, combining surrounding ``vmap`` axes into one FFI call."""

    def solve_one(qp_data, schur, state0, slack_weight):
        solve_impl = (
            _admm_ffi_solve_unbatched_impl
            if schur.S.ndim == 3
            else _admm_ffi_solve_impl
        )
        return solve_impl(
            qp_data,
            schur,
            state0,
            max_iter=max_iter,
            pcg_max_iter=pcg_max_iter,
            check_termination_every=check_termination_every,
            check_infeasibility_every=check_infeasibility_every,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            eps_prim_inf=eps_prim_inf,
            eps_dual_inf=eps_dual_inf,
            sigma=sigma,
            rho_f_factor=rho_f_factor,
            alpha=alpha,
            pcg_eps=pcg_eps,
            adapt_rho_every=adapt_rho_every,
            adaptive_rho_tolerance=adaptive_rho_tolerance,
            rho_update_max_factor=rho_update_max_factor,
            rho_min=rho_min,
            rho_max=rho_max,
            slack_weight=slack_weight,
            use_slack=use_slack,
            constraint_handling=constraint_handling,
            log_barrier_kappa=log_barrier_kappa,
        )

    solve_one = jax.custom_batching.custom_vmap(solve_one)

    @solve_one.def_vmap
    def solve_one_vmap(axis_size, in_batched, qp_data, schur, state0, slack_weight):
        args = (qp_data, schur, state0, slack_weight)
        leaves, treedef = jax.tree.flatten(args)
        batched_leaves = jax.tree.leaves(in_batched)
        if len(leaves) != len(batched_leaves):
            raise ValueError("custom vmap flags do not match ADMM input leaves")

        materialized_leaves = [
            (
                leaf
                if is_batched
                else jnp.broadcast_to(jnp.asarray(leaf), (axis_size,) + jnp.shape(leaf))
            )
            for leaf, is_batched in zip(leaves, batched_leaves)
        ]
        qp_batched, schur_batched, state_batched, slack_weight_batched = (
            jax.tree.unflatten(treedef, materialized_leaves)
        )
        batch_shape = schur_batched.S.shape[:-3]
        num_batch_dims = len(batch_shape)
        flat_args = jax.tree.map(
            lambda leaf: leaf.reshape((-1,) + leaf.shape[num_batch_dims:]),
            (qp_batched, schur_batched, state_batched, slack_weight_batched),
        )
        output_flat = solve_one(*flat_args)
        output = jax.tree.map(
            lambda leaf: leaf.reshape(batch_shape + leaf.shape[1:]), output_flat
        )
        return output, jax.tree.map(lambda _: True, output)

    return solve_one(qp_data, schur, state0, slack_weight)
