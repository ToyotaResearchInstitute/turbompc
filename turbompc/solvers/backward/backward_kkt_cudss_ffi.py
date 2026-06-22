"""cuDSS-based sparse KKT solver via JAX FFI for backward pass optimization.

The key design:
  - rowPtr and colIdx are computed *once* from the problem shape and cached.
  - Only the CSR `values` array changes each call; it is assembled as a pure
    JAX operation (no host callbacks, no cuSPARSE, no cudaStreamSynchronize).
  - The FFI kernel (CudssSparseKktF32/F64) receives a pre-built CSR triple
    (rowPtr, colIdx, values) already on device, with nnz inferred from
    colIdx.shape[0] in C++ — eliminating the previous device→host nnz sync.
  - A custom_vmap rule intercepts JAX's batching of solve_backward_kkt_cudss_ffi:
    instead of Nb sequential FFI calls, it assembles one block-diagonal CSR
    system of size Nb*N_kkt and issues a single cuDSS call.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.custom_batching
import jax.numpy as jnp
from turbompc.solvers.backward.backward_kkt_csr import (
    BatchedKKTCSRPattern,
    KKTCSRPattern,
    assemble_batched_kkt_csr_values,
    assemble_kkt_csr_values,
    build_batched_kkt_csr_pattern,
    build_kkt_csr_pattern,
)
from turbompc.solvers.qp_data import QPData
from turbompc.solvers.qp_utils import ZShape, unpack_x


def _find_lib() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent.parent
        / "solvers"
        / "csrc"
        / "build"
        / "libcudss_sparse_kkt_ffi.so",
        Path(__file__).resolve().parent.parent.parent.parent
        / "build"
        / "ffi"
        / "libcudss_sparse_kkt_ffi.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find libcudss_sparse_kkt_ffi.so. Build first:\n"
        "  cmake -S turbompc/solvers/csrc -B build/ffi -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build build/ffi -j\n"
    )


_LIB = ctypes.cdll.LoadLibrary(str(_find_lib()))

jax.ffi.register_ffi_target(
    "cudss_sparse_kkt_f32",
    jax.ffi.pycapsule(_LIB.CudssSparseKktF32),
    platform="CUDA",
)
jax.ffi.register_ffi_target(
    "cudss_sparse_kkt_f64",
    jax.ffi.pycapsule(_LIB.CudssSparseKktF64),
    platform="CUDA",
)
# Dense-input targets kept registered (unused by default, available for testing)
jax.ffi.register_ffi_target(
    "cudss_sparse_kkt_dense_f32",
    jax.ffi.pycapsule(_LIB.CudssSparseKktDenseF32),
    platform="CUDA",
)
jax.ffi.register_ffi_target(
    "cudss_sparse_kkt_dense_f64",
    jax.ffi.pycapsule(_LIB.CudssSparseKktDenseF64),
    platform="CUDA",
)

# ---------------------------------------------------------------------------
# Analysis-cache control (ctypes, NOT FFI — side-effecting cleanup/diagnostics,
# not part of the traced graph). The C++ cuDSS symbolic-analysis cache is keyed
# by (n, nnz, dtype) and otherwise lives for the process lifetime.
# ---------------------------------------------------------------------------
_LIB.ClearCudssAnalysisCacheImpl.restype = None
_LIB.CudssAnalysisCacheSize.restype = ctypes.c_int
_LIB.CudssProbeDeviceBytesImpl.restype = ctypes.c_int
_LIB.CudssProbeDeviceBytesImpl.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # rowPtr_host (n+1)
    ctypes.POINTER(ctypes.c_int32),  # colIdx_host (nnz)
    ctypes.c_int32,  # n
    ctypes.c_int32,  # nnz
    ctypes.c_int,  # is_double
    ctypes.POINTER(ctypes.c_int64),  # out[2]
]


def _probe_cudss_device_bytes(pattern) -> Tuple[int, int]:
    """Eagerly run single-block cuDSS PHASE_ANALYSIS for `pattern` and return
    (per_block_device_bytes, free_device_bytes). (0, 0) on probe failure.

    Side-effecting, runs OUTSIDE the traced graph (raw ctypes, not a JAX op).
    Called once per shape from `_safe_cudss_chunk` and cached there.

    Retries once on a (0,0)/error result to absorb a cold-CUDA-context
    first-call flake.
    """
    import numpy as _np

    rowPtr = _np.ascontiguousarray(pattern.rowPtr, dtype=_np.int32)
    colIdx = _np.ascontiguousarray(pattern.colIdx, dtype=_np.int32)
    n = ctypes.c_int32(int(pattern.N_kkt))
    nnz = ctypes.c_int32(int(pattern.nnz))
    for _ in range(2):
        out = (ctypes.c_int64 * 2)()
        rc = _LIB.CudssProbeDeviceBytesImpl(
            rowPtr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            colIdx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n,
            nnz,
            ctypes.c_int(1),  # f64 (backward solve is float64)
            out,
        )
        if rc == 0 and int(out[0]) > 0:
            return (int(out[0]), int(out[1]))
    return (0, 0)


def clear_analysis_cache() -> None:
    """Free all cached cuDSS symbolic-analysis objects (one per sparsity
    pattern: (n, nnz, dtype)).

    The cache speeds up repeated solves of the same problem shape but is never
    evicted, so a long-lived process that solves many distinct shapes
    (interactive sessions, shape-sweeping training) accumulates GPU memory.
    Call this to release it. (The per-config sweep is unaffected — each config
    is a fresh process with a single fixed shape.)
    """
    _LIB.ClearCudssAnalysisCacheImpl()


def analysis_cache_size() -> int:
    """Live number of cached cuDSS analysis objects (debug/test helper)."""
    return int(_LIB.CudssAnalysisCacheSize())


# ---------------------------------------------------------------------------
# Pattern caches
# ---------------------------------------------------------------------------

# Single-problem cache: (N, n, m0, m, m_g) → KKTCSRPattern (numpy only, no JAX arrays)
_PATTERN_CACHE: Dict[Tuple[int, int, int, int, int], KKTCSRPattern] = {}

# Batched cache: (Nb, N, n, m0, m, m_g) → BatchedKKTCSRPattern (numpy only, no JAX arrays)
_BATCHED_PATTERN_CACHE: Dict[
    Tuple[int, int, int, int, int, int], BatchedKKTCSRPattern
] = {}


def _single_pattern(N: int, n: int, m0: int, m: int, m_g: int) -> KKTCSRPattern:
    """Cached single-problem CSR pattern (numpy only — no device copy)."""
    key = (N, n, m0, m, m_g)
    if key not in _PATTERN_CACHE:
        _PATTERN_CACHE[key] = build_kkt_csr_pattern(N, n, m0, m, m_g)
    return _PATTERN_CACHE[key]


def _get_or_build_pattern(
    N: int, n: int, m0: int, m: int, m_g: int
) -> Tuple[jax.Array, jax.Array, KKTCSRPattern]:
    """Return (rowPtr_dev, colIdx_dev, pattern), building pattern on first call.

    The numpy pattern is cached; jnp.asarray() is called fresh each time so that
    no JAX tracer ever escapes its transformation scope into the global cache.
    """
    key = (N, n, m0, m, m_g)
    if key not in _PATTERN_CACHE:
        _PATTERN_CACHE[key] = build_kkt_csr_pattern(N, n, m0, m, m_g)
    pat = _PATTERN_CACHE[key]
    return jnp.asarray(pat.rowPtr), jnp.asarray(pat.colIdx), pat


def _get_or_build_batched_pattern(
    Nb: int, N: int, n: int, m0: int, m: int, m_g: int
) -> Tuple[jax.Array, jax.Array, BatchedKKTCSRPattern]:
    """Return (rowPtr_dev, colIdx_dev, batched_pattern), building on first call.

    The numpy pattern is cached; jnp.asarray() is called fresh each time so that
    no JAX tracer ever escapes its transformation scope into the global cache.
    """
    key = (Nb, N, n, m0, m, m_g)
    if key not in _BATCHED_PATTERN_CACHE:
        # Reuse (or build) the single-problem pattern, then build the batched one.
        single_key = (N, n, m0, m, m_g)
        if single_key not in _PATTERN_CACHE:
            _PATTERN_CACHE[single_key] = build_kkt_csr_pattern(N, n, m0, m, m_g)
        single = _PATTERN_CACHE[single_key]
        _BATCHED_PATTERN_CACHE[key] = build_batched_kkt_csr_pattern(Nb, single)
    bat = _BATCHED_PATTERN_CACHE[key]
    return jnp.asarray(bat.rowPtr), jnp.asarray(bat.colIdx), bat


# ---------------------------------------------------------------------------
# Core FFI call (single unbatched problem)
# ---------------------------------------------------------------------------


def _call_ffi(
    rowPtr_dev: jax.Array,
    colIdx_dev: jax.Array,
    values: jax.Array,
    rhs: jax.Array,
    N_kkt: int,
    dtype,
) -> jax.Array:
    """Issue a single cuDSS FFI call. Returns solution vector of length N_kkt."""
    ffi_target = (
        "cudss_sparse_kkt_f64" if dtype == jnp.float64 else "cudss_sparse_kkt_f32"
    )
    sol_shape = jax.ShapeDtypeStruct((N_kkt,), dtype)
    call = jax.ffi.ffi_call(ffi_target, sol_shape, vmap_method="sequential")
    return call(rowPtr_dev, colIdx_dev, values, rhs)


# ---------------------------------------------------------------------------
# Public solver (single problem) — decorated with custom_vmap
# ---------------------------------------------------------------------------


@jax.custom_batching.custom_vmap
def solve_backward_kkt_cudss_ffi(
    qp_data: QPData,
    zshape: ZShape,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Solve the backward-pass KKT system using cuDSS sparse FFI.

    When called under jax.vmap (e.g. from the SQP backward pass), the
    custom_vmap rule intercepts and issues a single block-diagonal cuDSS
    call instead of Nb sequential calls.

    Args:
        qp_data: QPData for the backward QP (unbatched, active-set only).
        zshape: ZShape defining (N, n_x, n_u).

    Returns:
        ((states_sensitivity, controls_sensitivity), constraint_multipliers)
    """
    D = qp_data.cost.D  # (N+1, n, n)
    E = qp_data.cost.E  # (N,   n, n)
    A0 = qp_data.eq.A0  # (m0,  n)
    A_minus = qp_data.eq.A_minus  # (N,   m, n)
    A_plus = qp_data.eq.A_plus  # (N,   m, n)
    G = qp_data.ineq.G  # (N+1, m_g, n)

    N = D.shape[0] - 1
    n = D.shape[1]
    m0 = A0.shape[0]
    m = A_minus.shape[1]
    m_g = G.shape[1]
    dtype = D.dtype

    rowPtr_dev, colIdx_dev, pattern = _get_or_build_pattern(N, n, m0, m, m_g)
    values = assemble_kkt_csr_values(D, E, A0, A_minus, A_plus, G, pattern)

    q_flat = qp_data.cost.q.reshape(-1)
    rhs = jnp.concatenate([-q_flat, jnp.zeros(pattern.m_total, dtype=dtype)])

    solution = _call_ffi(rowPtr_dev, colIdx_dev, values, rhs, pattern.N_kkt, dtype)

    n_total = pattern.n_total
    x_blocks = solution[:n_total].reshape(N + 1, n)
    x_solution, y_solution = unpack_x(x_blocks, zshape)
    multipliers = solution[n_total:]

    return (x_solution, y_solution), multipliers


# Per-shape chunk calibration.
#
# The batched backward solve stacks `chunk` IDENTICAL, INDEPENDENT KKT blocks
# into one block-diagonal cuDSS factorization, so its device-memory footprint
# is exactly `chunk * M_block`, where M_block is the (fill-in-dominated)
# device memory ONE block's PHASE_ANALYSIS needs.  M_block is not predictable
# from nnz; we MEASURE it once per sparsity shape via the eager cuDSS probe
# (`_probe_cudss_device_bytes`, which reads cuDSS's own
# CUDSS_DATA_MEMORY_ESTIMATES) and divide a measured device budget by it.
#
# `safety` (<1) absorbs factorization's incremental allocation, fragmentation
# and free-memory drift between the eager probe and the jitted solve.  It is a
# single coarse global knob, NOT a per-shape constant: being conservative only
# costs throughput (more, smaller chunks); it can never cause an OOM because
# chunk >= 1 always and one block provably fits.
#
# GPU-only by design: no host spill, no retry, no per-shape magic constant
# (cuDSS supplies the per-shape number).
_CUDSS_BUDGET_SAFETY = 0.8

# (N, n, m0, m, m_g) -> calibrated static chunk.  Mirrors _PATTERN_CACHE:
# numpy/int only, no JAX arrays, computed once per shape.
_SAFE_CHUNK_CACHE: Dict[Tuple[int, int, int, int, int], int] = {}


def _safe_cudss_chunk(N: int, n: int, m0: int, m: int, m_g: int) -> int:
    """Largest block-diagonal chunk that provably fits cuDSS device memory.

    Calibrated once per (N, n, m0, m, m_g): probe one block's device cost
    and the free device budget, then
    chunk = max(1, floor(free*safety / M_block)).  Cached.  Probe failure
    (or zero estimate) => conservative chunk = 1 (one independent block per
    cuDSS call: provably safe, just slower).  Called once per shape from the
    chunked vmap path; the probe is eager (raw ctypes), never in the graph.
    """
    key = (N, n, m0, m, m_g)
    cached = _SAFE_CHUNK_CACHE.get(key)
    if cached is not None:
        return cached
    pattern = _single_pattern(N, n, m0, m, m_g)
    m_block, free_bytes = _probe_cudss_device_bytes(pattern)
    if m_block <= 0 or free_bytes <= 0:
        chunk = 1
    else:
        chunk = max(1, int(free_bytes * _CUDSS_BUDGET_SAFETY) // m_block)
    _SAFE_CHUNK_CACHE[key] = chunk
    return chunk


@solve_backward_kkt_cudss_ffi.def_vmap
def _solve_backward_kkt_cudss_ffi_vmap(axis_size, in_batched, qp_data, zshape):
    """Custom vmap rule: assemble Nb problems as one block-diagonal system.

    Called by JAX when solve_backward_kkt_cudss_ffi is inside a jax.vmap.
    axis_size = Nb (the batch dimension size).
    in_batched = tuple of booleans indicating which args have a batch axis.
    qp_data and zshape are the raw (batched) pytree values passed by JAX.

    For large batches the block-diagonal system is split into chunks whose
    stacked (nnz, N_kkt) stay within the cuDSS workspace budget (see
    `_safe_cudss_chunk`).  Block-diagonal blocks are independent, so chunking
    is numerically exact - it only changes the factorization granularity.
    """
    # Extract batched arrays from qp_data pytree.
    # Under jax.vmap with axis_size=Nb, dynamic leaves that vary across the
    # batch have a leading Nb dim.  Static matrices (linear dynamics, shared
    # constraints) may NOT have a leading Nb dim — handle both cases.
    D = qp_data.cost.D  # (Nb, N+1, n, n)
    E = qp_data.cost.E  # (Nb, N,   n, n)
    A0 = qp_data.eq.A0  # (m0, n) or (Nb, m0, n)
    A_minus = qp_data.eq.A_minus  # (N, m, n) or (Nb, N, m, n)
    A_plus = qp_data.eq.A_plus  # (N, m, n) or (Nb, N, m, n)
    G = qp_data.ineq.G  # (N+1, m_g, n) or (Nb, N+1, m_g, n)

    Nb = axis_size
    if D.ndim == 4:
        N = D.shape[1] - 1
        n = D.shape[2]
    else:
        N = D.shape[0] - 1
        n = D.shape[1]
    m0 = A0.shape[-2] if A0.ndim == 3 else A0.shape[0]
    m = A_minus.shape[-2] if A_minus.ndim == 4 else A_minus.shape[1]
    m_g = G.shape[-2] if G.ndim == 4 else G.shape[1]
    dtype = D.dtype

    if Nb > _safe_cudss_chunk(N, n, m0, m, m_g):
        return _solve_backward_kkt_chunked(axis_size, in_batched, qp_data, zshape)

    rowPtr_dev, colIdx_dev, bat = _get_or_build_batched_pattern(Nb, N, n, m0, m, m_g)
    values = assemble_batched_kkt_csr_values(D, E, A0, A_minus, A_plus, G, bat)

    # Build batched RHS: concatenate [-q_i; 0_i] for all i → (Nb * N_kkt,)
    # q may have shape (N+1, n) [unbatched] or (Nb, N+1, n) [batched].
    q = qp_data.cost.q
    if q.ndim == 2:
        q_flat = jnp.tile(q.ravel(), (Nb, 1))  # (Nb, n_total)
    else:
        q_flat = q.reshape(Nb, -1)  # (Nb, n_total)
    zeros_dual = jnp.zeros((Nb, bat.single.m_total), dtype=dtype)
    rhs_batched = jnp.concatenate([-q_flat, zeros_dual], axis=1)  # (Nb, N_kkt)
    rhs = rhs_batched.ravel()  # (Nb * N_kkt,)

    solution = _call_ffi(rowPtr_dev, colIdx_dev, values, rhs, bat.N_kkt_total, dtype)

    # Unpack: split into Nb solutions
    N_kkt = bat.single.N_kkt
    n_total = bat.single.n_total
    solutions = solution.reshape(Nb, N_kkt)  # (Nb, N_kkt)

    x_blocks_batch = solutions[:, :n_total].reshape(Nb, N + 1, n)
    multipliers_batch = solutions[:, n_total:]  # (Nb, m_total)

    # Unpack in batch directly to avoid shape-order surprises from nested vmap.
    x_solution_batch = x_blocks_batch[:, :, : zshape.num_states]
    y_solution_batch = x_blocks_batch[:, :, zshape.num_states :]

    # Return (output, out_batched) — required by custom_vmap def_vmap API.
    # out_batched mirrors the output pytree structure with True/False per leaf.
    out = (x_solution_batch, y_solution_batch), multipliers_batch
    out_batched = (True, True), True
    return out, out_batched


def _solve_backward_kkt_chunked(axis_size, in_batched, qp_data, zshape):
    """Split a large batch into chunks, solve each as one block-diagonal
    cuDSS call, concatenate.  Numerically exact (blocks are independent).

    Compile cost is O(1) in n_chunks: the per-chunk body is compiled ONCE
    via jax.lax.map and looped at runtime (NOT a Python for-loop, which
    unrolls into the XLA graph and made compile O(n_chunks)).

    Which leaves carry the batch axis comes from JAX's authoritative
    `in_batched` (a QPData-shaped tree of bools), never a shape heuristic:
    a static (unbatched) leaf whose leading dim equals Nb (e.g. A0
    (m0=nx, n) when nx == Nb) must NOT be sliced/mapped.
    """
    qp_batched = in_batched[0]  # QPData-structured tree of per-leaf bools

    D = qp_data.cost.D
    A0 = qp_data.eq.A0
    A_minus = qp_data.eq.A_minus
    G = qp_data.ineq.G
    if D.ndim == 4:
        N = D.shape[1] - 1
        n = D.shape[2]
    else:
        N = D.shape[0] - 1
        n = D.shape[1]
    m0 = A0.shape[-2] if A0.ndim == 3 else A0.shape[0]
    m = A_minus.shape[-2] if A_minus.ndim == 4 else A_minus.shape[1]
    m_g = G.shape[-2] if G.ndim == 4 else G.shape[1]

    chunk = _safe_cudss_chunk(N, n, m0, m, m_g)
    Nb = axis_size
    n_chunks = (Nb + chunk - 1) // chunk
    padded = n_chunks * chunk

    leaves, treedef = jax.tree.flatten(qp_data)
    flags, _ = jax.tree.flatten(qp_batched)  # same treedef => aligned bools

    # Reshape each BATCHED leaf to (n_chunks, chunk, ...); pad Nb->padded by
    # repeating the edge sample (padded rows are discarded after the map).
    mapped_xs = []
    for x, is_b in zip(leaves, flags):
        if not is_b:
            continue
        pad_w = [(0, padded - Nb)] + [(0, 0)] * (x.ndim - 1)
        xp = jnp.pad(x, pad_w, mode="edge")
        mapped_xs.append(xp.reshape((n_chunks, chunk) + tuple(x.shape[1:])))

    def _body(packed):
        # Rebuild qp_data for this chunk: batched leaves from `packed`
        # (each (chunk, ...)); static leaves closed over from `leaves`.
        it = iter(packed)
        new_leaves = [next(it) if is_b else x for x, is_b in zip(leaves, flags)]
        qp_chunk = jax.tree.unflatten(treedef, new_leaves)
        out_chunk, _ = _solve_backward_kkt_cudss_ffi_vmap(
            chunk, in_batched, qp_chunk, zshape
        )
        (xs_, ys_), ms_ = out_chunk
        return xs_, ys_, ms_

    xb, yb, mb = jax.lax.map(_body, tuple(mapped_xs))  # each (n_chunks, chunk, ...)

    def _flat(a):
        return a.reshape((padded,) + tuple(a.shape[2:]))[:Nb]

    x_solution_batch = _flat(xb)
    y_solution_batch = _flat(yb)
    multipliers_batch = _flat(mb)
    out = (x_solution_batch, y_solution_batch), multipliers_batch
    out_batched = (True, True), True
    return out, out_batched
