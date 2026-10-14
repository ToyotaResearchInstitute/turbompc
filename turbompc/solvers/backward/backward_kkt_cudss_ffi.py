"""cuDSS-based sparse KKT solver via JAX FFI for backward pass optimization.

The key design:
  - rowPtr and colIdx are computed once from the problem shape.
  - Only the CSR `values` array changes each call; it is assembled as a pure
    JAX operation (no host callbacks, no cuSPARSE, no cudaStreamSynchronize).
  - A Python-owned opaque workspace keeps the device CSR structure, cuDSS
    descriptors, and execution buffers alive. Only values and rhs enter the
    traced FFI call.
  - A custom_vmap rule intercepts JAX's batching of solve_backward_kkt_cudss_ffi:
    instead of Nb sequential FFI calls, it assembles one block-diagonal CSR
    system of size Nb*N_kkt and issues a single cuDSS call.
"""

from __future__ import annotations

import jax
import jax.custom_batching
import jax.numpy as jnp
import numpy as np

from turbompc.solvers.backward.backward_kkt_csr import (
    KKTCSRPattern,
    assemble_batched_kkt_csr_values,
    assemble_kkt_csr_values,
    build_batched_kkt_csr_pattern,
    build_kkt_csr_pattern,
)
from turbompc.solvers.ffi_loader import load_ffi_lib
from turbompc.solvers.qp_data import QPData
from turbompc.solvers.qp_utils import ZShape, unpack_x

_LIB_NAME = "libcudss_sparse_kkt_ffi.so"
_LIB = load_ffi_lib(_LIB_NAME)
jax.ffi.register_ffi_target(
    "cudss_sparse_kkt_workspace_destroy",
    jax.ffi.pycapsule(_LIB.TurboMpcDestroyCudssSparseKktWorkspace),
    platform="CUDA",
)
jax.ffi.register_ffi_target(
    "cudss_sparse_kkt_probe",
    jax.ffi.pycapsule(_LIB.CudssSparseKktProbe),
    platform="CUDA",
)

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
jax.ffi.register_ffi_target(
    "cudss_sparse_kkt_workspace_create",
    jax.ffi.pycapsule(_LIB.TurboMpcCreateCudssSparseKktWorkspace),
    platform="CUDA",
)


def _probe_cudss_device_bytes(pattern) -> tuple[int, int]:
    """Return a cuDSS memory estimate and the available JAX allocator budget."""
    for _ in range(2):
        workspace = None
        try:
            with jax.ensure_compile_time_eval():
                workspace = CudssSparseKktWorkspace(pattern, np.float64)
                with jax.default_device(workspace._device):
                    out = jax.ffi.ffi_call(
                        "cudss_sparse_kkt_probe",
                        jax.ShapeDtypeStruct((4,), np.uint32),
                        has_side_effect=True,
                    )(workspace_handle=np.int64(workspace.handle))
                    # Transport two 64-bit counters without requiring JAX x64.
                    estimate, driver_free = map(int, np.asarray(out).view(np.int64))
                device = workspace._device
                workspace.close()
                workspace = None
                stats = device.memory_stats() or {}
                bytes_in_use = int(stats.get("bytes_in_use", 0))
                pool_free = max(0, int(stats.get("bytes_reserved", 0)) - bytes_in_use)
                # XLA can use both unreserved driver memory and free memory in
                # JAX's pool. Cap that total by the allocator's own limit.
                free = driver_free + pool_free
                if (bytes_limit := stats.get("bytes_limit")) is not None:
                    free = min(free, max(0, int(bytes_limit) - bytes_in_use))
                if estimate > 0:
                    return estimate, free
        except RuntimeError:
            pass
        finally:
            if workspace is not None:
                workspace.close()
    return (0, 0)


class CudssSparseKktWorkspace:
    """Own one native cuDSS workspace and its CSR structure."""

    def __init__(self, pattern, dtype) -> None:
        self._handle = 0
        self._device = jax.config.jax_default_device
        if getattr(self._device, "platform", None) != "gpu":
            self._device = jax.local_devices(backend="gpu")[0]
        row_ptr = np.ascontiguousarray(pattern.rowPtr, dtype=np.int32)
        col_idx = np.ascontiguousarray(pattern.colIdx, dtype=np.int32)
        self.n = int(pattern.N_kkt)
        self.nnz = int(pattern.nnz)
        self.dtype = jnp.dtype(dtype)
        if self.dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise TypeError("sparse KKT workspace dtype must be float32 or float64")
        dtype_bytes = self.dtype.itemsize
        with jax.ensure_compile_time_eval(), jax.default_device(self._device):
            result = jax.ffi.ffi_call(
                "cudss_sparse_kkt_workspace_create",
                jax.ShapeDtypeStruct((2,), np.uint32),
                has_side_effect=True,
            )(row_ptr=row_ptr, col_idx=col_idx, dtype_bytes=np.int64(dtype_bytes))
            result.block_until_ready()
            self._handle = int(np.asarray(result).view(np.uint64).item())

    @property
    def handle(self) -> int:
        if self._handle == 0:
            raise RuntimeError("sparse KKT workspace is closed")
        return self._handle

    def close(self) -> None:
        if self._handle == 0:
            return
        with jax.ensure_compile_time_eval(), jax.default_device(self._device):
            result = jax.ffi.ffi_call(
                "cudss_sparse_kkt_workspace_destroy",
                jax.ShapeDtypeStruct((), np.uint32),
                has_side_effect=True,
            )(workspace_handle=np.int64(self.handle))
            result.block_until_ready()
        self._handle = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass


@jax.tree_util.register_pytree_node_class
class CudssBackwardKktSolver:
    """Python owner for all native KKT workspaces used by one MPC solver."""

    def __init__(self) -> None:
        self._workspaces = {}
        self._patterns = {}
        self._batched_patterns = {}
        self._safe_chunks = {}

    def tree_flatten(self):
        return (), self

    @classmethod
    def tree_unflatten(cls, owner, children):
        del children
        return owner

    def workspace(self, pattern, dtype) -> CudssSparseKktWorkspace:
        key = (int(pattern.N_kkt), int(pattern.nnz), jnp.dtype(dtype))
        workspace = self._workspaces.get(key)
        if workspace is None:
            workspace = CudssSparseKktWorkspace(pattern, dtype)
            self._workspaces[key] = workspace
        return workspace

    def pattern(self, N: int, n: int, m0: int, m: int, m_g: int):
        key = (N, n, m0, m, m_g)
        pattern = self._patterns.get(key)
        if pattern is None:
            pattern = build_kkt_csr_pattern(*key)
            self._patterns[key] = pattern
        return pattern

    def batched_pattern(
        self, batch_size: int, N: int, n: int, m0: int, m: int, m_g: int
    ):
        key = (batch_size, N, n, m0, m, m_g)
        pattern = self._batched_patterns.get(key)
        if pattern is None:
            pattern = build_batched_kkt_csr_pattern(
                batch_size, self.pattern(N, n, m0, m, m_g)
            )
            self._batched_patterns[key] = pattern
        return pattern

    def safe_chunk(self, N: int, n: int, m0: int, m: int, m_g: int) -> int:
        key = (N, n, m0, m, m_g)
        chunk = self._safe_chunks.get(key)
        if chunk is None:
            chunk = _safe_cudss_chunk(*key)
            self._safe_chunks[key] = chunk
        return chunk

    def solve(self, qp_data: QPData, zshape: ZShape):
        return solve_backward_kkt_cudss_ffi(qp_data, zshape, self)

    def close(self) -> None:
        workspaces = self._workspaces
        self._workspaces = {}
        for workspace in workspaces.values():
            workspace.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass


def _single_pattern(N: int, n: int, m0: int, m: int, m_g: int) -> KKTCSRPattern:
    """Build one host CSR pattern for memory probing."""
    return build_kkt_csr_pattern(N, n, m0, m, m_g)


# ---------------------------------------------------------------------------
# Core FFI call (single unbatched problem)
# ---------------------------------------------------------------------------


def _call_ffi(
    values: jax.Array,
    rhs: jax.Array,
    workspace: CudssSparseKktWorkspace,
) -> jax.Array:
    """Issue a single cuDSS FFI call. Returns solution vector of length N_kkt."""
    dtype = workspace.dtype
    ffi_target = (
        "cudss_sparse_kkt_f64" if dtype == jnp.float64 else "cudss_sparse_kkt_f32"
    )
    sol_shape = jax.ShapeDtypeStruct((workspace.n,), dtype)
    call = jax.ffi.ffi_call(ffi_target, sol_shape, vmap_method="sequential")
    return call(
        values,
        rhs,
        n=workspace.n,
        nnz=workspace.nnz,
        workspace_handle=workspace.handle,
    )


# ---------------------------------------------------------------------------
# Public solver (single problem) — decorated with custom_vmap
# ---------------------------------------------------------------------------


@jax.custom_batching.custom_vmap
def solve_backward_kkt_cudss_ffi(
    qp_data: QPData,
    zshape: ZShape,
    solver: CudssBackwardKktSolver,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
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

    pattern = solver.pattern(N, n, m0, m, m_g)
    values = assemble_kkt_csr_values(D, E, A0, A_minus, A_plus, G, pattern)

    q_flat = qp_data.cost.q.reshape(-1)
    rhs = jnp.concatenate([-q_flat, jnp.zeros(pattern.m_total, dtype=dtype)])

    solution = _call_ffi(values, rhs, solver.workspace(pattern, dtype))

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


def _safe_cudss_chunk(N: int, n: int, m0: int, m: int, m_g: int) -> int:
    """Largest block-diagonal chunk that provably fits cuDSS device memory.

    Calibrated once per (N, n, m0, m, m_g): probe one block's device cost
    and the free device budget, then
    chunk = max(1, floor(free*safety / M_block)).  Cached.  Probe failure
    (or zero estimate) => conservative chunk = 1 (one independent block per
    cuDSS call: provably safe, just slower).  Called once per shape from the
    chunked vmap path; the probe executes eagerly through FFI,
    outside the solve graph.
    """
    pattern = _single_pattern(N, n, m0, m, m_g)
    m_block, free_bytes = _probe_cudss_device_bytes(pattern)
    if m_block <= 0 or free_bytes <= 0:
        chunk = 1
    else:
        chunk = max(1, int(free_bytes * _CUDSS_BUDGET_SAFETY) // m_block)
    return chunk


@solve_backward_kkt_cudss_ffi.def_vmap
def _solve_backward_kkt_cudss_ffi_vmap(axis_size, in_batched, qp_data, zshape, solver):
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

    if Nb > solver.safe_chunk(N, n, m0, m, m_g):
        return _solve_backward_kkt_chunked(
            axis_size, in_batched, qp_data, zshape, solver
        )

    bat = solver.batched_pattern(Nb, N, n, m0, m, m_g)
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

    solution = _call_ffi(values, rhs, solver.workspace(bat, dtype))

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


def _solve_backward_kkt_chunked(axis_size, in_batched, qp_data, zshape, solver):
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

    chunk = solver.safe_chunk(N, n, m0, m, m_g)
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
            chunk, in_batched, qp_chunk, zshape, solver
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
