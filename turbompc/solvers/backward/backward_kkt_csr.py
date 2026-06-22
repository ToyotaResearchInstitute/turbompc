"""CSR pattern + JAX values assembly for the backward KKT system.

The backward KKT linear system is:

    [ P    C^T ] [ x ]   [ -q ]
    [ C    0   ] [ y ] = [  0 ]

where
    P   (n_total x n_total) - block-tridiagonal Hessian
    C   (m_total x n_total) - constraint Jacobian  [A; G_assembled]
    n_total = n * (N+1)
    m_total = m_eq + m_ineq  where  m_eq = m0 + N*m,  m_ineq = (N+1)*m_g

The sparsity pattern (rowPtr, colIdx) depends only on the problem *shape*
(N, n, m0, m, m_g) - all Python constants for a given problem.  It can be
computed once and cached.  Only the *values* change each backward call.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class KKTCSRPattern:
    """Immutable CSR sparsity pattern for one problem shape."""

    rowPtr: np.ndarray  # int32 (N_kkt + 1,)
    colIdx: np.ndarray  # int32 (nnz,)
    nnz: int
    N_kkt: int
    n_total: int
    m_total: int
    # Problem shape
    N: int
    n: int
    m0: int
    m: int
    m_g: int


def build_kkt_csr_pattern(N: int, n: int, m0: int, m: int, m_g: int) -> KKTCSRPattern:
    """Build the static CSR sparsity pattern of the KKT matrix.

    Called once per problem shape (N, n, m0, m, m_g).  Returns numpy arrays
    that can be converted to JAX device arrays and reused across all backward
    calls with the same problem shape.

    Column layout within each row (sorted ascending):

    Primal row in block b (b = 0 … N):
      P sub-diagonal  : cols (b-1)*n … b*n-1          (only if b > 0)
      P diagonal      : cols b*n     … (b+1)*n-1
      P super-diagonal: cols (b+1)*n … (b+2)*n-1      (only if b < N)
      A0^T            : cols n_total … n_total+m0-1    (only if b == 0 and m0 > 0)
      A_plus[b-1]^T   : cols n_total+m0+(b-1)*m …     (only if b > 0 and m > 0)
      A_minus[b]^T    : cols n_total+m0+b*m …          (only if b < N and m > 0)
      G[b]^T          : cols n_total+m_eq+b*m_g …      (only if m_g > 0)

    Dual row (constraint row) ordering matches C = [A0; dynamics; G]:
      A0 rows     : n entries each (cols 0 … n-1)
      Dynamics    : 2n entries each (cols i*n … (i+2)*n-1 for step i)
      G rows      : n entries each (cols i*n … (i+1)*n-1 for timestep i)
    """
    n_total = n * (N + 1)
    m_eq = m0 + N * m
    m_ineq = (N + 1) * m_g
    m_total = m_eq + m_ineq
    N_kkt = n_total + m_total

    row_cols: list[list[int]] = []

    # ---- Primal rows ----
    for b in range(N + 1):
        for _ in range(n):
            cols: list[int] = []
            if b > 0:
                cols += list(range((b - 1) * n, b * n))  # P sub-diagonal
            cols += list(range(b * n, (b + 1) * n))  # P diagonal
            if b < N:
                cols += list(range((b + 1) * n, (b + 2) * n))  # P super-diagonal
            if b == 0 and m0 > 0:
                cols += list(range(n_total, n_total + m0))  # A0^T
            if b > 0 and m > 0:
                cs = n_total + m0 + (b - 1) * m
                cols += list(range(cs, cs + m))  # A_plus[b-1]^T
            if b < N and m > 0:
                cs = n_total + m0 + b * m
                cols += list(range(cs, cs + m))  # A_minus[b]^T
            if m_g > 0:
                cs = n_total + m_eq + b * m_g
                cols += list(range(cs, cs + m_g))  # G[b]^T
            row_cols.append(cols)

    # ---- Dual rows: A0 section ----
    for _ in range(m0):
        row_cols.append(list(range(n)))

    # ---- Dual rows: dynamics section ----
    for i in range(N):
        for _ in range(m):
            row_cols.append(
                list(range(i * n, (i + 1) * n)) + list(range((i + 1) * n, (i + 2) * n))
            )

    # ---- Dual rows: G section ----
    for i in range(N + 1):
        for _ in range(m_g):
            row_cols.append(list(range(i * n, (i + 1) * n)))

    assert (
        len(row_cols) == N_kkt
    ), f"build_kkt_csr_pattern: expected {N_kkt} rows, got {len(row_cols)}"

    # Build rowPtr
    rowPtr = np.zeros(N_kkt + 1, dtype=np.int32)
    total_nnz = 0
    for i, cols in enumerate(row_cols):
        rowPtr[i] = total_nnz
        total_nnz += len(cols)
    rowPtr[N_kkt] = total_nnz

    # Build colIdx
    colIdx = np.empty(total_nnz, dtype=np.int32)
    offset = 0
    for cols in row_cols:
        nc = len(cols)
        colIdx[offset : offset + nc] = cols
        offset += nc

    return KKTCSRPattern(
        rowPtr=rowPtr,
        colIdx=colIdx,
        nnz=total_nnz,
        N_kkt=N_kkt,
        n_total=n_total,
        m_total=m_total,
        N=N,
        n=n,
        m0=m0,
        m=m,
        m_g=m_g,
    )


def assemble_kkt_csr_values(
    D: jnp.ndarray,  # (N+1, n, n)
    E: jnp.ndarray,  # (N,   n, n)
    A0: jnp.ndarray,  # (m0,  n)
    A_minus: jnp.ndarray,  # (N,   m, n)
    A_plus: jnp.ndarray,  # (N,   m, n)
    G: jnp.ndarray,  # (N+1, m_g, n)
    pattern: KKTCSRPattern,
) -> jnp.ndarray:
    """Assemble the CSR values array for the KKT matrix — pure JAX, no host sync.

    The values are assembled directly from the block matrices in the same
    row-major order as `pattern.colIdx`, so the resulting array is a valid
    CSR values buffer for (pattern.rowPtr, pattern.colIdx).

    For each primal block b the n rows share the same column structure.
    The values for all n rows of block b form an (n, nnz_b) sub-matrix:

        [ E[b-1]  |  D[b]  |  E[b].T  |  A0.T  |  A_plus[b-1].T  |  A_minus[b].T  |  G[b].T ]
            (sub)    (diag)   (super)    (A0^T)       (Aplus^T)        (Aminus^T)      (G^T)

    with the presence of each segment matching the column pattern above.

    Dual (constraint) rows are assembled as flattened block matrices:
        A0 section:       A0.ravel()
        Dynamics section: [A_minus[i] | A_plus[i]].ravel() for each i
        G section:        G[i].ravel() for each i
    """
    N = pattern.N
    m0 = pattern.m0
    m = pattern.m
    m_g = pattern.m_g

    all_parts: list[jnp.ndarray] = []

    # ---- Primal blocks ----
    for b in range(N + 1):
        row_parts: list[jnp.ndarray] = []
        if b > 0:
            row_parts.append(E[b - 1])  # (n, n)  sub-diagonal
        row_parts.append(D[b])  # (n, n)  diagonal
        if b < N:
            row_parts.append(E[b].T)  # (n, n)  super-diagonal
        if b == 0 and m0 > 0:
            row_parts.append(A0.T)  # (n, m0)
        if b > 0 and m > 0:
            row_parts.append(A_plus[b - 1].T)  # (n, m)
        if b < N and m > 0:
            row_parts.append(A_minus[b].T)  # (n, m)
        if m_g > 0:
            row_parts.append(G[b].T)  # (n, m_g)
        block = jnp.concatenate(row_parts, axis=1)  # (n, nnz_b)
        all_parts.append(block.ravel())  # (n * nnz_b,)

    # ---- Dual: A0 rows ----
    if m0 > 0:
        all_parts.append(A0.ravel())  # (m0 * n,)

    # ---- Dual: dynamics rows ----
    for i in range(N):
        if m > 0:
            dyn = jnp.concatenate([A_minus[i], A_plus[i]], axis=1)  # (m, 2n)
            all_parts.append(dyn.ravel())

    # ---- Dual: G rows ----
    for i in range(N + 1):
        if m_g > 0:
            all_parts.append(G[i].ravel())  # (m_g * n,)

    return jnp.concatenate(all_parts)


# ---------------------------------------------------------------------------
# Batched (block-diagonal) variants
# ---------------------------------------------------------------------------


@dataclass
class BatchedKKTCSRPattern:
    """CSR sparsity pattern for Nb block-diagonal KKT systems."""

    rowPtr: np.ndarray  # int32 (Nb*N_kkt + 1,)
    colIdx: np.ndarray  # int32 (Nb*nnz_per_sys,)
    nnz: int  # = Nb * nnz_per_sys
    N_kkt_total: int  # = Nb * N_kkt_per_sys
    single: KKTCSRPattern
    Nb: int


def build_batched_kkt_csr_pattern(
    Nb: int, single: KKTCSRPattern
) -> BatchedKKTCSRPattern:
    """Build block-diagonal CSR pattern for Nb copies of the same KKT system.

    The resulting matrix is block-diagonal: no sparsity coupling across
    problems.  rowPtr and colIdx are computed from the single-problem
    pattern by offsetting row-pointer counts and column indices.
    """
    N_kkt = single.N_kkt
    nnz_per = single.nnz

    total_rows = Nb * N_kkt
    total_nnz = Nb * nnz_per

    rowPtr = np.empty(total_rows + 1, dtype=np.int32)
    colIdx = np.empty(total_nnz, dtype=np.int32)

    # single.rowPtr gives the per-row nnz counts for one system
    # row_counts = np.diff(single.rowPtr).astype(np.int32)  # (N_kkt,)

    for b in range(Nb):
        # rowPtr segment: shift by b*nnz_per
        rowPtr[b * N_kkt : (b + 1) * N_kkt] = single.rowPtr[:N_kkt] + b * nnz_per
        # colIdx segment: shift column indices by b*N_kkt
        colIdx[b * nnz_per : (b + 1) * nnz_per] = single.colIdx + b * N_kkt

    rowPtr[Nb * N_kkt] = total_nnz

    return BatchedKKTCSRPattern(
        rowPtr=rowPtr,
        colIdx=colIdx,
        nnz=total_nnz,
        N_kkt_total=total_rows,
        single=single,
        Nb=Nb,
    )


def assemble_batched_kkt_csr_values(
    D: jnp.ndarray,  # (N+1, n, n) or (Nb, N+1, n, n)
    E: jnp.ndarray,  # (N, n, n) or (Nb, N, n, n)
    A0: jnp.ndarray,  # (m0, n) or (Nb, m0, n)
    A_minus: jnp.ndarray,  # (N, m, n) or (Nb, N, m, n)
    A_plus: jnp.ndarray,  # (N, m, n) or (Nb, N, m, n)
    G: jnp.ndarray,  # (N+1, m_g, n) or (Nb, N+1, m_g, n)
    pattern: BatchedKKTCSRPattern,
) -> jnp.ndarray:
    """Assemble flattened CSR values for Nb block-diagonal KKT systems.

    Any input may be batched or unbatched. Under JAX batching, some leaves are
    shared across all mapped elements (passed without a batch axis), while
    others carry a leading Nb axis. We detect this from ndim and set in_axes
    accordingly so shared leaves are broadcast by vmap.

    Returns a 1D array of length Nb * nnz_per_sys.
    """
    import jax

    single = pattern.single

    # Determine in_axes: 0 if the array has a leading Nb dimension, None otherwise.
    in_ax_D = 0 if D.ndim == 4 else None
    in_ax_E = 0 if E.ndim == 4 else None
    in_ax_A0 = 0 if A0.ndim == 3 else None
    in_ax_Am = 0 if A_minus.ndim == 4 else None
    in_ax_Ap = 0 if A_plus.ndim == 4 else None
    in_ax_G = 0 if G.ndim == 4 else None

    def assemble_one(D_i, E_i, A0_i, Am_i, Ap_i, G_i):
        return assemble_kkt_csr_values(D_i, E_i, A0_i, Am_i, Ap_i, G_i, single)
 
    # Assemble once and replicate across the batch if every leaf is unbatched (vmap cannot infer an axis).
    if (
        in_ax_D is None
        and in_ax_E is None
        and in_ax_A0 is None
        and in_ax_Am is None
        and in_ax_Ap is None
        and in_ax_G is None
    ):
        values_one = assemble_one(D, E, A0, A_minus, A_plus, G)
        return jnp.tile(values_one, (pattern.Nb,))

    values_batched = jax.vmap(
        assemble_one,
        in_axes=(in_ax_D, in_ax_E, in_ax_A0, in_ax_Am, in_ax_Ap, in_ax_G),
    )(D, E, A0, A_minus, A_plus, G)
    return values_batched.ravel()  # (Nb * nnz_per_sys,)
