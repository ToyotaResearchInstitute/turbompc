"""Block-tridiagonal linear system utilities.

The Schur systems in this codebase are stored in block-tridiagonal form:
  schur_blocks[t] = [prev | diag | next] with shape (n, 3n).
"""

from __future__ import annotations

from typing import Union

import jax
import jax.numpy as jnp
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend

ArrayLike = Union[jax.Array, jnp.ndarray]


def block_tridi_apply(schur_blocks: ArrayLike, x_blocks: ArrayLike) -> jax.Array:
    """Apply block-tridiagonal blocks to a block vector.

    Args:
        schur_blocks: (T, n, 3n) blocks [prev, diag, next]
        x_blocks: (T, n)

    Returns:
        y_blocks: (T, n)
    """
    S = jnp.asarray(schur_blocks)
    x = jnp.asarray(x_blocks)
    if S.ndim != 3:
        raise ValueError(f"schur_blocks must be rank-3, got {S.ndim}")
    if x.ndim != 2:
        raise ValueError(f"x_blocks must be rank-2, got {x.ndim}")
    if S.shape[0] != x.shape[0]:
        raise ValueError("schur_blocks and x_blocks must have same time dimension")
    n = S.shape[1]
    if S.shape[2] != 3 * n:
        raise ValueError(f"schur_blocks last dim must be 3*n, got {S.shape[2]}")
    if x.shape[1] != n:
        raise ValueError(
            f"x_blocks has wrong block dim. Expected {n}, got {x.shape[1]}"
        )

    x_prev = jnp.concatenate([jnp.zeros((1, n), dtype=x.dtype), x[:-1]], axis=0)
    x_next = jnp.concatenate([x[1:], jnp.zeros((1, n), dtype=x.dtype)], axis=0)
    x_stack = jnp.concatenate([x_prev, x, x_next], axis=-1)  # (T, 3n)
    return jnp.einsum("tij,tj->ti", S, x_stack)


def block_tridi_to_dense(schur_blocks: ArrayLike) -> jax.Array:
    """Convert block-tridiagonal blocks to a dense matrix (JAX)."""
    S = jnp.asarray(schur_blocks)
    if S.ndim != 3:
        raise ValueError(f"schur_blocks must be rank-3, got {S.ndim}")
    T = S.shape[0]
    n = S.shape[1]
    if S.shape[2] != 3 * n:
        raise ValueError(f"schur_blocks last dim must be 3*n, got {S.shape[2]}")

    A0 = jnp.zeros((T * n, T * n), dtype=S.dtype)

    def body(t, A):
        St = jax.lax.dynamic_index_in_dim(S, t, axis=0, keepdims=False)  # (n, 3n)
        left = St[:, :n]
        diag = St[:, n : 2 * n]
        right = St[:, 2 * n :]

        row0 = t * n
        col0 = t * n
        A = jax.lax.dynamic_update_slice(A, diag, (row0, col0))

        def _set_left(A_):
            return jax.lax.dynamic_update_slice(A_, left, (row0, (t - 1) * n))

        def _set_right(A_):
            return jax.lax.dynamic_update_slice(A_, right, (row0, (t + 1) * n))

        A = jax.lax.cond(t > 0, _set_left, lambda x: x, A)
        A = jax.lax.cond(t < T - 1, _set_right, lambda x: x, A)
        return A

    return jax.lax.fori_loop(0, T, body, A0)


def solve_block_tridi_system(
    schur_blocks: ArrayLike,
    rhs_blocks: ArrayLike,
    backend: SchurSolverBackend = SchurSolverBackend.JAX_DENSE,
) -> jax.Array:
    """Solve block-tridiagonal system S x = b using a dense JAX solve."""
    if backend != SchurSolverBackend.JAX_DENSE:
        raise ValueError(f"unknown backend: {backend}")

    S = jnp.asarray(schur_blocks)
    rhs = jnp.asarray(rhs_blocks)
    T = S.shape[0]
    n = S.shape[1]

    if rhs.ndim == 1:
        if rhs.size != T * n:
            raise ValueError(f"rhs has wrong size. Expected {T * n}, got {rhs.size}")
        rhs_flat = rhs
    elif rhs.ndim == 2:
        if rhs.shape != (T, n):
            raise ValueError(f"rhs has wrong shape. Expected {(T, n)}, got {rhs.shape}")
        rhs_flat = rhs.reshape(-1)
    else:
        raise ValueError(f"rhs must be rank-1 or rank-2, got ndim={rhs.ndim}")

    A = block_tridi_to_dense(S)
    x_flat = jnp.linalg.solve(A, rhs_flat)
    return x_flat.reshape((T, n))
