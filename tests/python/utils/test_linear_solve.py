import numpy as np
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.linear_systems_solvers.linear_solve import (
    block_tridi_apply,
    block_tridi_to_dense,
    solve_block_tridi_system,
)


def _random_schur_blocks(T, n, seed=0):
    rng = np.random.default_rng(seed)
    blocks = rng.standard_normal((T, n, 3 * n))
    for t in range(T):
        diag = blocks[t, :, n : 2 * n]
        diag = diag + np.eye(n) * (3.0 * n)
        blocks[t, :, n : 2 * n] = diag
    return blocks


def test_block_tridi_to_dense_matches_apply():
    T = 4
    n = 3
    rng = np.random.default_rng(1)
    S = _random_schur_blocks(T, n, seed=0)
    x = rng.standard_normal((T, n))

    y_blocks = block_tridi_apply(S, x)
    A = block_tridi_to_dense(S)
    y_dense = (A @ x.reshape(-1)).reshape(T, n)

    np.testing.assert_allclose(y_blocks, y_dense, rtol=1e-10, atol=1e-10)


def test_solve_block_tridi_system_matches_dense():
    T = 3
    n = 2
    rng = np.random.default_rng(2)
    S = _random_schur_blocks(T, n, seed=3)
    b = rng.standard_normal((T, n))

    x_jax_dense = solve_block_tridi_system(S, b, backend=SchurSolverBackend.JAX_DENSE)
    A = block_tridi_to_dense(S)
    x_dense = np.linalg.solve(A, b.reshape(-1)).reshape(T, n)

    np.testing.assert_allclose(x_jax_dense, x_dense, rtol=1e-10, atol=1e-10)
