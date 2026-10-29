import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.backend_utils import backend_available
from tests.helpers.problem_fixtures import make_linear_params
from tests.helpers.solver_fixtures import turbompc_solver_params
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.linear_systems_solvers.backends import (
    AdmmBackend,
    SchurSolverBackend,
)
from turbompc.solvers.linear_systems_solvers.eigen_backend import (
    EigenBlockTridiWorkspace,
)
from turbompc.solvers.linear_systems_solvers.linear_solve import (
    solve_block_tridi_system,
)
from turbompc.solvers.status import SQPStatus
from turbompc.solvers.turbompc_solver import (
    _BACKWARD_ADMM_COMPONENTS,
    _FORWARD_BACKEND_COMPONENTS,
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
    parse_backward_backend,
    parse_forward_backend,
)

jax.config.update("jax_enable_x64", True)


requires_eigen = pytest.mark.skipif(
    not backend_available(SchurSolverBackend.EIGEN),
    reason="Eigen backend extension not built",
)


def _blocks(dtype):
    blocks = np.zeros((3, 2, 6), dtype=dtype)
    diagonal = np.array([[4.0, 0.2], [0.2, 3.0]], dtype=dtype)
    upper = np.array([[0.2, 0.1], [-0.05, 0.15]], dtype=dtype)
    blocks[:, :, 2:4] = diagonal
    blocks[:-1, :, 4:] = upper
    blocks[1:, :, :2] = upper.T
    return jnp.asarray(blocks)


@requires_eigen
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_eigen_solve_matches_jax_dense_under_jit(dtype):
    blocks = _blocks(dtype)
    rhs = jnp.arange(6, dtype=blocks.dtype).reshape(3, 2)
    workspace = EigenBlockTridiWorkspace()
    solve = jax.jit(workspace.solve)

    try:
        actual = solve(blocks, rhs, factorize=True)
        expected = solve_block_tridi_system(blocks, rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

        next_rhs = rhs + 1
        actual = solve(blocks, next_rhs, factorize=False)
        expected = solve_block_tridi_system(blocks, next_rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

        changed = blocks.at[:, :, 2:4].add(jnp.eye(2, dtype=blocks.dtype))
        actual = solve(changed, rhs, factorize=True)
        expected = solve_block_tridi_system(changed, rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    finally:
        workspace.close()


@requires_eigen
def test_eigen_solve_callback_fallback(monkeypatch):
    blocks = _blocks(np.float32)
    rhs = jnp.arange(6, dtype=blocks.dtype).reshape(3, 2)
    workspace = EigenBlockTridiWorkspace()
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    try:
        actual = jax.jit(workspace.solve)(blocks, rhs, factorize=True)
        expected = solve_block_tridi_system(blocks, rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

        nonfinite = jax.jit(workspace.solve)(
            blocks, rhs.at[0, 0].set(jnp.nan), factorize=False
        )
        assert jnp.all(jnp.isnan(nonfinite))
    finally:
        workspace.close()


@requires_eigen
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_eigen_workspace_solves_a_batch(dtype):
    first = _blocks(dtype)
    second = first.at[:, :, 2:4].add(jnp.eye(2, dtype=first.dtype))
    blocks = jnp.stack((first, second))
    rhs = jnp.arange(12, dtype=blocks.dtype).reshape(2, 3, 2)
    workspace = EigenBlockTridiWorkspace()

    try:
        solve = jax.jit(workspace.solve)
        actual = solve(blocks, rhs, factorize=True)
        expected = jax.vmap(solve_block_tridi_system)(blocks, rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

        next_rhs = rhs + 1
        actual = solve(blocks, next_rhs, factorize=False)
        expected = jax.vmap(solve_block_tridi_system)(blocks, next_rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    finally:
        workspace.close()


@requires_eigen
@pytest.mark.parametrize("batch_shape", [(1,), (2,), (2, 1)])
def test_eigen_custom_vmap_reuses_one_lazy_workspace(batch_shape):
    blocks = _blocks(np.float64)
    rhs = jnp.arange(6, dtype=blocks.dtype).reshape(3, 2)
    workspace = EigenBlockTridiWorkspace()

    try:
        expected = workspace.solve(blocks, rhs, factorize=True)
        workspace_handle = workspace.handle
        blocks_batched = jnp.broadcast_to(blocks, batch_shape + blocks.shape)
        rhs_batched = jnp.broadcast_to(rhs, batch_shape + rhs.shape)

        def solve_batched(b, r):
            return workspace.solve(b, r, factorize=True)

        for _ in batch_shape:
            solve_batched = jax.vmap(solve_batched)
        actual = jax.jit(solve_batched)(blocks_batched, rhs_batched)

        np.testing.assert_allclose(
            actual,
            jnp.broadcast_to(expected, batch_shape + expected.shape),
            rtol=1e-5,
            atol=1e-6,
        )
        scalar_again = workspace.solve(blocks, rhs, factorize=True)
        assert workspace.handle == workspace_handle
        np.testing.assert_allclose(scalar_again, expected, rtol=1e-5, atol=1e-6)
    finally:
        workspace.close()


@requires_eigen
def test_eigen_backend_runs_turbompc_solve_end_to_end():
    dynamics, params = make_linear_params(
        horizon=3,
        implicit=False,
        bounded=True,
        initial_state=jnp.ones((4,), dtype=jnp.float32),
    )
    problem = OptimalControlProblem(dynamics=dynamics, params=copy.deepcopy(params))
    solver_params = turbompc_solver_params(tol=1e-6, sqp_iters=2, admm_max=30)
    solver = TurboMPCSolver(
        program=problem,
        params=solver_params,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_EIGEN,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_EIGEN,
    )

    solution = jax.jit(solver.solve)(solver.initial_guess(params), params)

    assert solution.status == SQPStatus.MAX_ITER
    assert solution.num_iter > 0
    assert jnp.isfinite(solution.convergence_error)


def test_eigen_backend_parsing_and_mapping():
    forward = parse_forward_backend("admm_jax_loop_eigen")
    backward = parse_backward_backend("admm_jax_loop_eigen")

    assert forward == ForwardBackend.ADMM_JAX_LOOP_EIGEN
    assert backward == BackwardBackend.ADMM_JAX_LOOP_EIGEN
    assert _FORWARD_BACKEND_COMPONENTS[forward] == (
        AdmmBackend.JAX_LOOP,
        SchurSolverBackend.EIGEN,
    )
    assert _BACKWARD_ADMM_COMPONENTS[backward] == (
        AdmmBackend.JAX_LOOP,
        SchurSolverBackend.EIGEN,
    )
