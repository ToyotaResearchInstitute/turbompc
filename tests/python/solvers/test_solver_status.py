from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from tests.helpers.constraint_handling_fixtures import HANDLINGS, is_barrier
from tests.helpers.solver_fixtures import make_tiny_qp
from turbompc.solvers.admm.admm import ADMMSolver, _bound_support, _select_admm_status
from turbompc.solvers.constraint_handling import ConstraintHandling
from turbompc.solvers.linear_systems_solvers.backends import (
    AdmmBackend,
    SchurSolverBackend,
)
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver
from turbompc.solvers.qp_utils import ZShape
from turbompc.solvers.status import ADMMStatus, SQPStatus
from turbompc.solvers.turbompc_solver import _should_stop_after_admm


def _solver(
    *,
    max_iter=20,
    check_termination_every=5,
    constraint_handling=ConstraintHandling.PROJECTION,
    log_barrier_kappa=0.0,
):
    zshape = ZShape(horizon=1, num_states=1, num_controls=0)
    linear_solver = make_schur_solver(
        SchurSolverBackend.JAX_DENSE,
        horizon=1,
        num_states=1,
        num_controls=0,
        pcg_params={},
    )
    return ADMMSolver(
        zshape,
        {},
        schur_solver=linear_solver,
        max_iter=max_iter,
        check_termination_every=check_termination_every,
        check_infeasibility_every=1,
        adapt_rho_every=0,
        eps_abs=1.0e-8,
        eps_rel=1.0e-8,
        admm_backend=AdmmBackend.JAX_LOOP,
        constraint_handling=constraint_handling,
        log_barrier_kappa=log_barrier_kappa,
    )


def test_status_values_and_precedence_match_native_code():
    assert [int(status) for status in ADMMStatus] == [-1, 0, 1, 2, 3, 4]
    assert [int(status) for status in SQPStatus] == [-1, 0, 1, 2, 3]

    header = (
        Path(__file__).parents[3] / "turbompc/solvers/admm/csrc/admm_status.cuh"
    ).read_text()
    for status in ADMMStatus:
        assert f"ADMM_STATUS_{status.name} = {int(status)};" in header

    select = jax.jit(_select_admm_status)
    cases = (
        (True, True, True, True, True, ADMMStatus.NONFINITE),
        (False, True, True, True, True, ADMMStatus.SOLVED),
        (False, False, True, True, True, ADMMStatus.PRIMAL_INFEASIBLE),
        (False, False, False, True, True, ADMMStatus.DUAL_INFEASIBLE),
        (False, False, False, False, True, ADMMStatus.MAX_ITER),
        (False, False, False, False, False, ADMMStatus.NOT_RUN),
    )
    for *flags, expected in cases:
        assert int(select(*flags)) == expected


def test_sqp_continues_finite_qp_failures():
    for status in (
        ADMMStatus.SOLVED,
        ADMMStatus.MAX_ITER,
        ADMMStatus.PRIMAL_INFEASIBLE,
        ADMMStatus.DUAL_INFEASIBLE,
    ):
        assert not bool(_should_stop_after_admm(jnp.int32(status)))
    assert bool(_should_stop_after_admm(jnp.int32(ADMMStatus.NONFINITE)))


@pytest.mark.parametrize(("constraint_handling", "log_barrier_kappa"), HANDLINGS)
@pytest.mark.parametrize(
    ("qp", "expected"),
    [
        (
            make_tiny_qp(q=jnp.zeros((2, 1)), initial_rhs=1.0),
            ADMMStatus.PRIMAL_INFEASIBLE,
        ),
        (make_tiny_qp(q=-jnp.ones((2, 1))), ADMMStatus.DUAL_INFEASIBLE),
        (
            make_tiny_qp(q=jnp.full((2, 1), jnp.nan)),
            ADMMStatus.NONFINITE,
        ),
    ],
)
def test_admm_early_termination_status(
    constraint_handling, log_barrier_kappa, qp, expected
):
    solver = _solver(
        constraint_handling=constraint_handling, log_barrier_kappa=log_barrier_kappa
    )
    result = jax.jit(solver.solve)(qp)
    # The barrier has no infeasibility certificates; it runs to max_iter.
    if is_barrier(log_barrier_kappa) and expected != ADMMStatus.NONFINITE:
        expected, num_iter = ADMMStatus.MAX_ITER, solver.max_iter
    else:
        num_iter = 1
    assert int(result.stats.status) == expected
    assert int(result.stats.num_iter) == num_iter


def test_final_iteration_is_always_checked():
    qp = make_tiny_qp(q=jnp.ones((2, 1)), hessian=jnp.ones((2, 1, 1)))
    result = jax.jit(_solver(max_iter=2, check_termination_every=10).solve)(qp)
    assert int(result.stats.status) == ADMMStatus.SOLVED
    assert result.iterations.residuals_valid.tolist() == [True, True]


def test_infinite_bound_support_avoids_zero_times_infinity():
    lower = jnp.full((2,), -jnp.inf)
    upper = jnp.full((2,), jnp.inf)
    assert float(_bound_support(jnp.zeros((2,)), lower, upper)) == 0.0
    assert jnp.isposinf(_bound_support(jnp.asarray([1.0, 0.0]), lower, upper))
