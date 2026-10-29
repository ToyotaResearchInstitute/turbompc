"""Log-barrier-specific tests for the JAX-loop ADMM.

Behaviour shared with projection handling is tested by parametrizing the
existing ADMM tests over ``constraint_handling``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.constraint_handling_fixtures import (
    QP_PCG_PARAMS,
    SLACK_WEIGHT,
    assert_interior,
    make_qp,
)
from turbompc.solvers.admm import ADMMSolver
from turbompc.solvers.admm.admm import (
    _apply_C,
    _apply_Ct,
    _apply_G,
    _apply_Gt,
    _apply_P,
)
from turbompc.solvers.admm.inequality_log_barrier import barrier_gradient
from turbompc.solvers.constraint_handling import ConstraintHandling
from turbompc.solvers.linear_systems_solvers.backends import (
    AdmmBackend,
    SchurSolverBackend,
)
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver
from turbompc.solvers.status import ADMMStatus

_BARRIER = ConstraintHandling.LOG_BARRIER
_PROJECTION = ConstraintHandling.PROJECTION


def _make_solver(
    zshape,
    constraint_handling=_BARRIER,
    log_barrier_kappa=1.0e-3,
    *,
    use_slack_variables=False,
) -> ADMMSolver:
    return ADMMSolver(
        zshape=zshape,
        schur_solver=make_schur_solver(
            SchurSolverBackend.JAX_DENSE,
            zshape.horizon,
            zshape.num_states,
            zshape.num_controls,
            pcg_params=QP_PCG_PARAMS,
        ),
        pcg_params=QP_PCG_PARAMS,
        sigma=1.0e-8,
        max_iter=20000,
        eps_abs=1.0e-10,
        eps_rel=1.0e-9,
        adapt_rho_every=5,
        admm_backend=AdmmBackend.JAX_LOOP,
        use_slack_variables=use_slack_variables,
        constraint_handling=constraint_handling,
        log_barrier_kappa=log_barrier_kappa,
    )


@pytest.mark.parametrize(
    ("constraint_handling", "log_barrier_kappa", "message"),
    [
        (_BARRIER, 0.0, "log_barrier_kappa must be positive"),
        (_BARRIER, np.inf, "log_barrier_kappa must be positive"),
        (_PROJECTION, 1.0e-3, "log_barrier_kappa must be zero"),
    ],
)
def test_invalid_log_barrier_kappa_is_rejected(
    constraint_handling, log_barrier_kappa, message
):
    _, zshape = make_qp()
    with pytest.raises(ValueError, match=message):
        _make_solver(zshape, constraint_handling, log_barrier_kappa)


@pytest.mark.parametrize("use_slack", [False, True])
def test_solution_satisfies_the_smoothed_kkt_conditions(use_slack):
    log_barrier_kappa = 1.0e-3
    gamma = SLACK_WEIGHT if use_slack else 0.0
    qp_data, zshape = make_qp("mixed", slack_penalization_weight=gamma)
    result = _make_solver(
        zshape, log_barrier_kappa=log_barrier_kappa, use_slack_variables=use_slack
    ).solve(qp_data=qp_data, rho_bar=1.0)
    state = result.state
    lower, upper = qp_data.ineq.l, qp_data.ineq.u
    w = state.z_g + state.xi_g

    assert result.stats.status == ADMMStatus.SOLVED
    assert_interior(w, lower, upper)
    c_stack = jnp.concatenate([qp_data.eq.c0, qp_data.eq.c.reshape(-1)])
    stationarity = (
        _apply_P(qp_data, state.x_blocks)
        + qp_data.cost.q
        + _apply_Ct(qp_data, state.y_f_0, state.y_f_dyn)
        + _apply_Gt(qp_data, state.y_g)
    )
    residuals = {
        "consensus": _apply_G(qp_data, state.x_blocks) - state.z_g,
        "dynamics": _apply_C(qp_data, state.x_blocks) - c_stack,
        "stationarity": stationarity,
        "barrier dual": state.y_g - barrier_gradient(
            w, lower, upper, log_barrier_kappa
        ),
        # Relaxation stationarity; xi_g is identically zero without slack.
        "slack": gamma * state.xi_g + state.y_g if use_slack else state.xi_g,
    }
    for name, residual in residuals.items():
        assert float(jnp.max(jnp.abs(residual))) < 1.0e-6, name


def test_solution_approaches_projection_as_log_barrier_kappa_decreases():
    qp_data, zshape = make_qp("two_sided")
    reference = (
        _make_solver(zshape, _PROJECTION, 0.0)
        .solve(qp_data=qp_data, rho_bar=1.0)
        .state.x_blocks
    )

    errors = []
    for log_barrier_kappa in (1.0e-2, 1.0e-6):
        state = (
            _make_solver(zshape, log_barrier_kappa=log_barrier_kappa)
            .solve(qp_data=qp_data, rho_bar=1.0)
            .state
        )
        errors.append(float(jnp.max(jnp.abs(state.x_blocks - reference))))

    assert errors[1] < 0.1 * errors[0]
    assert errors[1] < 1.0e-4
