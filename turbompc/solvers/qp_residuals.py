"""QP residuals that do not depend on the inequality-handling method."""

from typing import NamedTuple, Tuple

import jax.numpy as jnp

from turbompc.solvers.admm import ADMMState
from turbompc.solvers.admm.admm import (
    _apply_C,
    _apply_Ct,
    _apply_G,
    _apply_Gt,
    _apply_P,
    _inf_norm,
)
from turbompc.solvers.qp_data import QPData


class _QPFirstOrderErrors(NamedTuple):
    stationarity: jnp.ndarray
    equality: jnp.ndarray
    inequality: jnp.ndarray
    slack_stationarity: jnp.ndarray

    @property
    def convergence_error(self) -> jnp.ndarray:
        residuals = tuple(_inf_norm(error) for error in self)
        return jnp.max(jnp.stack(residuals))


def _compute_qp_constraint_errors(
    qp_data: QPData,
    x_blocks: jnp.ndarray,
    slacks: jnp.ndarray,
    use_slack_variables: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return infinity-norm equality and inequality residuals for a QP."""
    c_stack = jnp.concatenate([qp_data.eq.c0, qp_data.eq.c.reshape(-1)])
    equality = _inf_norm(_apply_C(qp_data, x_blocks) - c_stack)

    inequality_values = _apply_G(qp_data, x_blocks)
    if use_slack_variables:
        inequality_values = inequality_values + slacks
    if qp_data.ineq.G.shape[1] > 0:
        inequality_violation = jnp.maximum(
            jnp.maximum(qp_data.ineq.l - inequality_values, 0.0),
            jnp.maximum(inequality_values - qp_data.ineq.u, 0.0),
        )
        inequality = _inf_norm(inequality_violation)
    else:
        inequality = jnp.array(0.0, dtype=x_blocks.dtype)
    return equality, inequality


def _compute_qp_first_order_errors(
    qp_data: QPData,
    x_blocks: jnp.ndarray,
    slacks: jnp.ndarray,
    admm_state: ADMMState,
    use_slack_variables: bool,
) -> _QPFirstOrderErrors:
    """Compute first-order errors."""
    stationarity = (
        _apply_P(qp_data, x_blocks)
        + qp_data.cost.q
        + _apply_Ct(qp_data, admm_state.y_f_0, admm_state.y_f_dyn)
        + _apply_Gt(qp_data, admm_state.y_g)
    )
    equality, inequality = _compute_qp_constraint_errors(
        qp_data, x_blocks, slacks, use_slack_variables
    )

    if use_slack_variables and qp_data.ineq.G.shape[1] > 0:
        slack_stationarity = _inf_norm(
            qp_data.ineq.slack_penalization_weight * slacks + admm_state.y_g
        )
    else:
        slack_stationarity = jnp.array(0.0, dtype=x_blocks.dtype)

    return _QPFirstOrderErrors(
        stationarity=stationarity,
        equality=equality,
        inequality=inequality,
        slack_stationarity=slack_stationarity,
    )
