"""QP containers and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class QPCostBlocks:
    """Block-tridiagonal cost P and linear term q in stage coordinates."""

    D: jnp.ndarray  # (N+1, n, n) diagonal blocks
    E: jnp.ndarray  # (N, n, n) lower off-diagonal blocks
    q: jnp.ndarray  # (N+1, n)

    def tree_flatten(self):
        return (self.D, self.E, self.q), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        D, E, q = children
        return cls(D=D, E=E, q=q)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class QPEqualityBlocks:
    """Block-bidiagonal equality operator Cx = c in stage coordinates."""

    A0: jnp.ndarray  # (n0, n)
    A_minus: jnp.ndarray  # (N, nx, n) coefficients for x_t
    A_plus: jnp.ndarray  # (N, nx, n) coefficients for x_{t+1}
    c0: jnp.ndarray  # (n0,)
    c: jnp.ndarray  # (N, nx)

    def tree_flatten(self):
        return (self.A0, self.A_minus, self.A_plus, self.c0, self.c), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        A0, A_minus, A_plus, c0, c = children
        return cls(A0=A0, A_minus=A_minus, A_plus=A_plus, c0=c0, c=c)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class QPInequalityBlocks:
    """Block-diagonal inequality operator and bounds."""

    G: jnp.ndarray  # (N+1, m, n)
    l: jnp.ndarray  # (N+1, m)
    u: jnp.ndarray  # (N+1, m)
    slack_penalization_weight: jnp.ndarray = field(
        default_factory=lambda: jnp.array(0.0)
    )
    use_slack_variables: bool = False

    def tree_flatten(self):
        children = (
            self.G,
            self.l,
            self.u,
            self.slack_penalization_weight,
        )
        aux_data = (self.use_slack_variables,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        G, l, u, slack_penalization_weight = children
        (use_slack_variables,) = aux_data
        return cls(
            G=G,
            l=l,
            u=u,
            slack_penalization_weight=slack_penalization_weight,
            use_slack_variables=use_slack_variables,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class QPData:
    """Structured QP data for the paper formulation."""

    cost: QPCostBlocks
    eq: QPEqualityBlocks
    ineq: QPInequalityBlocks

    def tree_flatten(self):
        return (self.cost, self.eq, self.ineq), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        cost, eq, ineq = children
        return cls(cost=cost, eq=eq, ineq=ineq)


def _equality_blocks_from_ocp_mats(
    A0: jnp.ndarray,
    c0: jnp.ndarray,
    As: jnp.ndarray,
    Bs: jnp.ndarray,
    As_next: jnp.ndarray,
    Bs_next: jnp.ndarray,
    c_dyn: jnp.ndarray,
) -> QPEqualityBlocks:
    A_minus = jnp.concatenate([As, Bs], axis=2)
    A_plus = jnp.concatenate([As_next, Bs_next], axis=2)

    return QPEqualityBlocks(
        A0=A0,
        A_minus=A_minus,
        A_plus=A_plus,
        c0=c0,
        c=c_dyn,
    )


def _inequality_blocks_from_ocp_mats(
    ineq_blocks: jnp.ndarray,
    ineq_l: jnp.ndarray,
    ineq_u: jnp.ndarray,
    use_slack_variables: bool = False,
    slack_penalization_weight: jnp.ndarray = jnp.array(0.0),
) -> QPInequalityBlocks:
    return QPInequalityBlocks(
        G=ineq_blocks,
        l=ineq_l,
        u=ineq_u,
        slack_penalization_weight=slack_penalization_weight,
        use_slack_variables=use_slack_variables,
    )


def qpdata_from_ocp_blocks(
    D: jnp.ndarray,
    E: jnp.ndarray,
    q: jnp.ndarray,
    A0: jnp.ndarray,
    c0: jnp.ndarray,
    As_next: jnp.ndarray,
    Bs_next: jnp.ndarray,
    As: jnp.ndarray,
    Bs: jnp.ndarray,
    c_dyn: jnp.ndarray,
    ineq_blocks: jnp.ndarray,
    ineq_l: jnp.ndarray,
    ineq_u: jnp.ndarray,
    use_slack_variables: bool = False,
    slack_penalization_weight: jnp.ndarray = jnp.array(0.0),
) -> QPData:
    """Build QPData from OCP linearization blocks."""
    cost = QPCostBlocks(D=D, E=E, q=q)
    eq = _equality_blocks_from_ocp_mats(
        A0=A0,
        c0=c0,
        As=As,
        Bs=Bs,
        As_next=As_next,
        Bs_next=Bs_next,
        c_dyn=c_dyn,
    )
    ineq = _inequality_blocks_from_ocp_mats(
        ineq_blocks=ineq_blocks,
        ineq_l=ineq_l,
        ineq_u=ineq_u,
        use_slack_variables=use_slack_variables,
        slack_penalization_weight=slack_penalization_weight,
    )
    return QPData(cost=cost, eq=eq, ineq=ineq)


def scale_qp_data(
    qp_data: QPData, state_scale: jnp.ndarray, control_scale: jnp.ndarray
) -> QPData:
    """Scale QP data.

    Column scaling applies the variable change z = S z_hat.
    """
    scale = jnp.concatenate([state_scale, control_scale], axis=0)
    cost = qp_data.cost
    eq = qp_data.eq
    ineq = qp_data.ineq

    D = cost.D * scale[jnp.newaxis, :, jnp.newaxis] * scale[jnp.newaxis, jnp.newaxis, :]
    E = cost.E * scale[jnp.newaxis, :, jnp.newaxis] * scale[jnp.newaxis, jnp.newaxis, :]
    q = cost.q * scale[jnp.newaxis, :]

    A0 = eq.A0 * scale[jnp.newaxis, :]
    A_minus = eq.A_minus * scale[jnp.newaxis, jnp.newaxis, :]
    A_plus = eq.A_plus * scale[jnp.newaxis, jnp.newaxis, :]
    c0 = eq.c0
    c = eq.c

    G = ineq.G * scale[jnp.newaxis, jnp.newaxis, :]
    ineq_scaled = QPInequalityBlocks(
        G=G,
        l=ineq.l,
        u=ineq.u,
        slack_penalization_weight=ineq.slack_penalization_weight,
        use_slack_variables=ineq.use_slack_variables,
    )
    eq_scaled = QPEqualityBlocks(A0=A0, A_minus=A_minus, A_plus=A_plus, c0=c0, c=c)
    cost_scaled = QPCostBlocks(D=D, E=E, q=q)
    return QPData(cost=cost_scaled, eq=eq_scaled, ineq=ineq_scaled)
