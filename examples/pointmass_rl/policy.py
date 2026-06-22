"""Small MLP policy used by the point-mass RL example."""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

Policy = Dict[str, jax.Array]


def init_policy(
    rng: jax.Array,
    obs_dim: int = 6,
    hidden: int = 8,
    out_dim: int = 81,
) -> Policy:
    """Initialize a two-layer MLP with a zero output layer.

    The zero output layer makes the initial policy emit zero log-multipliers,
    so the MPC starts from the hand-chosen default cost weights.
    """
    k1, _k2 = jax.random.split(rng)
    return {
        "W1": jax.random.normal(k1, (obs_dim, hidden)) * 0.5,
        "b1": jnp.zeros(hidden),
        "W2": jnp.zeros((hidden, out_dim)),
        "b2": jnp.zeros(out_dim),
    }


def policy_apply(theta: Policy, obs: jax.Array) -> jax.Array:
    """Return raw per-stage log-weight multipliers."""
    h = jnp.tanh(obs @ theta["W1"] + theta["b1"])
    return h @ theta["W2"] + theta["b2"]
