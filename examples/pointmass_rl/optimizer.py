"""Minimal pure-JAX Adam optimizer for the point-mass RL example."""
from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp


def adam_init(params) -> Dict:
    zeros_like = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros_like, "v": zeros_like, "t": jnp.int32(0)}


def adam_step(
    params,
    grads,
    state: Dict,
    *,
    lr: float = 3e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple:
    """Return ``(new_params, new_state)`` after one Adam update."""
    t = state["t"] + 1
    new_m = jax.tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    new_v = jax.tree.map(
        lambda v, g: b2 * v + (1.0 - b2) * (g**2),
        state["v"],
        grads,
    )
    bc1 = 1.0 - b1**t
    bc2 = 1.0 - b2**t
    new_params = jax.tree.map(
        lambda p, m, v: p - lr * (m / bc1) / (jnp.sqrt(v / bc2) + eps),
        params,
        new_m,
        new_v,
    )
    return new_params, {"m": new_m, "v": new_v, "t": t}
