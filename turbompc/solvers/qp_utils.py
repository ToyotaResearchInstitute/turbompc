"""
QP utilities for packing/unpacking decision variables and bounds.

Canonical packing (must match turbompc linesearch convention):
    z = concat([states, controls], axis=-1).flatten()

Where:
- states shape:  (N+1, nx)
- controls shape: (N+1, nu)
- z shape:       ((N+1)*(nx+nu), )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ZShape:
    """Shapes needed to unpack z."""

    horizon: int  # N (full horizon is N+1)
    num_states: int  # nx
    num_controls: int  # nu

    @property
    def block_dim(self) -> int:
        """Dimension of (x_t, u_t)."""
        return self.num_states + self.num_controls

    @property
    def z_dim(self) -> int:
        """Total length of z."""
        return (self.horizon + 1) * self.block_dim


# Register ZShape as a JAX pytree with no dynamic leaves.
# This allows ZShape to be passed as an argument to custom_vmap / jax.jit
# functions without JAX trying to trace or batch its integer fields.
jax.tree_util.register_pytree_node(
    ZShape,
    lambda z: ([], (z.horizon, z.num_states, z.num_controls)),
    lambda aux, _: ZShape(*aux),
)


def pack_z(states: jnp.ndarray, controls: jnp.ndarray) -> jnp.ndarray:
    """
    Pack (states, controls) into a single vector z.

    Args:
        states:   (N+1, nx)
        controls: (N+1, nu)

    Returns:
        z: ((N+1)*(nx+nu), )
    """
    if states.ndim != 2 or controls.ndim != 2:
        raise ValueError(
            f"states and controls must be rank-2 arrays, got {states.ndim},"
            f" {controls.ndim}"
        )
    if states.shape[0] != controls.shape[0]:
        raise ValueError(
            "states and controls must have the same time dimension. "
            f"Got {states.shape[0]} vs {controls.shape[0]}"
        )

    return jnp.concatenate([states, controls], axis=-1).reshape(-1)


def pack_x(states: jnp.ndarray, controls: jnp.ndarray) -> jnp.ndarray:
    """
    Pack (states, controls) into per-timestep blocks x_t = [x_t, u_t].

    Args:
        states:   (N+1, nx)
        controls: (N+1, nu)

    Returns:
        x_blocks: (N+1, nx+nu)
    """
    if states.ndim != 2 or controls.ndim != 2:
        raise ValueError(
            f"states and controls must be rank-2 arrays, got {states.ndim},"
            f" {controls.ndim}"
        )
    if states.shape[0] != controls.shape[0]:
        raise ValueError(
            "states and controls must have the same time dimension. "
            f"Got {states.shape[0]} vs {controls.shape[0]}"
        )
    return jnp.concatenate([states, controls], axis=-1)


def unpack_x(x_blocks: jnp.ndarray, shape: ZShape) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unpack per-timestep blocks x_t = [x_t, u_t] into (states, controls).

    Args:
        x_blocks: (N+1, nx+nu)
        shape: ZShape(horizon=N, num_states=nx, num_controls=nu)

    Returns:
        states:   (N+1, nx)
        controls: (N+1, nu)
    """
    if x_blocks.ndim != 2:
        raise ValueError(f"x_blocks must be rank-2 array, got ndim={x_blocks.ndim}")
    if x_blocks.shape[0] != shape.horizon + 1:
        raise ValueError(
            f"x_blocks has wrong time dimension. Expected {shape.horizon + 1}, "
            f"got {x_blocks.shape[0]}"
        )
    if x_blocks.shape[1] != shape.block_dim:
        raise ValueError(
            f"x_blocks has wrong block dimension. Expected {shape.block_dim}, "
            f"got {x_blocks.shape[1]}"
        )

    states = x_blocks[:, : shape.num_states]
    controls = x_blocks[:, shape.num_states :]
    return states, controls


def unpack_z(z: jnp.ndarray, shape: ZShape) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unpack a vector z into (states, controls).

    Args:
        z:     ((N+1)*(nx+nu), )
        shape: ZShape(horizon=N, num_states=nx, num_controls=nu)

    Returns:
        states:   (N+1, nx)
        controls: (N+1, nu)
    """
    if z.ndim != 1:
        raise ValueError(f"z must be rank-1 array, got ndim={z.ndim}")

    if z.size != shape.z_dim:
        raise ValueError(
            f"z has wrong size. Expected {shape.z_dim}, got {z.size}. "
            f"(N={shape.horizon}, nx={shape.num_states}, nu={shape.num_controls})"
        )

    z_mat = z.reshape((shape.horizon + 1, shape.block_dim))
    states = z_mat[:, : shape.num_states]
    controls = z_mat[:, shape.num_states :]
    return states, controls


def pack_box_bounds(
    x_lower: jnp.ndarray,
    x_upper: jnp.ndarray,
    u_lower: jnp.ndarray,
    u_upper: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pack box bounds on (states, controls) into bounds on z in canonical order.

    Args:
        x_lower, x_upper: (N+1, nx)
        u_lower, u_upper: (N+1, nu)

    Returns:
        z_lower, z_upper: ((N+1)*(nx+nu), )
    """
    for name, arr in [
        ("x_lower", x_lower),
        ("x_upper", x_upper),
        ("u_lower", u_lower),
        ("u_upper", u_upper),
    ]:
        if arr.ndim != 2:
            raise ValueError(f"{name} must be rank-2 array, got ndim={arr.ndim}")

    if x_lower.shape != x_upper.shape:
        raise ValueError(
            f"x_lower and x_upper shape mismatch: {x_lower.shape} vs {x_upper.shape}"
        )
    if u_lower.shape != u_upper.shape:
        raise ValueError(
            f"u_lower and u_upper shape mismatch: {u_lower.shape} vs {u_upper.shape}"
        )
    if x_lower.shape[0] != u_lower.shape[0]:
        raise ValueError(
            "x bounds and u bounds must have the same time dimension. "
            f"Got {x_lower.shape[0]} vs {u_lower.shape[0]}"
        )

    return pack_z(x_lower, u_lower), pack_z(x_upper, u_upper)


def slice_z_block(z: jnp.ndarray, shape: ZShape, t: int) -> jnp.ndarray:
    """
    Return the per-timestep block z_t = [x_t, u_t] from packed z.

    Args:
        z: packed vector, shape (shape.z_dim,)
        shape: ZShape
        t: timestep index in [0, N]

    Returns:
        z_t: (nx+nu, )
    """
    if t < 0 or t > shape.horizon:
        raise ValueError(f"t must be in [0, {shape.horizon}], got {t}")

    start = t * shape.block_dim
    end = start + shape.block_dim
    return z[start:end]
