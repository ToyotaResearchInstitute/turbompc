"""Elementwise log-barrier operators for the ADMM and SQP inequality paths."""

from typing import Tuple

import jax.numpy as jnp

_NEWTON_STEPS = 7
_FRACTION_TO_BOUNDARY = 0.99

# Relative distance from a finite bound used when an outer iterate has to be
# moved into the strict interior of the barrier domain.
_INTERIOR_MARGIN_FRACTION = 1.0e-2


def _positive_quadratic_root(offset: jnp.ndarray, kappa: jnp.ndarray) -> jnp.ndarray:
    """Stably compute ``(offset + sqrt(offset**2 + 4*kappa)) / 2``."""
    root = jnp.sqrt(offset * offset + 4.0 * kappa)
    return jnp.where(
        offset >= 0.0,
        0.5 * (offset + root),
        2.0 * kappa / (root - offset),
    )


def _strictly_between(
    value: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray
) -> jnp.ndarray:
    """Round a mathematically interior value toward a representable interior."""
    interior_lower = jnp.nextafter(lower, upper)
    interior_upper = jnp.nextafter(upper, lower)
    return jnp.minimum(jnp.maximum(value, interior_lower), interior_upper)


def _finite_two_sided_log_barrier_prox(
    v: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    kappa: jnp.ndarray,
) -> jnp.ndarray:
    """Proximal operator when both bounds are finite."""
    midpoint = lower + 0.5 * (upper - lower)
    lower_initial = lower + _positive_quadratic_root(v - lower, kappa)
    upper_initial = upper - _positive_quadratic_root(upper - v, kappa)

    # g(midpoint) = midpoint - v puts the root on v's side of the midpoint,
    # where the nearer bound's one-sided prox approximates it.
    initial = jnp.where(
        v <= midpoint,
        jnp.minimum(lower_initial, midpoint),
        jnp.maximum(upper_initial, midpoint),
    )
    initial = _strictly_between(initial, lower, upper)

    value = initial
    for _ in range(_NEWTON_STEPS):
        lower_clearance = value - lower
        upper_clearance = upper - value
        residual = value - v + kappa * (1.0 / upper_clearance - 1.0 / lower_clearance)
        derivative = 1.0 + kappa * (
            1.0 / (upper_clearance * upper_clearance)
            + 1.0 / (lower_clearance * lower_clearance)
        )
        step = -residual / derivative
        lower_step = jnp.where(step < 0.0, -step, 1.0)
        upper_step = jnp.where(step > 0.0, step, 1.0)
        lower_alpha = _FRACTION_TO_BOUNDARY * lower_clearance / lower_step
        upper_alpha = _FRACTION_TO_BOUNDARY * upper_clearance / upper_step
        alpha = jnp.where(step < 0.0, jnp.minimum(1.0, lower_alpha), 1.0)
        alpha = jnp.where(step > 0.0, jnp.minimum(1.0, upper_alpha), alpha)
        value = _strictly_between(value + alpha * step, lower, upper)

    return value


def prox_two_sided_log_barrier(
    v: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    kappa: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the proximal operator of a two-sided logarithmic barrier.

    Infinite bounds select the corresponding one-sided proximal operator. An
    element with two infinite bounds is returned unchanged. Inputs follow JAX
    broadcasting rules and ``kappa`` must be positive.
    """
    finite_lower = jnp.isfinite(lower)
    finite_upper = jnp.isfinite(upper)

    # Substitute harmless finite bounds before evaluating all JAX branches.
    safe_lower = jnp.where(finite_lower, lower, v - 1.0)
    safe_upper = jnp.where(finite_upper, upper, v + 1.0)
    two_sided = _finite_two_sided_log_barrier_prox(v, safe_lower, safe_upper, kappa)

    lower_clearance = _positive_quadratic_root(v - safe_lower, kappa)
    lower_only = jnp.maximum(
        safe_lower + lower_clearance,
        jnp.nextafter(safe_lower, jnp.asarray(jnp.inf, dtype=v.dtype)),
    )
    upper_clearance = _positive_quadratic_root(safe_upper - v, kappa)
    upper_only = jnp.minimum(
        safe_upper - upper_clearance,
        jnp.nextafter(safe_upper, jnp.asarray(-jnp.inf, dtype=v.dtype)),
    )

    result = jnp.where(finite_lower & finite_upper, two_sided, v)
    result = jnp.where(finite_lower & ~finite_upper, lower_only, result)
    return jnp.where(~finite_lower & finite_upper, upper_only, result)


def log_barrier_prox_update(
    v: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    kappa: jnp.ndarray,
    rho: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(z_g, xi_g)`` for the pure log-barrier ADMM update."""
    z_g = prox_two_sided_log_barrier(v, lower, upper, kappa / rho)
    return z_g, jnp.zeros_like(z_g)


def relaxed_log_barrier_prox_update(
    v: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    kappa: jnp.ndarray,
    rho: jnp.ndarray,
    slack_penalization_weight: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(z_g, xi_g)`` for the slack-relaxed barrier ADMM update."""
    gamma = slack_penalization_weight
    w = prox_two_sided_log_barrier(v, lower, upper, kappa * (1.0 / rho + 1.0 / gamma))
    xi_g = rho / (rho + gamma) * (w - v)
    return w - xi_g, xi_g


def barrier_value(
    s: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    kappa: jnp.ndarray,
) -> jnp.ndarray:
    """Sum ``-kappa*log(s-l) - kappa*log(u-s)`` over all finite bounds.

    Elements outside a finite bound contribute ``+inf``.
    """
    finite_lower = jnp.isfinite(lower)
    finite_upper = jnp.isfinite(upper)
    lower_clearance = jnp.where(finite_lower, s - lower, 1.0)
    upper_clearance = jnp.where(finite_upper, upper - s, 1.0)
    terms = jnp.where(
        (lower_clearance > 0.0) & (upper_clearance > 0.0),
        -kappa
        * (jnp.log(jnp.abs(lower_clearance)) + jnp.log(jnp.abs(upper_clearance))),
        jnp.inf,
    )
    return jnp.sum(terms)


def barrier_gradient(
    s: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    kappa: jnp.ndarray,
) -> jnp.ndarray:
    """Elementwise ``kappa/(u-s) - kappa/(s-l)``, masking infinite bounds."""
    upper_term = jnp.where(jnp.isfinite(upper), kappa / (upper - s), 0.0)
    lower_term = jnp.where(jnp.isfinite(lower), kappa / (s - lower), 0.0)
    return upper_term - lower_term


def initialize_barrier_image(
    value: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
) -> jnp.ndarray:
    """Initialize a barrier image strictly inside every finite bound.

    A finite row with ``lower >= upper`` yields ``nan``.
    """
    finite_lower = jnp.isfinite(lower)
    finite_upper = jnp.isfinite(upper)
    both = finite_lower & finite_upper
    one = jnp.asarray(1.0, dtype=value.dtype)

    safe_lower = jnp.where(finite_lower, lower, value - 1.0)
    safe_upper = jnp.where(finite_upper, upper, value + 1.0)
    width = jnp.where(both, safe_upper - safe_lower, one)
    eps = jnp.asarray(jnp.finfo(value.dtype).eps, dtype=value.dtype)
    bound_scale = jnp.maximum(
        one, jnp.maximum(jnp.abs(safe_lower), jnp.abs(safe_upper))
    )
    roundoff_margin = eps * bound_scale
    two_sided_margin = jnp.maximum(
        _INTERIOR_MARGIN_FRACTION * jnp.minimum(one, width), roundoff_margin
    )
    two_sided_margin = jnp.minimum(two_sided_margin, 0.25 * width)
    two_sided = _strictly_between(
        jnp.clip(value, safe_lower + two_sided_margin, safe_upper - two_sided_margin),
        safe_lower,
        safe_upper,
    )

    lower_scale = jnp.maximum(one, jnp.abs(safe_lower))
    lower_margin = jnp.maximum(
        _INTERIOR_MARGIN_FRACTION * lower_scale, eps * lower_scale
    )
    lower_only = jnp.maximum(value, safe_lower + lower_margin)
    upper_scale = jnp.maximum(one, jnp.abs(safe_upper))
    upper_margin = jnp.maximum(
        _INTERIOR_MARGIN_FRACTION * upper_scale, eps * upper_scale
    )
    upper_only = jnp.minimum(value, safe_upper - upper_margin)

    result = jnp.where(both, two_sided, value)
    result = jnp.where(finite_lower & ~finite_upper, lower_only, result)
    result = jnp.where(~finite_lower & finite_upper, upper_only, result)
    return jnp.where(both & (lower >= upper), jnp.nan, result)


def fraction_to_boundary_step(
    s: jnp.ndarray,
    direction: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    tau: float,
) -> jnp.ndarray:
    """Largest step in ``(0, 1]`` keeping ``s + alpha*direction`` interior.

    At least a fraction ``1 - tau`` of every finite clearance is preserved.
    """
    one = jnp.asarray(1.0, dtype=s.dtype)
    if s.size == 0:
        return one

    descending = direction < 0.0
    ascending = direction > 0.0
    lower_limit = jnp.where(
        jnp.isfinite(lower) & descending,
        tau * (s - lower) / jnp.where(descending, -direction, one),
        jnp.inf,
    )
    upper_limit = jnp.where(
        jnp.isfinite(upper) & ascending,
        tau * (upper - s) / jnp.where(ascending, direction, one),
        jnp.inf,
    )
    limit = jnp.minimum(jnp.min(lower_limit), jnp.min(upper_limit))
    return jnp.minimum(one, jnp.maximum(limit, jnp.zeros_like(one)))
