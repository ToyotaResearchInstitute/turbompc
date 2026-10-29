"""Tests for the elementwise log-barrier operators."""

from fractions import Fraction
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from turbompc.solvers.admm.inequality_log_barrier import (
    barrier_gradient,
    barrier_value,
    fraction_to_boundary_step,
    initialize_barrier_image,
    log_barrier_prox_update,
    prox_two_sided_log_barrier,
    relaxed_log_barrier_prox_update,
)

_BITS = {
    np.dtype(np.float32): (np.uint32, 32),
    np.dtype(np.float64): (np.uint64, 64),
}


def _key(x: float, dtype) -> int:
    """Map a float to an integer whose ordering matches the float ordering."""
    utype, nbits = _BITS[np.dtype(dtype)]
    bits = int(np.array(x, dtype=dtype).view(utype))
    sign = 1 << (nbits - 1)
    return bits if bits < sign else -(bits - sign)


def _from_key(key: int, dtype) -> float:
    utype, nbits = _BITS[np.dtype(dtype)]
    bits = key if key >= 0 else (-key) | (1 << (nbits - 1))
    return float(np.array(bits, dtype=utype).view(dtype))


def _residual_is_positive(
    x: float,
    v: Fraction,
    lower: Optional[Fraction],
    upper: Optional[Fraction],
    kappa: Fraction,
) -> bool:
    x = Fraction(x)
    residual = x - v
    if upper is not None:
        residual += kappa / (upper - x)
    if lower is not None:
        residual -= kappa / (x - lower)
    return residual > 0


def _exact_bracket(
    guess: float, v: float, lower: float, upper: float, kappa: float, dtype
) -> Tuple[float, float]:
    """Adjacent representable ``(a, b)`` with the exact prox in ``[a, b]``."""
    finfo = np.finfo(dtype)
    lower_q = Fraction(lower) if np.isfinite(lower) else None
    upper_q = Fraction(upper) if np.isfinite(upper) else None
    v_q, kappa_q = Fraction(v), Fraction(kappa)
    lo_limit = _key(
        np.nextafter(dtype(lower), dtype(upper)) if lower_q is not None else -finfo.max,
        dtype,
    )
    hi_limit = _key(
        np.nextafter(dtype(upper), dtype(lower)) if upper_q is not None else finfo.max,
        dtype,
    )

    def positive(key):
        return _residual_is_positive(
            _from_key(key, dtype), v_q, lower_q, upper_q, kappa_q
        )

    # The root may lie between a bound and its nearest interior float.
    if positive(lo_limit):
        return _from_key(lo_limit, dtype), _from_key(lo_limit, dtype)
    if not positive(hi_limit):
        return _from_key(hi_limit, dtype), _from_key(hi_limit, dtype)

    # Gallop outward from the guess until the residual changes sign, then bisect.
    start = min(max(_key(guess, dtype), lo_limit), hi_limit)
    if positive(start):
        hi, step = start, 1
        lo = max(hi - step, lo_limit)
        while positive(lo):
            hi, step = lo, 2 * step
            lo = max(hi - step, lo_limit)
    else:
        lo, step = start, 1
        hi = min(lo + step, hi_limit)
        while not positive(hi):
            lo, step = hi, 2 * step
            hi = min(lo + step, hi_limit)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if positive(mid):
            hi = mid
        else:
            lo = mid
    return _from_key(lo, dtype), _from_key(hi, dtype)


def _prox_errors(v, lower, upper, kappa, dtype):
    """Return prox values and errors in units of ``clearance`` and in ulps."""
    v = np.asarray(v, dtype=dtype)
    lower = np.broadcast_to(np.asarray(lower, dtype=dtype), v.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=dtype), v.shape)
    kappa = np.broadcast_to(np.asarray(kappa, dtype=dtype), v.shape)
    actual = np.asarray(
        jax.jit(prox_two_sided_log_barrier)(
            jnp.asarray(v), jnp.asarray(lower), jnp.asarray(upper), jnp.asarray(kappa)
        )
    )
    clearance_errors = np.empty(v.shape)
    ulp_errors = np.empty(v.shape)
    for i in np.ndindex(v.shape):
        a, b = _exact_bracket(
            float(actual[i]),
            float(v[i]),
            float(lower[i]),
            float(upper[i]),
            float(kappa[i]),
            dtype,
        )
        error = max(a - float(actual[i]), float(actual[i]) - b, 0.0)
        clearance = min(
            (
                Fraction(a) - Fraction(float(lower[i]))
                if np.isfinite(lower[i])
                else np.inf
            ),
            (
                Fraction(float(upper[i])) - Fraction(b)
                if np.isfinite(upper[i])
                else np.inf
            ),
        )
        clearance_errors[i] = error / float(clearance) if clearance > 0 else 0.0
        ulp_errors[i] = error / float(np.spacing(dtype(max(abs(a), abs(b)))))
    return actual, clearance_errors, ulp_errors


def _assert_accurate(
    actual, clearance_errors, ulp_errors, v, lower, upper, kappa, dtype
):
    # Accurate to a few roundings of the clearance, or to a few ulps of the
    # solution where the clearance itself is below the representable spacing.
    rtol = 64 * np.finfo(dtype).eps
    ok = (clearance_errors <= rtol) | (ulp_errors <= 4.0)
    if not np.all(ok):
        worst = np.unravel_index(
            np.argmax(np.where(ok, 0.0, clearance_errors)), ok.shape
        )
        pytest.fail(
            f"{np.count_nonzero(~ok)} of {ok.size} prox values inaccurate; worst at"
            f" v={np.asarray(v)[worst]!r},"
            f" lower={np.broadcast_to(lower, ok.shape)[worst]!r},"
            f" upper={np.broadcast_to(upper, ok.shape)[worst]!r},"
            f" kappa={np.broadcast_to(kappa, ok.shape)[worst]!r}:"
            f" result={actual[worst]!r}, clearance error={clearance_errors[worst]:.3e},"
            f" ulp error={ulp_errors[worst]:.3e}"
        )
    lower_b = np.broadcast_to(lower, actual.shape)
    upper_b = np.broadcast_to(upper, actual.shape)
    assert np.all(actual > lower_b) and np.all(actual < upper_b)


_KAPPAS = np.logspace(-20, 10, 16)

# (lower, upper): unit, asymmetric, wide, narrow, offset far from zero, and a
# narrow interval far from zero where clearances approach the float spacing.
_INTERVALS = [
    (0.0, 1.0),
    (-1.3, 2.7),
    (-1e6, 1e6),
    (0.0, 1e-8),
    (-5e-7, 5e-7),
    (1e3, 1e3 + 1.0),
    (-1e3 - 1e-6, -1e3),
]


def _edge_case_inputs(lower: float, upper: float, kappa: float, dtype) -> np.ndarray:
    """Inputs outside, on, just inside, and deep inside the interval."""
    lower, upper = dtype(lower), dtype(upper)
    width = float(upper) - float(lower)
    midpoint = dtype(float(lower) + 0.5 * width)
    root = np.sqrt(kappa)
    values = [
        lower,
        upper,
        midpoint,
        np.nextafter(lower, -np.inf),
        np.nextafter(lower, np.inf),
        np.nextafter(upper, -np.inf),
        np.nextafter(upper, np.inf),
        np.nextafter(midpoint, -np.inf),
        np.nextafter(midpoint, np.inf),
    ]
    scale = max(width, 1.0)
    for distance in (
        1e8 * scale,
        10 * width,
        width,
        1e-3 * width,
        10 * root,
        root,
        1e-3 * root,
    ):
        values += [float(lower) - distance, float(upper) + distance]
    # Interior points near a bound, on the scale of the interval and of
    # sqrt(kappa); inputs with v - lower << sqrt(kappa) used to stall Newton.
    for fraction in (1e-12, 1e-6, 1e-3, 0.1, 0.3):
        values += [float(lower) + fraction * width, float(upper) - fraction * width]
    for multiple in (1e-6, 1e-3, 0.1, 1.0, 10.0):
        if multiple * root < 0.5 * width:
            values += [float(lower) + multiple * root, float(upper) - multiple * root]
    values = np.asarray(values, dtype=dtype)
    return np.unique(values[np.isfinite(values)])


def _grid(dtype):
    vs, lowers, uppers, kappas = [], [], [], []
    for lower, upper in _INTERVALS:
        lo, hi = dtype(lower), dtype(upper)
        # Skip intervals the dtype cannot resolve into many representable values.
        if _key(hi, dtype) - _key(lo, dtype) < 1000:
            continue
        for kappa in _KAPPAS:
            v = _edge_case_inputs(lower, upper, kappa, dtype)
            vs.append(v)
            lowers.append(np.full_like(v, lo))
            uppers.append(np.full_like(v, hi))
            kappas.append(np.full_like(v, dtype(kappa)))
    return tuple(np.concatenate(a) for a in (vs, lowers, uppers, kappas))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_two_sided_prox_matches_exact_root_on_edge_case_grid(dtype):
    v, lower, upper, kappa = _grid(dtype)
    actual, clearance_errors, ulp_errors = _prox_errors(v, lower, upper, kappa, dtype)
    _assert_accurate(
        actual, clearance_errors, ulp_errors, v, lower, upper, kappa, dtype
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_one_sided_prox_matches_exact_root(dtype, side):
    bound = dtype(-1.3)
    v_list, kappa_list = [], []
    for kappa in _KAPPAS:
        root = np.sqrt(kappa)
        v = [bound, np.nextafter(bound, -np.inf), np.nextafter(bound, np.inf)]
        for distance in (1e8, 10.0, 1.0, 1e-3, 10 * root, root, 1e-3 * root):
            v += [float(bound) - distance, float(bound) + distance]
        v = np.unique(np.asarray(v, dtype=dtype))
        v_list.append(v)
        kappa_list.append(np.full_like(v, dtype(kappa)))
    v, kappa = np.concatenate(v_list), np.concatenate(kappa_list)
    if side == "lower":
        lower, upper = np.full_like(v, bound), np.full_like(v, np.inf)
    else:
        lower, upper = np.full_like(v, -np.inf), np.full_like(v, bound)

    actual, clearance_errors, ulp_errors = _prox_errors(v, lower, upper, kappa, dtype)
    _assert_accurate(
        actual, clearance_errors, ulp_errors, v, lower, upper, kappa, dtype
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prox_without_bounds_is_identity(dtype):
    v = np.asarray([-1e8, -1.0, 0.0, 3.5, 1e8], dtype=dtype)
    for kappa in _KAPPAS:
        result = prox_two_sided_log_barrier(
            jnp.asarray(v), -jnp.inf, jnp.inf, jnp.asarray(kappa, dtype=dtype)
        )
        np.testing.assert_array_equal(np.asarray(result), v)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prox_is_monotone_and_reflection_symmetric(dtype):
    lower, upper = dtype(-1.3), dtype(2.7)
    v = np.linspace(-6.0, 7.0, 2001).astype(dtype)
    tol = 8 * np.finfo(dtype).eps * 7.0
    for kappa in _KAPPAS:
        result = np.asarray(
            prox_two_sided_log_barrier(
                jnp.asarray(v), lower, upper, jnp.asarray(kappa, dtype=dtype)
            )
        )
        assert np.all(np.diff(result) >= -tol), f"not monotone for kappa={kappa}"
        # prox(v; l, u) = -prox(-v; -u, -l)
        reflected = -np.asarray(
            prox_two_sided_log_barrier(
                jnp.asarray(-v), -upper, -lower, jnp.asarray(kappa, dtype=dtype)
            )
        )
        np.testing.assert_allclose(result, reflected, rtol=0.0, atol=tol)


def test_interior_input_near_bound_regression():
    # An interior input with v - lower << sqrt(kappa) previously started
    # Newton at v, which only doubles the clearance per step, and returned a
    # value close to v instead of the prox. This stalled ADMM on a scalar QP.
    v = np.array([1e-6, 1e-4, 1e-3, 1.0 - 1e-4], dtype=np.float64)
    for kappa in (1e-2, 1e-3, 1e-4):
        actual, clearance_errors, ulp_errors = _prox_errors(
            v, 0.0, 1.0, kappa, np.float64
        )
        _assert_accurate(
            actual, clearance_errors, ulp_errors, v, 0.0, 1.0, kappa, np.float64
        )


def test_barrier_dominated_narrow_interval_regression():
    # With kappa >> width**2 the one-sided initial guess can land next to the
    # far bound; Newton then crawls back instead of converging.
    v = np.array([-1e4, -1.0, 1.0, 1e4], dtype=np.float64)
    actual, clearance_errors, ulp_errors = _prox_errors(v, 0.0, 1e-6, 1e-2, np.float64)
    _assert_accurate(
        actual, clearance_errors, ulp_errors, v, 0.0, 1e-6, 1e-2, np.float64
    )


_MIXED_V = jnp.array([-3.0, 0.25, 4.0, -2.0, 3.0])
_MIXED_LOWER = jnp.array([-1.0, -1.0, -1.0, -1.0, -jnp.inf])
_MIXED_UPPER = jnp.array([2.0, 2.0, 2.0, jnp.inf, 2.0])


def test_relaxed_update_satisfies_its_optimality_conditions():
    # (z, xi) minimizes phi(z + xi) + gamma/2 xi**2 + rho/2 (z - v)**2, so with
    # w = z + xi both phi'(w) + rho (z - v) and phi'(w) + gamma xi vanish.
    kappa, rho, gamma = 0.1, 2.0, 5.0
    z_g, xi_g = relaxed_log_barrier_prox_update(
        _MIXED_V, _MIXED_LOWER, _MIXED_UPPER, kappa, rho, gamma
    )
    gradient = barrier_gradient(z_g + xi_g, _MIXED_LOWER, _MIXED_UPPER, kappa)

    np.testing.assert_allclose(gradient + rho * (z_g - _MIXED_V), 0.0, atol=1e-12)
    np.testing.assert_allclose(gradient + gamma * xi_g, 0.0, atol=1e-12)


@pytest.mark.parametrize("relaxed", [False, True])
def test_small_kappa_approaches_projection(relaxed):
    projection = jnp.clip(_MIXED_V, _MIXED_LOWER, _MIXED_UPPER)
    kappa, rho, gamma = 1e-12, 1.7, 4.2

    if relaxed:
        z_g, xi_g = relaxed_log_barrier_prox_update(
            _MIXED_V, _MIXED_LOWER, _MIXED_UPPER, kappa, rho, gamma
        )
        expected_xi = rho / (rho + gamma) * (projection - _MIXED_V)
    else:
        z_g, xi_g = log_barrier_prox_update(
            _MIXED_V, _MIXED_LOWER, _MIXED_UPPER, kappa, rho
        )
        expected_xi = jnp.zeros_like(projection)
    np.testing.assert_allclose(xi_g, expected_xi, atol=2e-6)
    np.testing.assert_allclose(z_g, projection - expected_xi, atol=2e-6)


def test_barrier_value_and_gradient():
    s = jnp.array([0.5, 0.5, 0.5, 0.5])
    lower = jnp.array([0.0, -jnp.inf, 0.0, -jnp.inf])
    upper = jnp.array([1.0, 1.0, jnp.inf, jnp.inf])
    kappa = 0.3

    np.testing.assert_allclose(
        jax.grad(barrier_value)(s, lower, upper, kappa),
        barrier_gradient(s, lower, upper, kappa),
        rtol=1e-12,
    )
    for outside in (jnp.array([-0.1, 0.5]), jnp.array([0.5, 1.0])):
        value = barrier_value(outside, lower[:2], upper[:2], kappa)
        assert float(value) == np.inf


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_initialize_barrier_image_is_strictly_interior(dtype):
    lower = jnp.asarray([-1.0, -jnp.inf, 0.0, -jnp.inf, 0.0], dtype=dtype)
    upper = jnp.asarray([1.0, 2.0, jnp.inf, jnp.inf, 1e-30], dtype=dtype)
    value = jnp.asarray([-5.0, 7.0, -3.0, 42.0, 1.0], dtype=dtype)
    inside = initialize_barrier_image(value, lower, upper)

    assert bool(jnp.all(jnp.where(jnp.isfinite(lower), inside > lower, True)))
    assert bool(jnp.all(jnp.where(jnp.isfinite(upper), inside < upper, True)))
    assert float(inside[3]) == 42.0
    empty = initialize_barrier_image(
        jnp.zeros(1, dtype), jnp.ones(1, dtype), -jnp.ones(1, dtype)
    )
    assert bool(jnp.all(jnp.isnan(empty)))


def test_fraction_to_boundary_step():
    s = jnp.array([0.5, 0.5, 0.5])
    lower = jnp.array([0.0, -jnp.inf, 0.0])
    upper = jnp.array([1.0, 1.0, jnp.inf])
    tau = 0.995

    alpha = fraction_to_boundary_step(
        s, jnp.array([-10.0, 10.0, -10.0]), lower, upper, tau
    )
    # Every limiting row keeps exactly a (1 - tau) fraction of its clearance.
    np.testing.assert_allclose(float(alpha), tau * 0.5 / 10.0)

    alpha = fraction_to_boundary_step(s, jnp.array([0.1, -0.1, 0.1]), lower, upper, tau)
    assert float(alpha) == 1.0
