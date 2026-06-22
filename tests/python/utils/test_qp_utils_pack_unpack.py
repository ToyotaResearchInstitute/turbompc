import jax.numpy as jnp
import numpy as np
from turbompc.solvers.qp_utils import (
    ZShape,
    pack_box_bounds,
    pack_z,
    slice_z_block,
    unpack_z,
)


def test_pack_unpack_round_trip():
    N = 7
    nx = 3
    nu = 2

    rng = np.random.default_rng(0)
    states = jnp.array(rng.normal(size=(N + 1, nx)))
    controls = jnp.array(rng.normal(size=(N + 1, nu)))

    z = pack_z(states, controls)
    assert z.shape == ((N + 1) * (nx + nu),)

    states2, controls2 = unpack_z(z, ZShape(horizon=N, num_states=nx, num_controls=nu))
    assert states2.shape == states.shape
    assert controls2.shape == controls.shape
    assert jnp.all(states2 == states)
    assert jnp.all(controls2 == controls)


def test_pack_box_bounds_shapes_and_order():
    N = 4
    nx = 2
    nu = 3

    x_lo = jnp.arange((N + 1) * nx, dtype=jnp.float64).reshape(N + 1, nx) - 10.0
    x_hi = x_lo + 100.0
    u_lo = jnp.arange((N + 1) * nu, dtype=jnp.float64).reshape(N + 1, nu) - 20.0
    u_hi = u_lo + 200.0

    z_lo, z_hi = pack_box_bounds(x_lo, x_hi, u_lo, u_hi)
    assert z_lo.shape == ((N + 1) * (nx + nu),)
    assert z_hi.shape == ((N + 1) * (nx + nu),)

    shape = ZShape(horizon=N, num_states=nx, num_controls=nu)

    for t in [0, 2, N]:
        zt_lo = slice_z_block(z_lo, shape, t)
        assert jnp.all(zt_lo[:nx] == x_lo[t])
        assert jnp.all(zt_lo[nx:] == u_lo[t])

        zt_hi = slice_z_block(z_hi, shape, t)
        assert jnp.all(zt_hi[:nx] == x_hi[t])
        assert jnp.all(zt_hi[nx:] == u_hi[t])
