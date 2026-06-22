"""
Tests the dynamics-Lagrangian Hessian computation for the backward pass for different configured integrators.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from turbompc.dynamics.cartpole_dynamics import CartpoleDynamics
from turbompc.dynamics.integrators import DiscretizationScheme, predict_next_state
from turbompc.problems.optimal_control_problem import OptimalControlProblem

EXPLICIT_SCHEMES = [
    DiscretizationScheme.EULER,
    DiscretizationScheme.MIDPOINT,
    DiscretizationScheme.RUNGEKUTTA4,
]
NX, NU = 4, 1


def _cartpole_ocp(horizon: int, scheme: int):
    """A small nonlinear cartpole OCP with a chosen explicit integrator (complete params dict)."""
    params = {
        "horizon": horizon,
        "discretization_resolution": 0.04,
        "discretization_scheme": int(scheme),
        "initial_state": jnp.array([0.0, 0.0, 0.5, 0.0]),
        "initial_guess_final_state": jnp.zeros((NX,)),
        "constrain_initial_control": False,
        "initial_control": jnp.zeros((NU,)),
        "reference_state_trajectory": jnp.zeros((horizon + 1, NX)),
        "reference_control_trajectory": jnp.zeros((horizon + 1, NU)),
        "penalize_control_reference": False,
        "weights_penalization_reference_state_trajectory": jnp.array([1.0, 1.0, 10.0, 1.0]),
        "weights_penalization_final_state": jnp.zeros((NX,)),
        "weights_penalization_control_squared": jnp.array([0.1]),
        "weights_penalization_control_rate": jnp.zeros((NU,)),
        "rescale_optimization_variables": False,
        "state_rescaling_min": -jnp.ones((NX,)),
        "state_rescaling_max": jnp.ones((NX,)),
        "control_rescaling_min": -jnp.ones((NU,)),
        "control_rescaling_max": jnp.ones((NU,)),
        "dynamics_state_dot_params": {
            "masscart": 1.0,
            "masspole": 0.1,
            "length": 0.5,
            "gravity": 9.81,
        },
    }
    ocp = OptimalControlProblem(dynamics=CartpoleDynamics(), params=params)
    return ocp, params


@pytest.mark.parametrize("scheme", EXPLICIT_SCHEMES)
def test_dynamics_hessian_matches_configured_integrator(scheme):
    """Each per-stage block must equal λᵀ∇²(predict_next_state(scheme)) — the Hessian of the integrator
    the forward uses. Fails for Midpoint/RK4 if the block is hardcoded to the Euler dt·∇²f term."""
    horizon = 5
    ocp, params = _cartpole_ocp(horizon, scheme)
    N = ocp.horizon
    nx, nu = ocp.num_state_variables, ocp.num_control_variables

    rng = np.random.default_rng(1)
    states = jnp.asarray(rng.standard_normal((N + 1, nx)) * 0.2)
    controls = jnp.asarray(rng.standard_normal((N + 1, nu)) * 0.2)
    lambdas = jnp.asarray(rng.standard_normal((N, nx)))

    hess = ocp.get_dynamics_lagrangian_hessian(states, controls, params, lambdas)
    dyn_params_seq = ocp._get_dynamics_params_sequence(params, N)

    for t in range(N):
        step_params = jax.tree.map(lambda v: v[t], dyn_params_seq)  # noqa: B023

        def scalar_fn(xu, step_params=step_params, lam_t=lambdas[t]):
            x_next = predict_next_state(
                ocp.dynamics,
                params["discretization_resolution"],
                ocp.discretization_scheme,
                step_params,
                xu[:nx],
                xu[nx:],
                xu[nx:],
            )
            return lam_t @ x_next

        hess_manual = jax.hessian(scalar_fn)(jnp.concatenate([states[t], controls[t]]))
        err = float(jnp.max(jnp.abs(hess[t] - hess_manual)))
        assert err < 1e-10, (
            f"scheme={int(scheme)} stage {t}: dynamics Hessian block != grad^2(predict_next_state), "
            f"max|Δ|={err:.2e}"
        )

    # Last block is terminal (no dynamics constraint) — must be exactly zero.
    assert float(jnp.max(jnp.abs(hess[N]))) == 0.0
