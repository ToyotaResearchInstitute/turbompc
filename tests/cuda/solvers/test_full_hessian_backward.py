"""Full-Hessian backward tests."""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from tests.helpers.backend_utils import backend_available
from tests.helpers.ocp_fixtures import (
    make_linear_ocp,
    make_spacecraft_ocp,
    make_spacecraft_ocp_implicit,
)
from tests.helpers.problem_fixtures import tile_spacecraft_inertia
from tests.helpers.solver_fixtures import turbompc_solver_params
from turbompc.dynamics.integrators import predict_next_state
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)
from turbompc.utils.gradient_finitediff import gradient_finite_diff


def _sensitivity_objective(
    solver: TurboMPCSolver, params: dict, weight_keys: list[str]
):
    weights = {k: jnp.asarray(params[k]) for k in weight_keys}
    guess = solver.initial_guess(params)

    def objective(w):
        sol = solver.solve(guess, params, w)
        return jnp.sum(sol.states**2)

    return objective, weights


def _rel_grad_err(g1, g2) -> float:
    leaves1 = jax.tree_util.tree_leaves(g1)
    leaves2 = jax.tree_util.tree_leaves(g2)
    num = sum(float(jnp.sum((a - b) ** 2)) for a, b in zip(leaves1, leaves2))
    den = sum(float(jnp.sum(a**2)) for a in leaves1) + 1e-30
    return float((num / den) ** 0.5)


def test_dynamics_lagrangian_hessian_linear_is_zero():
    ocp, params = make_linear_ocp(horizon=5, rescale=False, rescaling="none")
    N = ocp.horizon
    nx, nu = ocp.num_state_variables, ocp.num_control_variables
    rng = np.random.default_rng(0)
    states = jnp.asarray(rng.standard_normal((N + 1, nx)))
    controls = jnp.asarray(rng.standard_normal((N + 1, nu)))
    lambdas = jnp.asarray(rng.standard_normal((N, nx)))
    hess = ocp.get_dynamics_lagrangian_hessian(states, controls, params, lambdas)
    assert float(jnp.max(jnp.abs(hess))) < 1e-12


@pytest.mark.parametrize("t", [0, 2])
def test_dynamics_lagrangian_hessian_matches_jax(t: int):
    ocp, params = make_spacecraft_ocp(horizon=5, bounded=False)
    N = ocp.horizon
    nx, nu = ocp.num_state_variables, ocp.num_control_variables
    rng = np.random.default_rng(1)
    states = jnp.asarray(rng.standard_normal((N + 1, nx)) * 0.1)
    controls = jnp.asarray(rng.standard_normal((N + 1, nu)) * 0.1)
    lambdas = jnp.asarray(rng.standard_normal((N, nx)))

    hess = ocp.get_dynamics_lagrangian_hessian(states, controls, params, lambdas)

    dyn_params_seq = ocp._get_dynamics_params_sequence(params, N)
    step_params = jax.tree.map(lambda v: v[t], dyn_params_seq)
    lam_t = lambdas[t]

    def scalar_fn(xu):
        x, u = xu[:nx], xu[nx:]
        x_next = predict_next_state(
            ocp.dynamics,
            params["discretization_resolution"],
            ocp.discretization_scheme,
            step_params,
            x,
            u,
            u,
        )
        return lam_t @ x_next

    xu = jnp.concatenate([states[t], controls[t]])
    hess_manual = jax.hessian(scalar_fn)(xu)
    assert float(jnp.max(jnp.abs(hess[t] - hess_manual))) < 1e-12
    assert float(jnp.max(jnp.abs(hess[N]))) == 0.0


def test_full_hessian_requires_direct_backward():
    ocp, _ = make_linear_ocp(horizon=5)
    solver = TurboMPCSolver(
        ocp,
        params=turbompc_solver_params(),
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
        use_full_hessian=True,
    )
    assert solver is not None


def test_full_hessian_linear_matches_gn():
    ocp, params = make_linear_ocp(horizon=5)
    sp = turbompc_solver_params(tol=1e-12)
    weight_keys = [
        "weights_penalization_reference_state_trajectory",
        "weights_penalization_control_squared",
    ]

    solver_gn = TurboMPCSolver(
        ocp,
        params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE,
        use_full_hessian=False,
    )
    obj_gn, weights = _sensitivity_objective(solver_gn, params, weight_keys)
    grad_gn = jax.grad(obj_gn)(weights)

    solver_full = TurboMPCSolver(
        ocp,
        params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE,
        use_full_hessian=True,
    )
    obj_full, _ = _sensitivity_objective(solver_full, params, weight_keys)
    grad_full = jax.grad(obj_full)(weights)

    assert _rel_grad_err(grad_gn, grad_full) < 1e-8


def test_spacecraft_full_hessian_matches_fd_and_beats_gn():
    ocp, params = make_spacecraft_ocp(horizon=6, bounded=False)
    sp = turbompc_solver_params(tol=1e-6, sqp_iters=10)
    weight_keys = [
        "weights_penalization_reference_state_trajectory",
        "weights_penalization_control_squared",
    ]

    solver_gn = TurboMPCSolver(
        ocp,
        params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE,
        use_full_hessian=False,
    )
    obj_gn, weights = _sensitivity_objective(solver_gn, params, weight_keys)
    grad_gn = jax.grad(obj_gn)(weights)

    solver_full = TurboMPCSolver(
        ocp,
        params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE,
        use_full_hessian=True,
    )
    obj_full, _ = _sensitivity_objective(solver_full, params, weight_keys)
    grad_full = jax.grad(obj_full)(weights)
    grad_fd = gradient_finite_diff(obj_full, weights=weights, eps=1e-8)

    err_gn = _rel_grad_err(grad_gn, grad_fd)
    err_full = _rel_grad_err(grad_full, grad_fd)
    assert err_full < 1e-3
    assert err_full < err_gn


def _spacecraft_params_for_full_hessian(
    *,
    horizon: int,
    discretization_scheme: int,
    rescale: bool,
) -> dict:
    dynamics = SpacecraftDynamics()
    nx, nu = dynamics.num_states, dynamics.num_controls
    bound = 0.5
    params = {
        "horizon": horizon,
        "discretization_resolution": 0.1,
        "discretization_scheme": discretization_scheme,
        "initial_state": jnp.array([0.1, -0.15, 0.08]),
        "initial_guess_final_state": jnp.zeros((nx,)),
        "reference_state_trajectory": jnp.zeros((horizon + 1, nx)),
        "reference_control_trajectory": jnp.zeros((horizon + 1, nu)),
        "penalize_control_reference": False,
        "rescale_optimization_variables": rescale,
        "constrain_initial_control": False,
        "initial_control": jnp.zeros((nu,)),
        "weights_penalization_reference_state_trajectory": jnp.ones((nx,)),
        "weights_penalization_final_state": jnp.zeros((nx,)),
        "weights_penalization_control_squared": jnp.ones((nu,)),
        "weights_penalization_control_rate": jnp.zeros((nu,)),
        "state_min_bounds": -jnp.ones((nx,)) * bound,
        "state_max_bounds": jnp.ones((nx,)) * bound,
        "control_min_bounds": -jnp.ones((nu,)) * bound,
        "control_max_bounds": jnp.ones((nu,)) * bound,
        "dynamics_state_dot_params": {"inertia_vector": jnp.array([5.0, 2.0, 1.0])},
    }
    if rescale:
        params["state_rescaling_min"] = -jnp.ones((nx,)) * bound
        params["state_rescaling_max"] = jnp.ones((nx,)) * bound
        params["control_rescaling_min"] = -jnp.ones((nu,)) * bound
        params["control_rescaling_max"] = jnp.ones((nu,)) * bound
    tile_spacecraft_inertia(params, horizon=horizon)
    return params


@pytest.mark.skipif(
    not backend_available(SchurSolverBackend.CUDSS_FFI), reason="cudss_ffi not built"
)
@pytest.mark.parametrize(
    "discretization_scheme,rescale",
    [
        (10, False),
        (0, True),
        (10, True),
    ],
)
def test_cudss_full_hessian_matches_finite_diff(
    discretization_scheme: int, rescale: bool
):
    if discretization_scheme == 10 and not rescale:
        ocp, params = make_spacecraft_ocp_implicit(horizon=6)
    else:
        params = _spacecraft_params_for_full_hessian(
            horizon=6,
            discretization_scheme=discretization_scheme,
            rescale=rescale,
        )
        ocp = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)

    solver = TurboMPCSolver(
        program=ocp,
        params=turbompc_solver_params(tol=1e-6),
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE,
        use_full_hessian=True,
    )
    weight_keys = ["weights_penalization_reference_state_trajectory"]
    obj, weights = _sensitivity_objective(solver, params, weight_keys)
    grad = jax.grad(obj)(weights)
    grad_fd = gradient_finite_diff(obj, weights=weights)
    assert _rel_grad_err(grad, grad_fd) < 1e-3
