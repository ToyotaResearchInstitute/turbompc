"""Stress tests: push nx, nu, H to find where each backend breaks.

These tests are extended coverage and are skipped unless --run-extended is used.
"""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.helpers.backend_utils import backend_available
from tests.helpers.solver_fixtures import turbompc_solver_params
from turbompc.dynamics.linear_dynamics import LinearDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)


def _make_linear_problem(nx: int, nu: int, horizon: int):
    """Create a random stable linear OCP with given dimensions."""
    np.random.seed(42)
    # Stable A (eigenvalues inside unit circle)
    A_base = np.eye(nx) + 0.05 * np.random.randn(nx, nx)
    lam, V = np.linalg.eig(A_base)
    for i in range(len(lam)):
        if abs(lam[i]) >= 1 - 1e-2:
            lam[i] /= abs(lam[i]) + 1e-2
    A_cont = (V @ np.diag(lam) @ np.linalg.inv(V)).real - np.eye(nx)
    B_cont = 0.1 * np.random.randn(nx, nu)
    b_cont = 0.01 * np.random.randn(nx)

    dynamics_params = {
        "verbose": False,
        "num_states": nx,
        "num_controls": nu,
        "names_states": [f"x{i}" for i in range(nx)],
        "names_controls": [f"u{i}" for i in range(nu)],
    }
    dynamics = LinearDynamics(dynamics_params)
    params = {
        "horizon": horizon,
        "discretization_resolution": 1.0,
        "discretization_scheme": 0,
        "initial_state": jnp.zeros((nx,)),
        "initial_guess_final_state": jnp.zeros((nx,)),
        "reference_state_trajectory": jnp.zeros((horizon + 1, nx)),
        "reference_control_trajectory": jnp.zeros((horizon + 1, nu)),
        "penalize_control_reference": False,
        "rescale_optimization_variables": False,
        "constrain_initial_control": False,
        "initial_control": jnp.zeros((nu,)),
        "state_rescaling_min": -jnp.ones((nx,)),
        "state_rescaling_max": jnp.ones((nx,)),
        "control_rescaling_min": -jnp.ones((nu,)),
        "control_rescaling_max": jnp.ones((nu,)),
        "weights_penalization_reference_state_trajectory": jnp.ones((nx,)),
        "weights_penalization_final_state": jnp.zeros((nx,)),
        "weights_penalization_control_squared": jnp.ones((nu,)),
        "weights_penalization_control_rate": jnp.zeros((nu,)),
        "state_min_bounds": -jnp.ones((nx,)) * 1e7,
        "state_max_bounds": jnp.ones((nx,)) * 1e7,
        "control_min_bounds": -jnp.ones((nu,)) * 10.0,
        "control_max_bounds": jnp.ones((nu,)) * 10.0,
        "dynamics_state_dot_params": {
            "A": jnp.array(A_cont),
            "B": jnp.array(B_cont),
            "b": jnp.array(b_cont),
        },
    }
    return dynamics, params


def _make_solver(fb, bb, problem, admm_eps=1e-4):
    sp = turbompc_solver_params(tol=1e-3, admm_max=500)
    sp["admm"]["eps_abs"] = admm_eps
    sp["admm"]["eps_rel"] = admm_eps
    sp["num_sqp_iteration_max"] = 2
    sp["linesearch"] = False
    return TurboMPCSolver(
        program=problem,
        params=sp,
        forward_backend=fb,
        backward_backend=bb,
    )


STRESS_CASES = [
    (3, 3, 25, "small-like H=25"),
    (3, 3, 80, "small-like H=80"),
    (3, 3, 150, "small-like H=150"),
    (8, 4, 20, "medium n=12 H=20"),
    (8, 4, 50, "medium n=12 H=50"),
    (8, 4, 80, "medium n=12 H=80"),
    (8, 4, 100, "medium n=12 H=100"),
    (16, 8, 30, "large n=24 H=30"),
    (16, 8, 50, "large n=24 H=50"),
    (32, 16, 20, "xlarge n=48 H=20"),
]

STRESS_BACKENDS = [
    (ForwardBackend.ADMM_FUSED_PCG, BackwardBackend.DIRECT_CUDSS_FFI, "fused_pcg"),
    (ForwardBackend.ADMM_FUSED_CUDSS, BackwardBackend.DIRECT_CUDSS_FFI, "fused_cudss"),
    (ForwardBackend.ADMM_JAX_LOOP_PCG, BackwardBackend.ADMM_JAX_LOOP_PCG, "jax_pcg"),
]


def _stress_backend_param(fb, bb, name):
    marks = []
    if not backend_available(fb) or not backend_available(bb):
        marks.append(pytest.mark.skip(reason=f"{name} not built/available"))
    return pytest.param(fb, bb, name, marks=marks, id=name)


@pytest.mark.extended
@pytest.mark.parametrize(
    "nx,nu,horizon,desc",
    [pytest.param(*c, id=c[3]) for c in STRESS_CASES],
)
@pytest.mark.parametrize(
    "fb,bb,backend_name",
    [_stress_backend_param(fb, bb, name) for fb, bb, name in STRESS_BACKENDS],
)
def test_stress_forward(nx, nu, horizon, desc, fb, bb, backend_name):
    """Forward solve: verify it runs without crashing and produces finite output."""
    dynamics, params = _make_linear_problem(nx, nu, horizon)
    problem = OptimalControlProblem(dynamics=dynamics, params=copy.deepcopy(params))
    try:
        solver = _make_solver(fb, bb, problem)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"{backend_name} not built: {e}")
    guess = solver.initial_guess(params)

    try:
        sol = solver.solve(guess, params)
        jax.block_until_ready(sol.states)
    except Exception as e:
        err = str(e)
        if any(
            k in err
            for k in (
                "too many resources",
                "CUDA error",
                "out of memory",
                "invalid argument",
                "cudaError",
            )
        ):
            pytest.skip(
                f"{backend_name} nx={nx} H={horizon}: GPU resource limit ({err[:80]})"
            )
        raise

    assert sol.status == 0, f"Solver failed: status={sol.status}"
    assert jnp.isfinite(sol.states).all(), "Non-finite states"
    assert jnp.isfinite(sol.controls).all(), "Non-finite controls"
