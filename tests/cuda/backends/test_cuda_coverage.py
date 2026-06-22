"""CUDA backend coverage using processors.

Exercises CUDA backends across selected discretization schemes, dynamics,
bound modes, rescaling variants, full-Hessian gradients, and parameter mutation.

Each test uses OCPProcessor + SolverProcessor to generate test matrices
without stacking @pytest.mark.parametrize decorators in each test.
"""
import jax
import jax.numpy as jnp
import pytest
from tests.cuda.helpers.processors import OCPProcessor, OCPSpec, SolverProcessor
from tests.helpers.assertions import (
    assert_equality_residual,
    assert_solution_nontrivial,
)
from turbompc.solvers.turbompc_solver import BackwardBackend, ForwardBackend

CUDA_DEFAULT_FORWARD_BACKENDS = [ForwardBackend.ADMM_FUSED_CUDSS]
CUDA_EXTENDED_FORWARD_BACKENDS = [
    ForwardBackend.ADMM_FUSED_CUDSS,
    ForwardBackend.ADMM_FUSED_PCG,
]


def _assert_meaningful_solution(ocp, params, sol, *, eq_tol=1e-3):
    assert_solution_nontrivial(sol)
    assert_equality_residual(ocp, sol, params, tol=eq_tol)


@pytest.mark.extended
@pytest.mark.parametrize(
    "ocp_spec",
    OCPProcessor.parametrize(
        dynamics=["spacecraft"],
        horizon=[5, 10],
        discretization=["euler", "implicit"],
    ),
)
@pytest.mark.parametrize(
    "solver_spec",
    SolverProcessor.parametrize(
        forward_backend=CUDA_EXTENDED_FORWARD_BACKENDS,
    ),
)
def test_cuda_integrators(ocp_spec, solver_spec):
    """Selected discretization schemes work on CUDA backends."""
    ocp, params = OCPProcessor.build(ocp_spec)
    try:
        solver = SolverProcessor.build(ocp, params, solver_spec)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"backend not built: {e}")
    sol = solver.solve(solver.initial_guess(params), params)
    jax.block_until_ready(sol.states)
    _assert_meaningful_solution(ocp, params, sol)


@pytest.mark.extended
@pytest.mark.parametrize(
    "ocp_spec",
    OCPProcessor.parametrize(
        dynamics=["spacecraft"],
        horizon=[5],
        bounds_mode=["both", "control_only", "state_only"],
    ),
)
@pytest.mark.parametrize(
    "solver_spec",
    SolverProcessor.parametrize(
        forward_backend=CUDA_EXTENDED_FORWARD_BACKENDS,
    ),
)
def test_cuda_bounds_modes(ocp_spec, solver_spec):
    """Different box-bound configurations work on CUDA backends."""
    ocp, params = OCPProcessor.build(ocp_spec)
    try:
        solver = SolverProcessor.build(ocp, params, solver_spec)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"backend not built: {e}")
    sol = solver.solve(solver.initial_guess(params), params)
    jax.block_until_ready(sol.states)
    _assert_meaningful_solution(ocp, params, sol)


@pytest.mark.extended
@pytest.mark.parametrize(
    "ocp_spec",
    OCPProcessor.parametrize(
        dynamics=["linear"],
        horizon=[5],
        rescale=[True],
        rescaling_mode=["unit", "linspace"],
    ),
)
@pytest.mark.parametrize(
    "solver_spec",
    SolverProcessor.parametrize(
        forward_backend=CUDA_EXTENDED_FORWARD_BACKENDS,
    ),
)
def test_cuda_rescaling(ocp_spec, solver_spec):
    """State/control rescaling works on CUDA backends."""
    ocp, params = OCPProcessor.build(ocp_spec)
    try:
        solver = SolverProcessor.build(ocp, params, solver_spec)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"backend not built: {e}")
    sol = solver.solve(solver.initial_guess(params), params)
    jax.block_until_ready(sol.states)
    _assert_meaningful_solution(ocp, params, sol)


@pytest.mark.extended
@pytest.mark.parametrize(
    "ocp_spec",
    OCPProcessor.parametrize(
        dynamics=["linear", "spacecraft"],
        horizon=[5, 10],
    ),
)
@pytest.mark.parametrize(
    "solver_spec",
    SolverProcessor.parametrize(
        forward_backend=CUDA_EXTENDED_FORWARD_BACKENDS,
    ),
)
def test_cuda_dynamics_sweep(ocp_spec, solver_spec):
    """CUDA backends handle linear + spacecraft dynamics."""
    ocp, params = OCPProcessor.build(ocp_spec)
    try:
        solver = SolverProcessor.build(ocp, params, solver_spec)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"backend not built: {e}")
    sol = solver.solve(solver.initial_guess(params), params)
    jax.block_until_ready(sol.states)
    _assert_meaningful_solution(ocp, params, sol)


@pytest.mark.extended
@pytest.mark.parametrize(
    "solver_spec",
    SolverProcessor.parametrize(
        forward_backend=[ForwardBackend.ADMM_FUSED_CUDSS],
        admm_alpha=[1.0, 1.6],
        admm_rho_bar=[0.01, 0.1, 1.0],
    ),
)
def test_cuda_admm_hyperparams(solver_spec):
    """Different ADMM hyperparameters (alpha, rho_bar) all converge."""
    ocp, params = OCPProcessor.build(OCPSpec(dynamics="spacecraft", horizon=5))
    solver = SolverProcessor.build(ocp, params, solver_spec)
    sol = solver.solve(solver.initial_guess(params), params)
    jax.block_until_ready(sol.states)
    _assert_meaningful_solution(ocp, params, sol)


@pytest.mark.extended
@pytest.mark.parametrize(
    "ocp_spec",
    OCPProcessor.parametrize(
        dynamics=["spacecraft"],
        horizon=[5, 10],
    ),
)
@pytest.mark.parametrize(
    "solver_spec",
    SolverProcessor.parametrize(
        forward_backend=[ForwardBackend.ADMM_FUSED_CUDSS],
        backward_backend=[BackwardBackend.DIRECT_CUDSS_FFI],
        use_full_hessian=[False, True],
    ),
)
def test_cuda_full_hessian_gradient(ocp_spec, solver_spec):
    """Full Hessian backward produces finite gradients."""
    ocp, params = OCPProcessor.build(ocp_spec)
    try:
        solver = SolverProcessor.build(ocp, params, solver_spec)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"backend not built: {e}")
    guess = solver.initial_guess(params)
    diff_fn = solver.get_differentiable_solve_function()

    weights = {
        "weights_penalization_reference_state_trajectory": jnp.asarray(
            params["weights_penalization_reference_state_trajectory"]
        ),
    }

    def loss(w):
        sol = diff_fn(guess, params, w)
        return jnp.sum(sol.states**2)

    g = jax.grad(loss)(weights)
    jax.block_until_ready(g)
    for key, grad_val in g.items():
        assert jnp.isfinite(grad_val).all(), f"non-finite gradient for {key}"
        assert float(jnp.max(jnp.abs(grad_val))) > 1e-10, f"zero gradient for {key}"


@pytest.mark.parametrize(
    "solver_spec",
    SolverProcessor.parametrize(
        forward_backend=CUDA_DEFAULT_FORWARD_BACKENDS,
    ),
)
def test_cuda_mutate_cost_weight_resolve(solver_spec):
    """Mutating a cost weight after build and re-solving changes the solution.

    Regression guard: the solver reads the param dict fresh on every solve call.
    """
    ocp, params = OCPProcessor.build(
        OCPSpec(
            dynamics="spacecraft",
            horizon=10,
            discretization="euler",
        )
    )
    try:
        solver = SolverProcessor.build(ocp, params, solver_spec)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"backend not built: {e}")

    guess = solver.initial_guess(params)

    sol1 = solver.solve(guess, params)
    jax.block_until_ready(sol1.states)

    params["weights_penalization_reference_state_trajectory"] = (
        jnp.asarray(params["weights_penalization_reference_state_trajectory"]) * 100.0
    )
    sol2 = solver.solve(guess, params)
    jax.block_until_ready(sol2.states)

    assert jnp.isfinite(sol2.states).all()
    diff = float(jnp.max(jnp.abs(sol1.states - sol2.states)))
    assert diff > 1e-4, f"solution unchanged after mutation: diff={diff:.2e}"


@pytest.mark.parametrize(
    "solver_spec",
    SolverProcessor.parametrize(
        forward_backend=CUDA_DEFAULT_FORWARD_BACKENDS,
    ),
)
def test_cuda_mutate_slack_weight_resolve(solver_spec):
    """Mutating slack_penalization_weight after build changes the solution.

    Without rebuilding the solver, a larger slack penalty must shrink ||slack||².
    Uses tight post-build control bounds so slack is active at the optimum.

    Regression guard — previously the CUDA paths cached slack_weight at solver
    construction (turbompc_solver.py self._slack_weight) and ignored param mutation.
    """
    ocp, params = OCPProcessor.build(
        OCPSpec(
            dynamics="linear",
            horizon=5,
            ocp_variant="slack",
            slack_weight=1.0,
        )
    )
    params["control_min_bounds"] = -jnp.ones_like(params["control_min_bounds"]) * 0.05
    params["control_max_bounds"] = jnp.ones_like(params["control_max_bounds"]) * 0.05

    try:
        solver = SolverProcessor.build(ocp, params, solver_spec)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"backend not built: {e}")

    guess = solver.initial_guess(params)

    params["slack_penalization_weight"] = jnp.array(1e-3)
    sol_low = solver.solve(guess, params)
    jax.block_until_ready(sol_low.states)

    params["slack_penalization_weight"] = jnp.array(1e3)
    sol_high = solver.solve(guess, params)
    jax.block_until_ready(sol_high.states)

    assert jnp.isfinite(sol_low.states).all()
    assert jnp.isfinite(sol_high.states).all()

    traj_diff = float(jnp.max(jnp.abs(sol_low.states - sol_high.states)))
    assert (
        traj_diff > 1e-3
    ), f"states unchanged after slack-weight mutation: {traj_diff:.2e}"

    slack_norm_low = float(jnp.sum(sol_low.slack**2))
    slack_norm_high = float(jnp.sum(sol_high.slack**2))
    assert slack_norm_high < slack_norm_low / 100.0, (
        "higher slack weight did not sufficiently reduce ||slack||²: "
        f"low={slack_norm_low:.3e}, high={slack_norm_high:.3e}"
    )
