"""Forward full-Hessian QP assembly tests."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.ocp_fixtures import make_linear_ocp, make_spacecraft_ocp
from tests.helpers.problem_fixtures import make_linear_params
from tests.helpers.solver_fixtures import turbompc_solver_params
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.qp_data import (
    QPCostBlocks,
    QPData,
    QPEqualityBlocks,
    QPInequalityBlocks,
)
from turbompc.solvers.status import SQPStatus
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    HessianApproximation,
    TurboMPCSolver,
)


def _make_solver(
    ocp,
    *,
    forward_hessian: HessianApproximation = HessianApproximation.GAUSS_NEWTON,
    forward_hessian_min_eig: float = 1.0e-12,
):
    solver_params = turbompc_solver_params(tol=1.0e-8, sqp_iters=2, admm_max=20)
    solver_params["forward_hessian_min_eig"] = forward_hessian_min_eig
    return TurboMPCSolver(
        ocp,
        params=solver_params,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
        backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
        forward_hessian=forward_hessian,
        backward_hessian=HessianApproximation.GAUSS_NEWTON,
    )


def _assert_qpdata_allclose(lhs: QPData, rhs: QPData):
    lhs_leaves = jax.tree_util.tree_leaves(lhs)
    rhs_leaves = jax.tree_util.tree_leaves(rhs)
    assert len(lhs_leaves) == len(rhs_leaves)
    for lhs_leaf, rhs_leaf in zip(lhs_leaves, rhs_leaves):
        np.testing.assert_allclose(lhs_leaf, rhs_leaf, rtol=1.0e-12, atol=1.0e-12)


def _build_qp(
    solver,
    states,
    controls,
    params,
    *,
    hessian=HessianApproximation.GAUSS_NEWTON,
    scale_variables=True,
    admm_state=None,
    slacks=None,
    mask_inactive_constraints=False,
):
    if slacks is None:
        slacks = jnp.zeros(
            (states.shape[0], solver.program.num_inequality_constraints),
            dtype=states.dtype,
        )
    return solver._build_qp(
        states,
        controls,
        slacks,
        params,
        admm_state,
        hessian=hessian,
        scale_variables=scale_variables,
        mask_inactive_constraints=mask_inactive_constraints,
    )


def _admm_state_with_multipliers(
    solver, states, controls, params, *, y_f_dyn=None, y_g=None
):
    qp_data = _build_qp(solver, states, controls, params)
    admm_state = solver._admm_solver_fwd.initial_state(
        qp_data=qp_data,
        rho_bar=solver.params["admm"]["rho"],
        states0=jnp.zeros_like(states),
        controls0=jnp.zeros_like(controls),
    )
    if y_f_dyn is not None:
        admm_state = admm_state._replace(y_f_dyn=y_f_dyn)
    if y_g is not None:
        admm_state = admm_state._replace(y_g=y_g)
    return admm_state


def _dense_hessian(cost: QPCostBlocks) -> jnp.ndarray:
    D, E = cost.D, cost.E
    T, n, _ = D.shape
    P = jnp.zeros((T * n, T * n), dtype=D.dtype)
    for t in range(T):
        P = P.at[t * n : (t + 1) * n, t * n : (t + 1) * n].set(D[t])
    for t in range(T - 1):
        rows_next = slice((t + 1) * n, (t + 2) * n)
        rows_curr = slice(t * n, (t + 1) * n)
        P = P.at[rows_next, rows_curr].set(E[t])
        P = P.at[rows_curr, rows_next].set(E[t].T)
    return P


def test_forward_hessian_approximation_modes():
    ocp, _ = make_linear_ocp(horizon=3, bounded=True, rescale=False)

    solver_default = _make_solver(ocp)
    assert solver_default._forward_hessian == HessianApproximation.GAUSS_NEWTON

    solver_full = _make_solver(ocp, forward_hessian=HessianApproximation.FULL_HESSIAN)
    assert solver_full._forward_hessian == HessianApproximation.FULL_HESSIAN

    solver_projected = _make_solver(
        ocp,
        forward_hessian=HessianApproximation.FULL_HESSIAN_PROJECTED,
    )
    assert (
        solver_projected._forward_hessian == HessianApproximation.FULL_HESSIAN_PROJECTED
    )


def test_projected_backward_hessian_is_rejected():
    ocp, _ = make_linear_ocp(horizon=3, bounded=True, rescale=False)
    with pytest.raises(ValueError, match="only supported for forward_hessian"):
        TurboMPCSolver(
            ocp,
            params=turbompc_solver_params(),
            forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
            backward_backend=BackwardBackend.ADMM_JAX_LOOP_PCG,
            backward_hessian=HessianApproximation.FULL_HESSIAN_PROJECTED,
        )


def test_forward_base_qp_unchanged_by_default():
    ocp, params = make_linear_ocp(
        horizon=4,
        bounded=True,
        rescale=True,
        rescaling="unit",
    )
    solver = _make_solver(ocp)
    guess = solver.initial_guess(params)

    base = _build_qp(solver, guess.states, guess.controls, params)
    forward = _build_qp(
        solver,
        guess.states,
        guess.controls,
        params,
        hessian=solver._forward_hessian,
    )

    _assert_qpdata_allclose(forward, base)


def test_forward_full_hessian_changes_d_for_nonlinear_dynamics_multipliers():
    ocp, params = make_spacecraft_ocp(horizon=5, bounded=False)
    solver = _make_solver(
        ocp,
        forward_hessian=HessianApproximation.FULL_HESSIAN,
    )
    rng = np.random.default_rng(0)
    states = jnp.asarray(
        0.2 * rng.standard_normal((ocp.horizon + 1, ocp.num_state_variables))
    )
    controls = jnp.asarray(
        0.2 * rng.standard_normal((ocp.horizon + 1, ocp.num_control_variables))
    )
    y_f_dyn = jnp.asarray(rng.standard_normal((ocp.horizon, ocp.num_state_variables)))
    admm_state = _admm_state_with_multipliers(
        solver, states, controls, params, y_f_dyn=y_f_dyn
    )
    base = _build_qp(solver, states, controls, params, scale_variables=False)
    full = _build_qp(
        solver,
        states,
        controls,
        params,
        hessian=HessianApproximation.FULL_HESSIAN,
        scale_variables=False,
        admm_state=admm_state,
    )

    assert float(jnp.max(jnp.abs(full.cost.D - base.cost.D))) > 1.0e-8
    linearization_blocks = jnp.concatenate([states, controls], axis=-1)
    base_gradient = (
        jnp.einsum("tij,tj->ti", base.cost.D, linearization_blocks) + base.cost.q
    )
    full_gradient = (
        jnp.einsum("tij,tj->ti", full.cost.D, linearization_blocks) + full.cost.q
    )
    np.testing.assert_allclose(full_gradient, base_gradient, rtol=1.0e-10, atol=1.0e-10)


def test_forward_full_hessian_solve_path_runs():
    ocp, params = make_linear_ocp(horizon=3, bounded=True, rescale=False)
    solver = _make_solver(
        ocp, forward_hessian=HessianApproximation.FULL_HESSIAN_PROJECTED
    )

    sol = solver.solve(solver.initial_guess(params), params)

    assert sol.status == SQPStatus.MAX_ITER
    assert jnp.isfinite(sol.convergence_error)


def test_linear_dynamics_and_box_constraints_add_zero_hessian_terms():
    ocp, params = make_linear_ocp(horizon=4, bounded=True, rescale=False)
    solver = _make_solver(
        ocp, forward_hessian=HessianApproximation.FULL_HESSIAN_PROJECTED
    )
    guess = solver.initial_guess(params)
    base = _build_qp(
        solver, guess.states, guess.controls, params, scale_variables=False
    )

    rng = np.random.default_rng(1)
    y_f_dyn = jnp.asarray(rng.standard_normal((ocp.horizon, ocp.num_state_variables)))
    y_ineq = jnp.asarray(
        rng.standard_normal((ocp.horizon + 1, ocp.num_inequality_constraints))
    )
    admm_state = _admm_state_with_multipliers(
        solver,
        guess.states,
        guess.controls,
        params,
        y_f_dyn=y_f_dyn,
        y_g=y_ineq,
    )
    augmented = _build_qp(
        solver,
        guess.states,
        guess.controls,
        params,
        hessian=HessianApproximation.FULL_HESSIAN,
        scale_variables=False,
        admm_state=admm_state,
    )

    np.testing.assert_allclose(augmented.cost.D, base.cost.D, atol=1.0e-12)


class QuadraticInequalityOCP(OptimalControlProblem):
    def step_inequality_constraints(self, state, control, params):
        del params
        g = jnp.array([state[0] ** 2 + control[0] ** 2])
        return g, jnp.array([-1.0e9]), jnp.array([1.0e9])


def test_nonlinear_inequality_adds_nonzero_hessian_when_active():
    dynamics, params = make_linear_params(
        horizon=3,
        implicit=False,
        bounded=False,
        rescale=False,
    )
    ocp = QuadraticInequalityOCP(dynamics=dynamics, params=params)
    solver = _make_solver(
        ocp, forward_hessian=HessianApproximation.FULL_HESSIAN_PROJECTED
    )
    guess = solver.initial_guess(params)
    base = _build_qp(
        solver, guess.states, guess.controls, params, scale_variables=False
    )
    y_f_dyn = jnp.zeros((ocp.horizon, ocp.num_state_variables))
    y_ineq = jnp.ones((ocp.horizon + 1, ocp.num_inequality_constraints))
    admm_state = _admm_state_with_multipliers(
        solver,
        guess.states,
        guess.controls,
        params,
        y_f_dyn=y_f_dyn,
        y_g=y_ineq,
    )
    augmented = _build_qp(
        solver,
        guess.states,
        guess.controls,
        params,
        hessian=HessianApproximation.FULL_HESSIAN,
        scale_variables=False,
        admm_state=admm_state,
    )

    assert float(jnp.max(jnp.abs(augmented.cost.D - base.cost.D))) > 1.0


def test_inactive_inequality_is_masked_only_when_requested():
    dynamics, params = make_linear_params(
        horizon=3,
        implicit=False,
        bounded=False,
        rescale=False,
    )
    ocp = QuadraticInequalityOCP(dynamics=dynamics, params=params)
    solver = _make_solver(ocp)
    guess = solver.initial_guess(params)
    base = _build_qp(
        solver, guess.states, guess.controls, params, scale_variables=False
    )
    y_ineq = jnp.full(
        (ocp.horizon + 1, ocp.num_inequality_constraints),
        0.5 * solver.params["admm"]["eps_abs"],
    )
    admm_state = _admm_state_with_multipliers(
        solver,
        guess.states,
        guess.controls,
        params,
        y_f_dyn=jnp.zeros((ocp.horizon, ocp.num_state_variables)),
        y_g=y_ineq,
    )

    unmasked = _build_qp(
        solver,
        guess.states,
        guess.controls,
        params,
        hessian=HessianApproximation.FULL_HESSIAN,
        scale_variables=False,
        admm_state=admm_state,
    )
    masked = _build_qp(
        solver,
        guess.states,
        guess.controls,
        params,
        hessian=HessianApproximation.FULL_HESSIAN,
        scale_variables=False,
        admm_state=admm_state,
        mask_inactive_constraints=True,
    )

    assert float(jnp.max(jnp.abs(unmasked.cost.D - base.cost.D))) > 0.0
    np.testing.assert_allclose(masked.cost.D, base.cost.D, atol=1.0e-12)


def test_forward_projection_floors_diagonal_blocks_and_preserves_e():
    ocp, _ = make_linear_ocp(horizon=2, bounded=True, rescale=False)
    min_eig = 0.25
    solver = _make_solver(
        ocp,
        forward_hessian=HessianApproximation.FULL_HESSIAN_PROJECTED,
        forward_hessian_min_eig=min_eig,
    )
    dtype = jnp.float64
    D = jnp.asarray(
        [
            [[-1.0, 0.0], [0.0, 0.0]],
            [[0.1, 0.0], [0.0, -0.2]],
            [[-0.5, 0.0], [0.0, 0.2]],
        ],
        dtype=dtype,
    )
    E = jnp.asarray(
        [
            [[0.3, -0.1], [0.0, 0.2]],
            [[-0.4, 0.0], [0.1, 0.2]],
        ],
        dtype=dtype,
    )
    q = jnp.asarray([[0.2, -0.3], [0.4, 0.1], [-0.2, 0.5]], dtype=dtype)
    eq = QPEqualityBlocks(
        A0=jnp.zeros((0, 2), dtype=dtype),
        A_minus=jnp.zeros((2, 0, 2), dtype=dtype),
        A_plus=jnp.zeros((2, 0, 2), dtype=dtype),
        c0=jnp.zeros((0,), dtype=dtype),
        c=jnp.zeros((2, 0), dtype=dtype),
    )
    ineq = QPInequalityBlocks(
        G=jnp.zeros((3, 0, 2), dtype=dtype),
        l=jnp.zeros((3, 0), dtype=dtype),
        u=jnp.zeros((3, 0), dtype=dtype),
        slack_penalization_weight=jnp.asarray(0.0, dtype=dtype),
    )
    qp_data = QPData(cost=QPCostBlocks(D=D, E=E, q=q), eq=eq, ineq=ineq)

    states = jnp.asarray([[0.4], [-0.2], [0.3]], dtype=dtype)
    controls = jnp.asarray([[-0.1], [0.5], [0.2]], dtype=dtype)
    linearization_blocks = jnp.concatenate([states, controls], axis=-1)
    gradient_before = jnp.einsum("tij,tj->ti", D, linearization_blocks) + q
    projected = solver._project_qp_hessian(qp_data, states=states, controls=controls)
    gradient_after = (
        jnp.einsum("tij,tj->ti", projected.cost.D, linearization_blocks)
        + projected.cost.q
    )

    np.testing.assert_allclose(projected.cost.E, E, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        gradient_after, gradient_before, rtol=1.0e-12, atol=1.0e-12
    )
    offdiag_norms = jnp.linalg.norm(E, axis=(-2, -1))
    expected_floor = jnp.zeros((3,), dtype=dtype)
    expected_floor = expected_floor.at[:-1].add(offdiag_norms)
    expected_floor = expected_floor.at[1:].add(offdiag_norms)
    expected_floor = expected_floor + min_eig
    eigvals = jnp.linalg.eigvalsh(projected.cost.D)
    np.testing.assert_allclose(
        jnp.min(eigvals, axis=1), expected_floor, rtol=1.0e-12, atol=1.0e-12
    )
    assert float(jnp.min(jnp.linalg.eigvalsh(_dense_hessian(projected.cost)))) > 0.0


def test_projection_runs_after_forward_full_hessian_terms_are_added():
    ocp, params = make_spacecraft_ocp(horizon=5, bounded=False)
    solver = _make_solver(
        ocp, forward_hessian=HessianApproximation.FULL_HESSIAN_PROJECTED
    )
    rng = np.random.default_rng(2)
    states = jnp.asarray(
        0.2 * rng.standard_normal((ocp.horizon + 1, ocp.num_state_variables))
    )
    controls = jnp.asarray(
        0.2 * rng.standard_normal((ocp.horizon + 1, ocp.num_control_variables))
    )
    y_f_dyn = jnp.asarray(rng.standard_normal((ocp.horizon, ocp.num_state_variables)))
    admm_state = _admm_state_with_multipliers(
        solver, states, controls, params, y_f_dyn=y_f_dyn
    )
    base = _build_qp(solver, states, controls, params, scale_variables=False)
    captured = {}

    def recording_projection(qp_data, **kwargs):
        captured["D_before_projection"] = qp_data.cost.D
        captured["linearization_point"] = (kwargs["states"], kwargs["controls"])
        return qp_data

    solver._project_qp_hessian = recording_projection
    _build_qp(
        solver,
        states,
        controls,
        params,
        hessian=HessianApproximation.FULL_HESSIAN_PROJECTED,
        scale_variables=False,
        admm_state=admm_state,
    )

    assert "D_before_projection" in captured
    np.testing.assert_allclose(captured["linearization_point"][0], states)
    np.testing.assert_allclose(captured["linearization_point"][1], controls)
    assert float(jnp.max(jnp.abs(captured["D_before_projection"] - base.cost.D))) > 0.0


def test_projection_runs_after_variable_scaling_in_solver_coordinates():
    ocp, params = make_linear_ocp(
        horizon=3,
        bounded=True,
        rescale=True,
        rescaling="unit",
    )
    params["state_rescaling_min"] = -jnp.array([1.0, 10.0, 0.5, 20.0])
    params["state_rescaling_max"] = jnp.array([1.0, 10.0, 0.5, 20.0])
    params["control_rescaling_min"] = -jnp.array([1.0, 5.0e3])
    params["control_rescaling_max"] = jnp.array([1.0, 5.0e3])
    params["weights_penalization_control_rate"] = jnp.array([2.0e4, 1.0e-5])
    solver = _make_solver(
        ocp,
        forward_hessian=HessianApproximation.FULL_HESSIAN_PROJECTED,
    )
    guess = solver.initial_guess(params)
    states = guess.states + 0.1
    controls = guess.controls + jnp.array([0.2, 100.0])
    scaled_states, scaled_controls = solver.program.scale_states_controls(
        states, controls, params
    )
    scaled_base = _build_qp(solver, states, controls, params)
    captured = {}
    original_projection = solver._project_qp_hessian

    def recording_projection(qp_data, **kwargs):
        captured["qp_data"] = qp_data
        captured["linearization_point"] = (kwargs["states"], kwargs["controls"])
        return original_projection(qp_data, **kwargs)

    solver._project_qp_hessian = recording_projection
    projected = _build_qp(
        solver,
        states,
        controls,
        params,
        hessian=HessianApproximation.FULL_HESSIAN_PROJECTED,
    )

    _assert_qpdata_allclose(captured["qp_data"], scaled_base)
    np.testing.assert_allclose(
        captured["linearization_point"][0], scaled_states, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        captured["linearization_point"][1], scaled_controls, rtol=0.0, atol=0.0
    )
    linearization_blocks = jnp.concatenate([scaled_states, scaled_controls], axis=-1)
    linearization_vector = linearization_blocks.reshape(-1)
    gradient_before = _dense_hessian(
        scaled_base.cost
    ) @ linearization_vector + scaled_base.cost.q.reshape(-1)
    gradient_after = _dense_hessian(
        projected.cost
    ) @ linearization_vector + projected.cost.q.reshape(-1)
    np.testing.assert_allclose(
        gradient_after,
        gradient_before,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
