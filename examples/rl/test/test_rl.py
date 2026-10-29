"""Optional RL checks; select JAX_PLATFORMS=cpu or cuda before running pytest."""

import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffrax")
jax.config.update("jax_enable_x64", True)

from examples.drone_utils import OBS_CENTERS, make_drone_config
from examples.rl.env.drone import DroneProblem, DroneRuntime
from examples.rl.racing_training_experiment import build, experiment_config
from examples.rl.util.jax_torch import jax_to_torch, torch_to_jax
from examples.rl.util.training import load_checkpoint, save_checkpoint
from turbompc.dynamics.linear_dynamics import LinearDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import TurboMPCSolver
from turbompc.utils.load_params import load_problem_params, load_solver_params


def test_paper_curve_uses_step_reward_for_different_episode_lengths():
    from examples.rl.racing_training_experiment import learning_curve

    history = [
        {
            "update": 10,
            "eval/return_mean": -100.0,
            "eval/reward_mean": -1.0,
            "time/collect_seconds": 1.0,
            "time/train_seconds": 2.0,
        },
        {
            "update": 20,
            "eval/return_mean": -500.0,
            "eval/reward_mean": -0.5,
            "time/collect_seconds": 1.0,
            "time/train_seconds": 2.0,
        },
    ]
    summary = {
        "method": "ppo_mpc",
        "seed": 0,
        "updates": 20,
        "transitions_per_update": 131072,
        "initial_evaluation": {},
    }
    curve = learning_curve(history, summary, baseline_reward=-0.75)
    assert [curve[step]["return"] for step in (0, 10, 20)] == [-0.75, -1.0, -0.5]
    assert curve[20]["transitions"] == 2621440


@pytest.mark.parametrize("handling,kappa", [("projection", 0.0), ("log_barrier", 1e-3)])
@pytest.mark.parametrize("use_slack", [False, True])
def test_cudss_policy_does_not_depend_on_batch_partners(handling, kappa, use_slack):
    if jax.default_backend() != "gpu":
        pytest.skip("cuDSS replay requires JAX_PLATFORMS=cuda")
    params = load_problem_params("linear.yaml")
    params.update(
        horizon=3,
        reference_state_trajectory=jnp.zeros((4, 4)),
        reference_control_trajectory=jnp.zeros((4, 2)),
        weights_penalization_reference_state_trajectory=jnp.ones(4),
        weights_penalization_control_squared=jnp.ones(2),
        state_min_bounds=jnp.full(4, -jnp.inf),
        state_max_bounds=jnp.full(4, jnp.inf),
        control_min_bounds=jnp.full(2, -0.2),
        control_max_bounds=jnp.full(2, 0.2),
        use_slack_variables=use_slack,
        slack_penalization_weight=10.0 if use_slack else 0.0,
    )
    params["dynamics_state_dot_params"] = {
        key: jnp.broadcast_to(value, (3, *value.shape))
        for key, value in params["dynamics_state_dot_params"].items()
    }
    settings = load_solver_params("turbompc.yaml")
    settings.update(
        constraint_handling=handling, log_barrier_kappa=kappa, num_sqp_iteration_max=1
    )
    settings["linesearch"]["method"] = "none"
    settings["admm"].update(
        eps_abs=1e-3, eps_rel=1e-3, max_iter=400, check_termination_every=5
    )
    solver = TurboMPCSolver(
        OptimalControlProblem(LinearDynamics(), params),
        settings,
        forward_backend="admm_fused_cudss",
        backward_backend="direct_jax_dense",
    )

    def solve(initial_state):
        data = {**params, "initial_state": initial_state}
        return solver.solve(solver.initial_guess(data), data)

    starts = jnp.array([[0.03, -0.02, 0.01, 0.04], [3.0, -2.0, 1.0, 4.0]])
    individual = [jax.jit(solve)(state) for state in starts]
    together = jax.jit(jax.vmap(solve))(starts)
    assert int(individual[0].admm_iters.sum()) < int(individual[1].admm_iters.sum())
    for index, expected in enumerate(individual):
        np.testing.assert_allclose(
            together.states[index], expected.states, rtol=1e-8, atol=1e-8
        )
        np.testing.assert_allclose(
            together.controls[index], expected.controls, rtol=1e-8, atol=1e-8
        )
        np.testing.assert_array_equal(together.admm_iters[index], expected.admm_iters)

    def loss(state):
        return jnp.sum(solve(state).controls ** 2)

    gradients = jax.jit(jax.grad(lambda states: jax.vmap(loss)(states).sum()))(starts)
    for index, state in enumerate(starts):
        expected = jax.jit(jax.grad(loss))(state)
        np.testing.assert_allclose(gradients[index], expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("use_slack", [False, True])
@pytest.mark.parametrize("handling,kappa", [("projection", 0.0), ("log_barrier", 1e-2)])
def test_cached_vjp_matches_autodiff_and_finite_difference(
    use_slack, handling, kappa, monkeypatch
):
    params = load_problem_params("linear.yaml")
    params.update(
        horizon=3,
        reference_state_trajectory=jnp.zeros((4, 4)),
        reference_control_trajectory=jnp.zeros((4, 2)),
        weights_penalization_reference_state_trajectory=jnp.ones(4),
        weights_penalization_control_squared=jnp.ones(2),
        control_min_bounds=jnp.full(2, -0.2),
        control_max_bounds=jnp.full(2, 0.2),
        use_slack_variables=use_slack,
        slack_penalization_weight=10.0 if use_slack else 0.0,
    )
    params["dynamics_state_dot_params"] = {
        key: jnp.broadcast_to(value, (3, *value.shape))
        for key, value in params["dynamics_state_dot_params"].items()
    }
    settings = load_solver_params("turbompc.yaml")
    settings.update(
        constraint_handling=handling,
        log_barrier_kappa=kappa,
        num_sqp_iteration_max=10,
        tol_convergence=1e-8,
    )
    settings["linesearch"]["method"] = "none"
    settings["admm"].update(eps_abs=1e-10, eps_rel=1e-10, max_iter=4000)
    solver = TurboMPCSolver(
        OptimalControlProblem(LinearDynamics(), params),
        settings,
        forward_backend="admm_jax_loop_jax_dense",
        backward_backend="direct_jax_dense",
    )
    key = "weights_penalization_control_squared"
    weights = {key: params[key]}
    seed = solver.initial_guess(params)
    calls, forward = [], solver._solve_impl

    def counted_forward(*args, **kwargs):
        jax.debug.callback(lambda: calls.append(1), ordered=True)
        return forward(*args, **kwargs)

    monkeypatch.setattr(solver, "_solve_impl", counted_forward)

    @jax.jit
    def plan(initial_state, costs):
        solution = solver.solve(seed, {**params, "initial_state": initial_state}, costs)
        return solution.states, solution.controls, solution.slack

    def loss(initial_state, costs):
        return sum(jnp.sum(value**2) for value in plan(initial_state, costs))

    outputs, pullback = jax.vjp(plan, params["initial_state"], weights)
    jax.effects_barrier()
    assert len(calls) == 1
    dstate, dweights = pullback(tuple(2 * value for value in outputs))
    jax.effects_barrier()
    assert len(calls) == 1  # Backward reuses the forward residuals.
    native_state, native_weights = jax.grad(loss, argnums=(0, 1))(
        params["initial_state"], weights
    )
    np.testing.assert_allclose(dweights[key], native_weights[key], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(dstate, native_state, rtol=1e-9, atol=1e-9)
    epsilon = 1e-4
    fd = (
        loss(params["initial_state"], {key: weights[key] + epsilon})
        - loss(params["initial_state"], {key: weights[key] - epsilon})
    ) / (2 * epsilon)
    np.testing.assert_allclose(dweights[key].sum(), fd, rtol=3e-3, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("nested", [False, True])
def test_framework_bridge_owns_its_snapshots(dtype, device, nested):
    if device == "cuda" and (
        not torch.cuda.is_available() or jax.default_backend() != "gpu"
    ):
        pytest.skip("CUDA snapshot checks require Torch and JAX on GPU")
    source = torch.arange(6, dtype=dtype, device=device).reshape(2, 3).T
    value = (source, {"column": source[:, 0], "empty": None}) if nested else source
    expected = jax.tree.map(lambda x: x.cpu().numpy().copy(), value)
    snapshot = torch_to_jax(value)
    source.fill_(-1)
    result = jax_to_torch(snapshot, like=source)
    for actual, saved in zip(jax.tree.leaves(result), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(actual.cpu(), saved)
        actual.fill_(-2)
    for actual, saved in zip(jax.tree.leaves(snapshot), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(actual, saved)


@pytest.mark.parametrize("umax", [1.0, 10.0])
@pytest.mark.parametrize("kappa", [1e-2, 1e-4])
def test_linear_benchmark_pullback_reuses_solve_and_matches_fd(
    umax, kappa, monkeypatch
):
    from benchmarking.linear_system.benchmark_forl_training_throughput import (
        LogBarrierAdapter,
    )
    from benchmarking.linear_system.utils import generate_problem_data

    _, _, A, B, b, initial_state = generate_problem_data(4, 7, n_state=4, n_ctrl=2)
    adapter = LogBarrierAdapter(
        A=A,
        B=B,
        b=b,
        n_state=4,
        n_ctrl=2,
        mpc_horizon=7,
        solver_batch_size=2,
        total_batch_size=4,
        umax=umax,
        tolerance=1e-9,
        admm_max_iterations=4000,
        target_kappa=kappa,
        device=torch.device("cpu"),
    )
    forward_calls = []
    solve_impl = adapter.solver._solve_impl

    def counted_solve(*args, **kwargs):
        jax.debug.callback(
            lambda _: forward_calls.append(1), args[0][0, 0], ordered=True
        )
        return solve_impl(*args, **kwargs)

    monkeypatch.setattr(adapter.solver, "_solve_impl", counted_solve)
    costs = torch.linspace(-0.4, 0.4, 24, dtype=torch.float64).reshape(4, 6)
    costs.requires_grad_()
    state = torch.tensor(initial_state, dtype=torch.float64, requires_grad=True)
    upstream = torch.tensor(
        [[0.2, -0.3], [-0.5, 0.7], [0.4, 0.1], [-0.2, 0.6]], dtype=state.dtype
    )
    # Implicit gradients require convergence; cold SQP steps may be damped.
    for _ in range(5):
        adapter.solve(costs.detach(), state.detach())
    guesses = list(adapter._guesses)
    controls = adapter.solve(costs, state)
    # Later solves may replace warm starts, but not the first call's pullbacks.
    adapter.solve(costs.detach() + 0.2, state.detach() * 0.8)
    jax.effects_barrier()
    assert len(forward_calls) == 28
    gradients = torch.autograd.grad((controls * upstream).sum(), (costs, state))
    jax.effects_barrier()
    assert len(forward_calls) == 28
    assert all(torch.isfinite(value).all() for value in gradients)

    def loss(cost, x):
        adapter._guesses = list(guesses)
        # Converge each perturbation: the implicit VJP describes the optimum.
        for _ in range(5):
            result = adapter.solve(cost, x)
        return (result * upstream).sum().detach()

    rng = np.random.default_rng(12)
    epsilon = 1e-4
    for index, (value, gradient) in enumerate(zip((costs, state), gradients)):
        direction = torch.tensor(rng.normal(size=value.shape), dtype=value.dtype)
        direction /= direction.norm()
        positive = [costs.detach(), state.detach()]
        negative = list(positive)
        positive[index] = positive[index] + epsilon * direction
        negative[index] = negative[index] - epsilon * direction
        fd = (loss(*positive) - loss(*negative)) / (2 * epsilon)
        torch.testing.assert_close(
            (gradient * direction).sum(), fd, rtol=3e-3, atol=1e-5
        )


@pytest.mark.parametrize("method", ["shac_mpc", "ppo_mpc", "shac_direct", "ppo_direct"])
def test_racing_update_replay_and_checkpoint(method, tmp_path):
    config = experiment_config(method, 0, device="cpu", smoke=True)
    if config["policy"] == "direct":
        config["controller"]["pd"]["compile"] = False
    learner = build(config)
    before = [p.detach().clone() for p in learner.actor.parameters()]
    rollout = learner.collect()
    if method.startswith("ppo"):
        obs = rollout.observations.flatten(0, 1)
        inputs = {key: value.flatten(0, 1) for key, value in rollout.inputs.items()}
        mean, valid = learner.runtime.replay(learner.actor(obs), inputs, retain=True)
        replay = (
            learner.distribution(mean).log_prob(rollout.actions.flatten(0, 1)).sum(-1)
        )
        assert valid.all()
        torch.testing.assert_close(
            replay, rollout.logprobs.flatten(), rtol=1e-4, atol=1e-4
        )
    metrics = learner.update(rollout)
    gradient_key = (
        "train/actor_gradient_norm"
        if method.startswith("shac")
        else "train/gradient_norm"
    )
    assert np.isfinite(metrics[gradient_key]) and metrics[gradient_key] > 0
    assert any(
        not torch.equal(a, b) for a, b in zip(before, learner.actor.parameters())
    )
    assert all(torch.isfinite(p).all() for p in learner.actor.parameters())
    query = torch.zeros(1, learner.runtime.observation_dim, dtype=learner.runtime.dtype)
    expected = learner.actor(query).detach()
    checkpoint = tmp_path / "policy.pt"
    save_checkpoint(learner, checkpoint)
    with torch.no_grad():
        for parameter in learner.actor.parameters():
            parameter.add_(1)
    assert load_checkpoint(learner, checkpoint) == 1
    assert not learner.state.steps.any()
    torch.testing.assert_close(learner.actor(query), expected)


@pytest.fixture(scope="module")
def drone():
    config = make_drone_config(
        horizon=4,
        dt=1.0,
        discretization_scheme=2,
        use_slack=False,
        slack_weight=0.0,
        sqp_iter=1,
        admm_max_iter=1000,
        admm_tol=1e-5,
    )
    settings = {
        **config.solver_params,
        "constraint_handling": "log_barrier",
        "log_barrier_kappa": 1e-4,
        "forward_backend": "admm_jax_loop_jax_dense",
        "backward_backend": "direct_jax_dense",
    }
    return DroneRuntime(
        DroneProblem(config),
        settings,
        device="cpu",
        episode_steps=2,
        control_dt=0.02,
        finite_horizon=True,
        reward_weights={"state": [1.0] * 6, "control": [0.01] * 3, "obstacle": 1000.0},
        reward_clearance=0.03,
    )


def test_mpc_builds_reference_only_for_cold_or_fallback_rows(drone, monkeypatch):
    state = drone.reset(2, 0)
    reference, calls = drone.problem.reference_seed, []

    def counted_reference(context):
        jax.debug.callback(lambda: calls.append(1))
        return reference(context)

    monkeypatch.setattr(drone.problem, "reference_seed", counted_reference)
    drone.backend._step_cache.clear()
    context, costs = torch_to_jax(state.x), jnp.zeros((2, 15))
    warm = state.warm_start
    cold = warm._replace(valid=jnp.array([True, False]))
    try:
        expected = drone.backend.solve_device(costs, context, warm, retain=False)
        jax.effects_barrier()
        assert not calls
        for seed, fallback in ((cold, warm), (warm, cold)):
            result = drone.backend.solve_device(
                costs, context, seed, retain=False, history_seed=fallback
            )
            jax.effects_barrier()
            np.testing.assert_allclose(result.mean, expected.mean, rtol=1e-7, atol=1e-7)
        assert len(calls) == 2
    finally:
        drone.backend._step_cache.clear()


def test_drone_retains_and_shifts_one_sqp_solution(drone, monkeypatch):
    assert not drone.backend.solver.program.use_slack_variables
    solve, calls = drone.backend.solve_device, []

    def record(costs, context, seed, **kwargs):
        result = solve(costs, context, seed, **kwargs)
        calls.append(result.stats["sqp_iterations"])
        assert np.all(seed.valid)
        return result

    monkeypatch.setattr(drone.backend, "solve_device", record)
    state = drone.reset(2, 0)
    assert drone.observe(state).shape == (2, 7)
    assert (drone.observe(state)[:, -1] == 1).all()
    costs = state.x.new_zeros((2, 15), requires_grad=True)
    for remaining in (0.5, 0.0):
        policy = drone.policy(costs, state, retain=True)
        step = drone.step(state, policy.mean, policy)
        assert step.valid.all()
        assert (step.observation[:, -1] == remaining).all()
        state = step.state
        np.testing.assert_allclose(
            state.warm_start.states[:, :-1],
            0.98 * policy.iterate.states[:, :-1] + 0.02 * policy.iterate.states[:, 1:],
        )
        np.testing.assert_allclose(
            state.warm_start.controls[:, :-1],
            0.98 * policy.iterate.controls[:, :-1]
            + 0.02 * policy.iterate.controls[:, 1:],
        )
    np.testing.assert_array_equal(calls, [[1, 1], [1, 1]])
    assert step.done.all() and not step.timeout.any()
    gradient = torch.autograd.grad(step.reward.sum(), costs)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    reset = drone.reset_done(state, step.done, seed=7)
    assert not reset.steps.any() and np.all(reset.warm_start.valid)
    assert (drone.observe(reset)[:, -1] == 1).all()
    torch.testing.assert_close(
        torch.autograd.grad(reset.x.sum(), state.x)[0], torch.zeros_like(state.x)
    )


def test_drone_distance_violation_reward_and_gradient(drone):
    # Outside, 1 cm inside, and 3 cm inside the first obstacle's clearance band.
    state = torch.zeros((3, 6), dtype=torch.float64)
    state[:, :2] = torch.tensor(OBS_CENTERS[0])
    state[:, 1] -= torch.tensor([0.35, 0.32, 0.30], dtype=state.dtype)
    state.requires_grad_()
    next_state, reward, violation, _ = drone.plant(
        state, torch.zeros((3, 3), dtype=state.dtype)
    )
    expected_violation = torch.tensor([0.0, 0.01**2, 0.03**2], dtype=state.dtype)
    torch.testing.assert_close(violation, expected_violation, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(next_state, state)
    torch.testing.assert_close(
        reward, -0.02 * (state.square().sum(-1) + 1000.0 * expected_violation)
    )
    gradient = torch.autograd.grad(reward.sum(), state)[0]
    expected_y = -0.02 * (
        2.0 * state[:, 1] + 2000.0 * state.new_tensor([0.0, 0.01, 0.03])
    )
    torch.testing.assert_close(gradient[:, 1], expected_y, atol=1e-9, rtol=1e-8)


def test_drone_goal_and_terminal_reward_gradient(drone):
    runtime = DroneRuntime(
        drone.problem,
        {
            **drone.problem.config.solver_params,
            "forward_backend": "admm_jax_loop_jax_dense",
            "backward_backend": "direct_jax_dense",
        },
        device="cpu",
        episode_steps=2,
        control_dt=0.02,
        finite_horizon=True,
        reward_weights={
            "state": [0.0] * 6,
            "control": [0.0] * 3,
            "obstacle": 0.0,
            "goal": 1.0,
            "goal_radius": 0.1,
            "terminal_state": [10.0] * 6,
        },
    )
    state = runtime.reset(2, 0, initial_states=np.tile([0.2, 0, 0, 0.1, 0, 0], (2, 1)))
    state = state._replace(x=state.x.requires_grad_(), steps=torch.tensor([0, 1]))
    action = torch.zeros((2, 3), dtype=runtime.dtype, requires_grad=True)
    policy = SimpleNamespace(
        valid=torch.ones(2, dtype=torch.bool), iterate=state.warm_start
    )
    transition = runtime.step(state, action, policy)
    assert transition.done.tolist() == [False, True]
    assert not transition.timeout.any()
    torch.testing.assert_close(
        transition.reward[0] - transition.reward[1],
        10.0 * transition.state.x[1].square().sum(),
    )
    dx, du = torch.autograd.grad(transition.reward.sum(), (state.x, action))
    assert dx[:, 0].lt(0).all()  # Moving toward the goal increases reward.
    epsilon = 1e-5
    direction = torch.tensor([1, -1, 0.5, 0.2, -0.3, 0.1], dtype=runtime.dtype)
    with torch.no_grad():
        plus = runtime.step(
            state._replace(x=state.x + epsilon * direction), action + epsilon, policy
        ).reward.sum()
        minus = runtime.step(
            state._replace(x=state.x - epsilon * direction), action - epsilon, policy
        ).reward.sum()
    torch.testing.assert_close(
        (dx * direction).sum() + du.sum(),
        (plus - minus) / (2 * epsilon),
        rtol=1e-6,
        atol=1e-8,
    )


def test_cached_pullbacks_mask_invalid_rows(drone):
    state = drone.reset(2, 0)
    result = drone.backend.solve_device(
        jnp.zeros((2, 15)), torch_to_jax(state.x), state.warm_start
    )
    dx = jnp.zeros_like(result.states)
    du = jnp.zeros_like(result.controls).at[:, 0].set(1.0)
    expected = drone.backend.vjp_plan_device(result.cache, dx, du)

    def corrupt_row(value):
        if jnp.issubdtype(value.dtype, jnp.floating):
            return value.at[1].set(jnp.nan)
        return value

    cache = replace(
        result.cache,
        valid=jnp.array([True, False]),
        pullbacks=jax.tree.map(corrupt_row, result.cache.pullbacks),
    )
    actual = drone.backend.vjp_plan_device(cache, dx, du)
    for key in expected:
        np.testing.assert_allclose(actual[key][0], expected[key][0])
        np.testing.assert_array_equal(actual[key][1], 0.0)
    empty = drone.backend.vjp_plan_device(
        replace(cache, valid=jnp.zeros(2, dtype=bool)), dx, du
    )
    for value in empty.values():
        np.testing.assert_array_equal(value, 0.0)


def test_inactive_evaluation_row_preserves_active_policy_and_fallback(drone):
    state = drone.reset(2, 0)
    costs, context = jnp.zeros((2, 15)), torch_to_jax(state.x)
    expected = drone.backend.solve_device(
        costs, context, state.warm_start, retain=False
    )
    actual = drone.backend.solve_device(
        costs.at[1].set(jnp.nan),
        context.at[1].set(jnp.nan),
        state.warm_start,
        retain=False,
        active=jnp.array([True, False]),
    )
    np.testing.assert_array_equal(actual.valid, [True, False])
    np.testing.assert_allclose(actual.mean[0], expected.mean[0], rtol=1e-7, atol=1e-7)
    np.testing.assert_array_equal(actual.mean[1], state.warm_start.controls[1, 0])
