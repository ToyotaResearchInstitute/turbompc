"""Drone obstacle-avoidance training with SHAC."""

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import torch

from examples.drone_utils import DRONE_X0_BASE, generate_drone_x0
from turbompc.dynamics.integrators import predict_next_state

from ..util.jax_torch import MPC, PlantStep, jax_to_torch, torch_to_jax
from ..util.mpc import MPCBackend, TurboSeed
from .runtime import Transition


class DroneProblem:
    """Learn stage, terminal, and control cost weights for drone MPC."""

    nx, nu, cost_dim = 6, 3, 15
    constrain_initial_control = False
    weight_keys = (
        "weights_penalization_reference_state_trajectory",
        "weights_penalization_final_state",
        "weights_penalization_control_squared",
    )

    def __init__(self, config, *, planning_margin=0.0):
        self.config = config
        self.ocp_base = {
            **config.problem_params,
            "obstacles_radii": (
                config.problem_params["obstacles_radii"] + planning_margin
            ),
        }
        self.N = int(self.ocp_base["horizon"])
        if int(self.ocp_base["discretization_scheme"]) != 2:
            raise ValueError("This minimal drone runtime uses RK4 (scheme 2).")
        centers, radii = (
            self.ocp_base["obstacles_centers"],
            self.ocp_base["obstacles_radii"],
        )
        # A reference through the corridor: above the first obstacle, below the others.
        self.guide_x = jnp.concatenate(
            (jnp.array([DRONE_X0_BASE[0]]), centers[:, 0], jnp.zeros(1))
        )
        self.guide_y = jnp.concatenate(
            (
                jnp.array([DRONE_X0_BASE[1]]),
                centers[:, 1] + jnp.array([1.0, -1.0, -1.0]) * (radii + 0.1),
                jnp.zeros(1),
            )
        )
        self.reference_seed = jax.jit(self.reference_seed)

    def make_ocp(self):
        return self.config.problem_class(self.config.dynamics, self.ocp_base)

    def cost_weights(self, logits, context=None):
        factors = jnp.exp(4.0 * jnp.tanh(logits / 4.0))
        return {
            key: self.ocp_base[key] * factor
            for key, factor in zip(
                self.weight_keys, jnp.split(factors, (6, 12), axis=-1)
            )
        }

    def params_from_context(self, context):
        return {**self.ocp_base, "initial_state": context}, None

    def clip_control(self, controls):
        return jnp.clip(
            controls,
            self.ocp_base["control_min_bounds"],
            self.ocp_base["control_max_bounds"],
        )

    @staticmethod
    def mpc_warm_start_state(predicted_state, context):
        return context

    def build_mpc_inputs(self, costs, context):
        params, _ = self.params_from_context(context)
        states, controls = self.reference_seed(context)
        return params, self.cost_weights(costs), states, controls

    def reference_seed(self, context):
        """Build a dynamically consistent warm start along the corridor."""
        fraction = jnp.linspace(0.0, 1.0, self.N + 1)
        positions = context[:, None, :3] * (1.0 - fraction[None, :, None])
        reference_y = jnp.interp(positions[..., 0], self.guide_x, self.guide_y)
        offset = context[:, 1] - reference_y[:, 0]
        positions = positions.at[..., 1].set(
            reference_y + offset[:, None] * (1.0 - fraction) ** 2
        )
        dt = self.ocp_base["discretization_resolution"]
        velocity = jnp.diff(positions, axis=1) / dt
        dynamics_params = jax.tree.map(
            lambda value: value[0], self.ocp_base["dynamics_state_dot_params"]
        )
        mass = dynamics_params["mass"].reshape(())

        def follow_reference(state, target):
            position, speed = target
            control = self.clip_control(
                mass * (0.5 * (position - state[:, :3]) + speed - state[:, 3:])
            )
            next_state = predict_next_state(
                self.config.dynamics, dt, 2, dynamics_params, state, control
            )
            return next_state, (next_state, control)

        _, (states, controls) = jax.lax.scan(
            follow_reference,
            context,
            (positions[:, 1:].swapaxes(0, 1), velocity.swapaxes(0, 1)),
        )
        # Simulating the reference controls keeps the initial plan dynamically consistent.
        return (
            jnp.concatenate((context[:, None], states.swapaxes(0, 1)), axis=1),
            jnp.concatenate(
                (controls.swapaxes(0, 1), jnp.zeros_like(controls[:1]).swapaxes(0, 1)),
                axis=1,
            ),
        )


class DroneState(NamedTuple):
    x: torch.Tensor
    steps: torch.Tensor
    warm_start: TurboSeed


class DroneRuntime:
    observation_dim, action_dim = 6, 3
    dtype = torch.float64

    def __init__(
        self,
        problem,
        solver_config,
        *,
        device="cuda",
        episode_steps=50,
        control_dt=None,
        reward_weights=None,
        finite_horizon=False,
        reward_clearance=0.0,
    ):
        self.problem = problem
        self.device = torch.device(device)
        self.episode_steps = episode_steps
        self.finite_horizon = finite_horizon
        self.observation_dim = problem.nx + int(finite_horizon)
        params = problem.ocp_base
        mpc_dt = float(params["discretization_resolution"])
        self.dt = mpc_dt if control_dt is None else float(control_dt)
        if not 0 < self.dt <= mpc_dt:
            raise ValueError(
                "control_dt must be positive and no larger than the MPC step"
            )
        self.backend = MPCBackend({"solver": solver_config}, problem)
        self.mpc = MPC(self.backend)
        self.control_low = self.tensor(params["control_min_bounds"])
        self.control_high = self.tensor(params["control_max_bounds"])
        dynamics_params = jax.tree.map(
            lambda value: value[0], params["dynamics_state_dot_params"]
        )
        reward_weights = reward_weights or {
            "state": params[problem.weight_keys[0]],
            "control": params[problem.weight_keys[2]],
            "obstacle": 100.0,
        }
        state_cost = jnp.asarray(reward_weights["state"])
        control_cost = jnp.asarray(reward_weights["control"])
        goal_weight = float(reward_weights.get("goal", 0.0))
        goal_radius = float(reward_weights.get("goal_radius", 0.1))
        if goal_radius <= 0:
            raise ValueError("goal_radius must be positive")
        self.terminal_cost = self.tensor(
            reward_weights.get("terminal_state", [0.0] * problem.nx)
        )
        centers = jnp.asarray(problem.config.problem_params["obstacles_centers"])
        radii = jnp.asarray(problem.config.problem_params["obstacles_radii"])

        def step_one(x, u):
            next_x = predict_next_state(
                problem.config.dynamics, self.dt, 2, dynamics_params, x, u
            )
            # Measure obstacle penetration in metres at the next control state.
            # The small offset keeps the distance gradient finite at an obstacle centre.
            distance = jnp.sqrt(jnp.sum((next_x[:2] - centers) ** 2, axis=-1) + 1e-12)
            penetration = jnp.maximum(radii + reward_clearance - distance, 0.0)
            violation = jnp.sum(penetration**2)
            goal_cost = (
                jnp.sqrt(1.0 + jnp.sum(next_x[:3] ** 2) / goal_radius**2) - 1.0
            )
            reward = -self.dt * (
                jnp.sum(state_cost * next_x**2)
                + jnp.sum(control_cost * u**2)
                + goal_weight * goal_cost
                + reward_weights["obstacle"] * violation
            )
            return next_x, reward, violation

        def transition(x, u, physics=None):
            return jax.vmap(step_one)(x, u), (), u

        self.plant = PlantStep(transition)

        # Shift by elapsed control time, which may be shorter than one MPC interval.
        fraction = self.dt / mpc_dt

        def advance(plan, done):
            def shift(values):
                shifted = (1.0 - fraction) * values[:, :-1] + fraction * values[:, 1:]
                return jnp.concatenate((shifted, values[:, -1:]), axis=1)

            return TurboSeed(
                shift(plan.states), shift(plan.controls), plan.valid & ~done
            )

        self._advance = jax.jit(advance)

    def tensor(self, value):
        return torch.tensor(np.array(value), device=self.device, dtype=self.dtype)

    def reset(self, batch_size, seed, *, evaluation=False, initial_states=None):
        initial = initial_states
        if initial is None:
            initial = (
                np.broadcast_to(DRONE_X0_BASE, (batch_size, self.problem.nx))
                if evaluation
                else generate_drone_x0(batch_size, seed)
            )
        if np.shape(initial) != (batch_size, self.problem.nx):
            raise ValueError("initial_states must have shape (batch_size, 6)")
        x = self.tensor(initial)
        states, controls = self.problem.reference_seed(torch_to_jax(x))
        return DroneState(
            x,
            torch.zeros(batch_size, device=self.device, dtype=torch.int64),
            TurboSeed(states, controls, jnp.ones(batch_size, dtype=bool)),
        )

    def observe(self, state):
        if not self.finite_horizon:
            return state.x
        remaining = (1.0 - state.steps.to(self.dtype) / self.episode_steps).clamp(0, 1)
        return torch.cat((state.x, remaining[:, None]), dim=-1)

    @staticmethod
    def detach(state):
        return state._replace(x=state.x.detach())

    def policy(self, parameters, state, *, retain):
        if retain:
            mean = self.mpc.mean_only(parameters, state.x, state.warm_start)
            result = self.mpc.pop_result()
            valid = jax_to_torch(result.valid, like=state.x)
        else:
            parameters_jax, state_jax = torch_to_jax((parameters, state.x))
            result = self.backend.solve_device(
                parameters_jax,
                state_jax,
                state.warm_start,
                retain=False,
            )
            mean, valid = jax_to_torch((result.mean, result.valid), like=state.x)
        return SimpleNamespace(mean=mean, valid=valid.bool(), iterate=result.iterate)

    def squash_action(self, action):
        return action.clamp(self.control_low, self.control_high)

    def step(self, state, action, policy):
        x, reward, violation, _ = self.plant(state.x, action)
        valid = policy.valid & torch.isfinite(x).all(-1) & torch.isfinite(reward)
        x = torch.where(valid[:, None], x, state.x.detach())
        reward = torch.where(valid, reward, -100.0)
        steps = state.steps + 1
        horizon_reached = steps >= self.episode_steps
        if self.finite_horizon:
            terminal = (self.terminal_cost * x.square()).sum(-1)
            reward = reward - torch.where(horizon_reached & valid, terminal, 0.0)
        timeout = (
            torch.zeros_like(valid) if self.finite_horizon else horizon_reached & valid
        )
        done = horizon_reached | ~valid
        warm_start = self._advance(policy.iterate, torch_to_jax(done))
        next_state = DroneState(x, steps, warm_start)
        return Transition(
            next_state,
            self.observe(next_state),
            reward,
            done,
            timeout,
            valid,
            {"obstacle_violation": violation.detach()},
        )

    def reset_done(self, state, done, seed):
        if not bool(done.any()):
            return state
        fresh = self.reset(len(done), seed)
        return DroneState(
            torch.where(done[:, None], fresh.x, state.x),
            torch.where(done, 0, state.steps),
            self.backend.select_seed_rows(
                fresh.warm_start, state.warm_start, torch_to_jax(done)
            ),
        )
