"""Seven-state racing MPC in path coordinates.

This code uses synthetic data and parameters, and does not represent any manufacturer's product.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from turbompc.dynamics.vehicle import PathVehicleDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem

from ...util.mpc import TurboSeed
from ...util.training import training_config
from .cost import Q_KEY, R_KEY, CostMap

# Local racing-example state layout:
# [yaw rate, speed, sideslip, rear wheel speed, lateral deviation,
#  relative heading, elapsed time].  The runtime context uses physical path
# evolution in its last channel and ``mpc_initial_state`` maps it to time.
YAW_RATE = 0
SPEED = 1
SIDESLIP = 2
REAR_WHEEL_SPEED = 3
LATERAL_DEVIATION = 4
RELATIVE_HEADING = 5
PATH_EVOLUTION = 6
PLANT_NX = 7
MPC_NX = 7
NU = 3

DEFAULT_Q = jnp.asarray([10.0, 0.02, 1.0, 0.002, 30.0, 5.0, 0.0])
DEFAULT_R = jnp.asarray([0.1, 1.0e-5, 1.0e-5])
DEFAULT_RATE = jnp.asarray([2.0e4, 1.0e-5, 1.0e-9])

CONTROL_MIN = jnp.asarray([-0.5, -4.0e3, -5.0e3])
CONTROL_MAX = jnp.asarray([0.5, 370.0, 0.0])
MPC_STATE_SCALE = jnp.asarray([1.0, 30.0, 0.5, 30.0, 5.0, 1.0, 10.0])
MPC_CONTROL_SCALE = jnp.asarray([0.5, 5.0e3, 5.0e3])


class RacingOptimalControlProblem(OptimalControlProblem):
    """Keep independent upper/lower relaxations, as in the reference RL MPC."""

    def step_inequality_constraints(self, state, control, params):
        values, lower, upper = super().step_inequality_constraints(
            state, control, params
        )
        values = jnp.concatenate((values, -values))
        return values, jnp.full_like(values, -jnp.inf), jnp.concatenate((upper, -lower))


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _shift_nodes_batched(values, progress, ds: float) -> jax.Array:
    """Shift batched trajectories by fractional node offsets."""
    values = jnp.asarray(values)
    progress = jnp.asarray(progress, dtype=values.dtype)
    nodes = values.shape[1]
    can_shift = jnp.isfinite(progress) & (progress >= 0.0)
    offset = jnp.where(can_shift, progress / float(ds), 0.0)
    source = jnp.arange(nodes, dtype=values.dtype)[None] + offset[:, None]
    source = jnp.clip(source, 0.0, float(nodes - 1))
    lower = jnp.floor(source).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, nodes - 1)
    gather_shape = lower.shape + (1,) * (values.ndim - 2)
    lower_values = jnp.take_along_axis(values, lower.reshape(gather_shape), axis=1)
    upper_values = jnp.take_along_axis(values, upper.reshape(gather_shape), axis=1)
    fraction = (source - lower).reshape(gather_shape)
    return lower_values + fraction * (upper_values - lower_values)


class RacingProblem:
    """Configure the racing optimal control problem."""

    nx = MPC_NX
    nu = NU
    context_dim = PLANT_NX
    context_includes_control = False
    action_dim = NU

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        track: Any,
        state_dot_params: Mapping[str, Any],
    ):
        full_config = dict(config)
        env = _section(full_config, "env") or full_config
        solver = _section(full_config, "solver")
        actor = _section(training_config(full_config), "actor")

        self.task = str(env.get("task", "tracking")).strip().lower()
        if self.task != "tracking":
            raise ValueError("racing_minimal supports only env.task=tracking")
        self.ds = float(env.get("ds", 3.0))
        self.N = int(env.get("horizon", 40))
        self.initial_s = float(env.get("initial_s", 37.4))
        if self.ds <= 0.0 or self.N < 1:
            raise ValueError("ds and horizon must be positive")

        self.constrain_initial_control = bool(
            env.get("constrain_initial_control", False)
        )
        self.context_includes_control = self.constrain_initial_control
        self.context_dim = PLANT_NX + (NU if self.constrain_initial_control else 0)
        terminal_linear_weight = np.asarray(
            actor.get(
                "weights_linear_penalization_final_state",
                [0.0] * MPC_NX,
            ),
            dtype=np.float64,
        )
        if terminal_linear_weight.shape != (MPC_NX,) or not np.all(
            np.isfinite(terminal_linear_weight)
        ):
            raise ValueError(
                "actor.weights_linear_penalization_final_state must "
                f"contain {MPC_NX} finite values"
            )
        self.terminal_linear_weight = jnp.asarray(terminal_linear_weight)
        self.track = track
        self.mpc_state_dot_params = dict(state_dot_params)
        self.mpc_model = PathVehicleDynamics()

        self.use_slack = bool(solver.get("use_slack", True))
        self.slack_weight = float(solver.get("slack_weight", 1.0e3))
        self.control_min = jnp.asarray(
            env.get("control_min_bounds", CONTROL_MIN), dtype=jnp.float64
        )
        self.control_max = jnp.asarray(
            env.get("control_max_bounds", CONTROL_MAX), dtype=jnp.float64
        )
        if self.control_min.shape != (NU,) or self.control_max.shape != (NU,):
            raise ValueError(f"env.control_min/max_bounds must have shape ({NU},)")
        if bool(jnp.any(self.control_min >= self.control_max)):
            raise ValueError("env control bounds must satisfy min < max")

        self.mpc_state_scale = MPC_STATE_SCALE
        self.state_min = jnp.asarray(
            env.get(
                "state_min_bounds",
                [-1.0e9, 0.5, -1.0e9, -1.0e9, -4.0, -1.0e9, -1.0e9],
            ),
            dtype=jnp.float64,
        )
        self.state_max = jnp.asarray(
            env.get(
                "state_max_bounds",
                [1.0e9, 30.0, 1.0e9, 1.0e9, 4.0, 1.0e9, 1.0e9],
            ),
            dtype=jnp.float64,
        )
        if self.state_min.shape != (MPC_NX,) or self.state_max.shape != (MPC_NX,):
            raise ValueError(f"env.state_min/max_bounds must have shape ({MPC_NX},)")
        if bool(jnp.any(self.state_min >= self.state_max)):
            raise ValueError("env state bounds must satisfy min < max")

        self.default_q = jnp.asarray(actor.get("Q_run_init", DEFAULT_Q))
        self.default_q_terminal = jnp.asarray(actor.get("Q_term_init", self.default_q))
        self.default_r = jnp.asarray(actor.get("R_init", DEFAULT_R))
        self.rate_weights = jnp.asarray(actor.get("dR_init", DEFAULT_RATE))
        if (
            self.default_q.shape != (MPC_NX,)
            or self.default_q_terminal.shape != (MPC_NX,)
            or self.default_r.shape != (NU,)
            or self.rate_weights.shape != (NU,)
        ):
            raise ValueError(
                f"Q_run_init/Q_term_init must have length {MPC_NX} and "
                "R_init/dR_init length 3"
            )

        self.cost_map = CostMap(
            self.N,
            actor,
            default_q=self.default_q,
            default_r=self.default_r,
            default_q_terminal=self.default_q_terminal,
            nx=MPC_NX,
            nu=NU,
        )
        self.cost_dim = self.cost_map.output_dim
        self._initial_cost_logits = self.cost_map.initial_logits()
        self._ocp_base = self._base_params()

        def advance(seed, next_state, progress, done):
            states = _shift_nodes_batched(seed.states, progress, self.ds)
            controls = _shift_nodes_batched(seed.controls, progress, self.ds)
            states = states.at[:, 0].set(
                self.mpc_warm_start_state(states[:, 0], next_state)
            )
            finite = (
                jnp.all(jnp.isfinite(next_state), axis=-1)
                & jnp.isfinite(progress)
                & (progress >= 0.0)
            )
            return TurboSeed(states, controls, seed.valid & ~done & finite)

        self._advance = jax.jit(advance)

    def make_ocp(self):
        from turbompc.utils.load_params import load_problem_params

        params = dict(load_problem_params("racing.yaml"))
        params.update(self.ocp_base)
        params.update(
            horizon=self.N,
            discretization_resolution=self.ds,
            use_slack_variables=self.use_slack,
            slack_penalization_weight=self.slack_weight if self.use_slack else 0.0,
        )
        return RacingOptimalControlProblem(self.mpc_model, params)

    def advance(self, seed, next_state, progress, done):
        return self._advance(seed, next_state, progress, done)

    @property
    def ocp_base(self) -> Mapping[str, Any]:
        """Return the shared OCP parameter template."""
        return self._ocp_base

    def clip_control(self, controls: Any) -> Any:
        return jnp.clip(controls, self.control_min, self.control_max)

    @staticmethod
    def mpc_initial_state(context: Any) -> Any:
        """Replace path position with zero elapsed time for MPC."""
        context = jnp.asarray(context)
        return context[..., :PLANT_NX].at[..., PATH_EVOLUTION].set(0.0)

    def mpc_warm_start_state(self, predicted_state: Any, context: Any) -> Any:
        """Update the MPC initial state from the observation."""
        del predicted_state
        return self.mpc_initial_state(context)

    def path_knots(self, context: Any) -> Any:
        context = jnp.asarray(context)
        if context.shape[-1] not in {PLANT_NX, self.context_dim}:
            raise ValueError(
                "MPC context rows must have "
                f"{PLANT_NX} state values or {self.context_dim} context "
                "values"
            )
        return context[..., PATH_EVOLUTION, None] + self.ds * jnp.arange(self.N + 1)

    def _dynamics_parameters_at(self, knots: Any) -> dict[str, Any]:
        curvature = self.track.curvature(knots)
        if curvature.ndim:
            curvature = curvature[..., None]
        return {**self.mpc_state_dot_params, "curvature": curvature}

    def reference_trajectory_at(self, knots: Any) -> tuple[Any, Any]:
        states, controls = self.track.references_at(knots)
        return states, self.clip_control(controls)

    def _mpc_references_at(self, knots: Any) -> tuple[Any, Any]:
        return self.reference_trajectory_at(knots)

    def params_from_context(self, context: Any) -> tuple[dict[str, Any], Any]:
        context = jnp.asarray(context)
        knots = self.path_knots(context)
        seed_states, _ = self.reference_trajectory_at(knots)
        references, controls = self._mpc_references_at(knots)
        minimum, maximum = self.track.state_bounds(
            knots, self.state_min, self.state_max
        )
        params = {
            **self._ocp_base,
            "initial_state": self.mpc_initial_state(context),
            "initial_guess_final_state": seed_states[..., -1, :],
            "reference_state_trajectory": references,
            "reference_control_trajectory": controls,
            "state_min_bounds": minimum,
            "state_max_bounds": maximum,
            "dynamics_state_dot_params": self._dynamics_parameters_at(knots),
        }
        if self.constrain_initial_control:
            params["initial_control"] = self.clip_control(context[..., PLANT_NX:])
        return params, knots

    def reference_seed(self, context: Any) -> tuple[Any, Any]:
        context = jnp.asarray(context)
        knots = self.path_knots(context)
        states, controls = self.reference_trajectory_at(knots)
        states = states.at[..., 0, :].set(self.mpc_initial_state(context))
        return states, controls

    def build_mpc_inputs(self, costs: Any, context: Any) -> Any:
        params, _ = self.params_from_context(context)
        weights = self.cost_weights(costs, context)
        seed_states, seed_controls = self.reference_seed(context)
        return params, weights, seed_states, seed_controls

    def _base_params(self) -> dict[str, Any]:
        knots = self.initial_s + self.ds * jnp.arange(self.N + 1)
        seed_states, seed_controls = self.reference_trajectory_at(knots)
        states, controls = self._mpc_references_at(knots)
        minimum, maximum = self.track.state_bounds(
            knots, self.state_min, self.state_max
        )
        return {
            "horizon": self.N,
            "discretization_resolution": self.ds,
            "discretization_scheme": 10,
            "initial_state": seed_states[0],
            "initial_guess_final_state": seed_states[-1],
            "constrain_initial_control": self.constrain_initial_control,
            "initial_control": self.clip_control(seed_controls[0]),
            "reference_state_trajectory": states,
            "reference_control_trajectory": controls,
            "penalize_control_reference": True,
            Q_KEY: self.default_q,
            "weights_linear_penalization_state_trajectory": jnp.zeros((MPC_NX,)),
            "weights_penalization_final_state": jnp.zeros((MPC_NX,)),
            "weights_linear_penalization_final_state": self.terminal_linear_weight,
            R_KEY: self.default_r,
            "weights_linear_penalization_control_trajectory": jnp.zeros((NU,)),
            "weights_penalization_control_rate": self.rate_weights,
            "state_min_bounds": minimum,
            "state_max_bounds": maximum,
            "control_min_bounds": self.control_min,
            "control_max_bounds": self.control_max,
            "rescale_optimization_variables": True,
            "state_rescaling_min": -self.mpc_state_scale,
            "state_rescaling_max": self.mpc_state_scale,
            "control_rescaling_min": -MPC_CONTROL_SCALE,
            "control_rescaling_max": MPC_CONTROL_SCALE,
            "use_slack_variables": self.use_slack,
            "slack_state_bounds": self.use_slack,
            "slack_control_bounds": True,
            "slack_penalization_weight": self.slack_weight,
            "dynamics_state_dot_params": self._dynamics_parameters_at(knots),
        }

    def _cost_map_inputs(self, context: Any) -> dict[str, Any]:
        knots = self.path_knots(context)
        references, controls = self._mpc_references_at(knots)
        return {
            "reference_state_trajectory": references,
            "reference_control_trajectory": controls,
        }

    def cost_weights(
        self,
        costs: Any,
        context: Any | None = None,
    ) -> Mapping[str, Any]:
        costs = jnp.asarray(costs)
        if context is None:
            return self.cost_map(costs)
        return self.cost_map(costs, **self._cost_map_inputs(context))

    def initial_cost_logits(self) -> Any:
        return self._initial_cost_logits


__all__ = [
    "CONTROL_MAX",
    "CONTROL_MIN",
    "LATERAL_DEVIATION",
    "MPC_NX",
    "NU",
    "PATH_EVOLUTION",
    "PLANT_NX",
    "RacingProblem",
]
