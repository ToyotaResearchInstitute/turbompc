"""Optimal control problem (OCP) definitions."""

import copy
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import vmap
from turbompc.dynamics.base_dynamics import Dynamics
from turbompc.dynamics.integrators import DiscretizationScheme, predict_next_state
from turbompc.utils.jax_utils import value_and_jacfwd
from turbompc.utils.load_params import (
    check_parameters_dictionary_or_raise_errors,
    normalize_problem_params,
)


def _stage_weight(w, horizon):
    """Coerce a cost-weight field to (horizon, *) for per-stage broadcast.

    Accepts:
      - 1-D (D,): constant across stages, repeated to (horizon, D).
      - 2-D (>= horizon, D): time-varying, first `horizon` rows used
        for the inner stages; the (horizon+1)-th row (if present) is
        consumed by the terminal cost via `_terminal_weight`.
    """
    if w.ndim == 1:
        return jnp.repeat(w[None], repeats=horizon, axis=0)
    return w[:horizon]


def _terminal_weight(w):
    """Pick the terminal-stage weight for terminal_cost.

    1-D inputs pass through (constant across stages including terminal).
    2-D inputs return the last row, matching how `reference_*_trajectory[-1]`
    is consumed for the final stage.
    """
    return w if w.ndim == 1 else w[-1]


class BaseOptimalControlProblem:
    """Base optimal control problem."""

    def initial_guess(
        self, params: Dict[str, Any] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns initial guess trajectories.

        Args:
            params: dictionary of parameters of the optimal control problem,
                (key=string, value=Any)

        Returns:
            states: (_horizon + 1, _num_state_variables) array
            controls: (_horizon + 1, _num_control_variables) array
        """
        raise NotImplementedError

    def stage_cost(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        next_state: jnp.ndarray,
        next_control: jnp.ndarray,
        params: Dict[str, Any],
    ) -> float:
        """Returns stage cost to minimize.

        Args:
            state: (_num_state_variables, ) array
            control: (_num_control_variables, ) array
            next_state: (_num_state_variables, ) array
            next_control: (_num_control_variables, ) array
            params: dictionary of one-stage cost parameters,
                (key=string, value=Any)

        Returns:
            cost: value of the stage cost,
                (float)
        """
        raise NotImplementedError

    def terminal_cost(
        self, state: jnp.ndarray, control: jnp.ndarray, params: Dict[str, Any]
    ) -> float:
        """Returns terminal cost.

        Args:
            state: (_num_state_variables, ) array
            control: (_num_control_variables, ) array
            params: dictionary of terminal cost parameters,
                (key=string, value=Any)

        Returns:
            cost: value of the terminal cost,
                (float)
        """
        raise NotImplementedError

    def step_inequality_constraints(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Returns inequality constraints g_l <= g(x, u) <= g_u at one timestep.

        Args:
            state: (_num_state_variables, ) array
            control: (_num_control_variables, ) array
            params: dictionary of one-timestep parameters,
                (key=string, value=Any).

        Returns:
            g(x): value of g(x) (_num_inequality_constraints, ) array
            g_l: value of g_l (_num_inequality_constraints, ) array
            g_u: value of g_u (_num_inequality_constraints, ) array
        """
        raise NotImplementedError


class OptimalControlProblem(BaseOptimalControlProblem):
    """
    Quadratic tracking optimal control problem.

    Trajectories have shape:
        states:   (N + 1, nx)
        controls: (N + 1, nu)

    Dynamics link stages t and t + 1 for t = 0..N-1. 

    Override `stage_cost`, `terminal_cost`, or `step_inequality_constraints`
    for customization.

    Optional box bounds are provided through params:
        state_min_bounds <= x_t <= state_max_bounds
        control_min_bounds <= u_t <= control_max_bounds
    for t = 0..N.

    Core params:
        - `horizon`: scalar N.
        - `reference_state_trajectory`: (N + 1, nx).
        - `reference_control_trajectory`: (N + 1, nu).
        - `weights_penalization_control_rate`: (nu,), (nu, nu), (N, nu),
          or (N, nu, nu).
        - `dynamics_state_dot_params`: optional pytree.
    """

    def __init__(
        self,
        dynamics: Dynamics,
        params: Dict[str, Any] = None,
        check_parameters_are_valid: bool = True,
    ):
        """Initializes the class."""
        self._dynamics = dynamics
        self._name = "OptimalControlProblem"
        if params is not None:
            params = copy.deepcopy(params)
        normalize_problem_params(params)
        if check_parameters_are_valid:
            check_parameters_dictionary_or_raise_errors(params)
        self._params = params
        self._num_variables = (
            self.num_control_variables + self.num_state_variables
        ) * (self.horizon + 1)
        self._active_state_bounds = None

        self._rescale_optimization_variables = params.get(
            "rescale_optimization_variables", False
        )
        self._constrain_initial_control = params.get("constrain_initial_control", False)
        self._penalize_control_reference = params.get(
            "penalize_control_reference", False
        )
        self._use_slack_variables = params.get("use_slack_variables", False)
        self._discretization_scheme = DiscretizationScheme(
            params["discretization_scheme"]
        )
        self._active_control_bounds = None
        self._set_active_bounds(params)

        g = self.inequality_constraints(
            states=jnp.zeros((self.horizon + 1, self.num_state_variables)),
            controls=jnp.zeros((self.horizon + 1, self.num_control_variables)),
            params=params,
        )[0].reshape((self.horizon + 1, -1))
        self._num_inequality_constraints = g.shape[1]

    def _set_active_bounds(self, params: Dict[str, Any]) -> None:
        """Compute masks for active box bounds (finite entries)."""
        x_min, x_max, u_min, u_max = self.get_box_bounds(params)
        self._active_state_bounds = jnp.any(
            jnp.logical_or(x_min > -1e6, x_max < 1e6), axis=0
        )
        self._active_control_bounds = jnp.any(
            jnp.logical_or(u_min > -1e6, u_max < 1e6), axis=0
        )

    @property
    def dynamics(self) -> Dynamics:
        """Returns the dynamics of the class."""
        return self._dynamics

    @property
    def name(self) -> str:
        """Returns the program name (for solver compatibility)."""
        return self._name

    @property
    def params(self) -> Dict:
        """Returns a dictionary of parameters of the program."""
        return self._params

    @property
    def horizon(self) -> int:
        """Returns the problem horizon."""
        return int(self.params["horizon"])

    @property
    def discretization_scheme(self) -> DiscretizationScheme:
        """Returns the discretization scheme."""
        return self._discretization_scheme

    @property
    def rescale_optimization_variables(self) -> bool:
        """Whether state/control variables are internally rescaled."""
        return self._rescale_optimization_variables

    @property
    def constrain_initial_control(self) -> bool:
        """Whether the problem includes the optional equality u0 = initial_control."""
        return self._constrain_initial_control

    @property
    def penalize_control_reference(self) -> bool:
        """Whether control tracking uses reference_control_trajectory."""
        return self._penalize_control_reference

    @property
    def use_slack_variables(self) -> bool:
        """Whether slack variable are used for inequalities."""
        return self._use_slack_variables

    @property
    def num_variables(self) -> int:
        """Returns the number of optimization variables."""
        return self._num_variables

    @property
    def num_inequality_constraints(self) -> int:
        """Returns the number of inequality constraints."""
        num = self._num_inequality_constraints
        return num

    @property
    def num_state_variables(self) -> int:
        """Returns the number of state variables."""
        return self.dynamics.num_states

    @property
    def num_control_variables(self) -> int:
        """Returns the number of control variables."""
        return self.dynamics.num_controls

    def _get_default_bounds(self):
        """
        Return unconstrained bounds for states and controls.

        Returns:
            x_min, x_max: (N+1, nx)
            u_min, u_max: (N+1, nu)
        """
        N = self.horizon
        nx = self.num_state_variables
        nu = self.num_control_variables
        x_min = -jnp.inf * jnp.ones((N + 1, nx))
        x_max = jnp.inf * jnp.ones((N + 1, nx))
        u_min = -jnp.inf * jnp.ones((N + 1, nu))
        u_max = jnp.inf * jnp.ones((N + 1, nu))
        return x_min, x_max, u_min, u_max

    def _get_rescaling_params(self, params: Dict[str, Any]):
        """
        Return rescaling bounds and scale factors.

        Scaling convention:
            x_scaled = x_true / state_diff
            u_scaled = u_true / control_diff
        where state_diff = (state_max - state_min) / 2 (same for control).
        """
        # Use instance variable instead of params.get() for JIT compatibility
        if not self._rescale_optimization_variables:
            state_diff = jnp.ones((self.num_state_variables,))
            control_diff = jnp.ones((self.num_control_variables,))
            return None, None, None, None, state_diff, control_diff

        state_min = params["state_rescaling_min"]
        state_max = params["state_rescaling_max"]
        control_min = params["control_rescaling_min"]
        control_max = params["control_rescaling_max"]
        if state_min.shape != (self.num_state_variables,):
            raise ValueError(
                "state_rescaling_min must have shape "
                f"({self.num_state_variables},), got {state_min.shape}"
            )
        if state_max.shape != (self.num_state_variables,):
            raise ValueError(
                "state_rescaling_max must have shape "
                f"({self.num_state_variables},), got {state_max.shape}"
            )
        if control_min.shape != (self.num_control_variables,):
            raise ValueError(
                "control_rescaling_min must have shape "
                f"({self.num_control_variables},), got {control_min.shape}"
            )
        if control_max.shape != (self.num_control_variables,):
            raise ValueError(
                "control_rescaling_max must have shape "
                f"({self.num_control_variables},), got {control_max.shape}"
            )
        state_diff = (state_max - state_min) / 2.0
        control_diff = (control_max - control_min) / 2.0
        return state_min, state_max, control_min, control_max, state_diff, control_diff

    def scale_states_controls(
        self, states: jnp.ndarray, controls: jnp.ndarray, params: Dict[str, Any]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Scale states and controls when rescaling is enabled.
        """
        if self._rescale_optimization_variables:
            _, _, _, _, state_diff, control_diff = self._get_rescaling_params(params)
            return states / state_diff, controls / control_diff
        return states, controls

    def unscale_states_controls(
        self, states: jnp.ndarray, controls: jnp.ndarray, params: Dict[str, Any]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Unscale states and controls when rescaling is enabled.

        Inverse of scale_states_controls.
        """
        if self._rescale_optimization_variables:
            _, _, _, _, state_diff, control_diff = self._get_rescaling_params(params)
            return states * state_diff, controls * control_diff
        return states, controls

    def get_box_bounds(self, params=None):
        """
        Return box bounds on states and controls.

        Bounds are optional. If not provided in params, they default
        to unconstrained.

        Accepted shapes:
            - state_min_bounds:   (nx,) or (N+1, nx)
            - state_max_bounds:   (nx,) or (N+1, nx)
            - control_min_bounds: (nu,) or (N+1, nu)
            - control_max_bounds: (nu,) or (N+1, nu)

        Returns:
            x_lower, x_upper, u_lower, u_upper
        """
        if params is None:
            params = self.params

        x_min, x_max, u_min, u_max = self._get_default_bounds()
        if "state_min_bounds" in params:
            x_min = params["state_min_bounds"]
        if "state_max_bounds" in params:
            x_max = params["state_max_bounds"]
        if "control_min_bounds" in params:
            u_min = params["control_min_bounds"]
        if "control_max_bounds" in params:
            u_max = params["control_max_bounds"]

        N = self.horizon
        nx = self.num_state_variables
        nu = self.num_control_variables

        def _reshape_bounds(name, arr, dim):
            """Reshape bounds to shape (N+1, dim), potentially duplicating over 1st axis."""
            if arr.ndim == 1:
                if arr.shape != (dim,):
                    raise ValueError(
                        f"{name} has shape {arr.shape}, expected ({dim},) or ({N+1},"
                        f" {dim})"
                    )
                return jnp.repeat(arr[None, :], repeats=(N + 1), axis=0)

            if arr.ndim == 2:
                if arr.shape != (N + 1, dim):
                    raise ValueError(
                        f"{name} has shape {arr.shape}, expected ({dim},) or ({N+1},"
                        f" {dim})"
                    )
                return arr

            raise ValueError(
                f"{name} must be rank-1 or rank-2 array, got ndim={arr.ndim}"
            )

        x_min = _reshape_bounds("state_min_bounds", x_min, nx)
        x_max = _reshape_bounds("state_max_bounds", x_max, nx)
        u_min = _reshape_bounds("control_min_bounds", u_min, nu)
        u_max = _reshape_bounds("control_max_bounds", u_max, nu)

        return x_min, x_max, u_min, u_max

    def step_inequality_constraints(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Return per-timestep inequality constraints g(x,u), scaled if enabled.

        Args:
            state: (nx,) state at time t.
            control: (nu,) control at time t.
            params: problem parameter.

        Returns:
            g: (m,) constraint values.
            l: (m,) lower bounds.
            u: (m,) upper bounds.
        """
        if (
            "state_min_bounds" not in params
            or "state_max_bounds" not in params
            or "control_min_bounds" not in params
            or "control_max_bounds" not in params
        ):
            return jnp.zeros((0,)), jnp.zeros((0,)), jnp.zeros((0,))

        x_min = jnp.asarray(params["state_min_bounds"])
        x_max = jnp.asarray(params["state_max_bounds"])
        u_min = jnp.asarray(params["control_min_bounds"])
        u_max = jnp.asarray(params["control_max_bounds"])

        state_mask = self._active_state_bounds
        control_mask = self._active_control_bounds
        if state_mask is not None:
            state = state[state_mask]
            x_min = x_min[state_mask]
            x_max = x_max[state_mask]
        if control_mask is not None:
            control = control[control_mask]
            u_min = u_min[control_mask]
            u_max = u_max[control_mask]

        g = jnp.concatenate([state, control], axis=0)
        l = jnp.concatenate([x_min, u_min], axis=0)
        u = jnp.concatenate([x_max, u_max], axis=0)

        if self._rescale_optimization_variables:
            _, _, _, _, state_diff, control_diff = self._get_rescaling_params(params)
            if state_mask is not None:
                state_diff = state_diff[state_mask]
            if control_mask is not None:
                control_diff = control_diff[control_mask]
            scale_vec = jnp.concatenate([state_diff, control_diff], axis=0)
            g = g / scale_vec
            l = l / scale_vec
            u = u / scale_vec

        return g, l, u

    def _get_param_in_axes(
        self,
        params: Dict[str, Any],
        time_length: int,
        unsliced_keys=None,
        time_varying_private_keys=None,
    ) -> Dict[str, Any]:
        """Return pytree-shaped vmap in_axes for time-indexed params."""
        unsliced_keys = set() if unsliced_keys is None else set(unsliced_keys)
        time_varying_private_keys = (
            set()
            if time_varying_private_keys is None
            else set(time_varying_private_keys)
        )

        def _is_unsliced_key(key):
            if key in time_varying_private_keys:
                return False
            return str(key).startswith("_") or key in unsliced_keys

        def _leaf_in_axes(value):
            if hasattr(value, "ndim") and hasattr(value, "shape"):
                if value.ndim > 0 and value.shape[0] == time_length:
                    return 0
            return None

        def _tree_in_axes(value):
            if isinstance(value, dict):
                return {
                    key: None if _is_unsliced_key(key) else _tree_in_axes(val)
                    for key, val in value.items()
                }
            if isinstance(value, tuple):
                return tuple(_tree_in_axes(val) for val in value)
            if isinstance(value, list):
                return [_tree_in_axes(val) for val in value]
            return _leaf_in_axes(value)

        return _tree_in_axes(params)

    def _prepare_pointwise_inequality_params(
        self, params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return params and vmap in_axes for one-step inequality constraints.

        JAX supports pytree-shaped in_axes. Example: if horizon N=4 and
        params contains
            {"obstacles_centers": array(5, n_obs, dim), "radii": array(n_obs)}
        then the returned in_axes contains
            {"obstacles_centers": 0, "radii": None}
        so vmap slices obstacle centers by timestep and reuses radii unchanged.
        """
        x_min, x_max, u_min, u_max = self.get_box_bounds(params)
        pointwise_params = dict(params)
        pointwise_params["state_min_bounds"] = x_min
        pointwise_params["state_max_bounds"] = x_max
        pointwise_params["control_min_bounds"] = u_min
        pointwise_params["control_max_bounds"] = u_max

        params_in_axes = self._get_param_in_axes(
            pointwise_params,
            self.horizon + 1,
            unsliced_keys={
                "state_rescaling_min",
                "state_rescaling_max",
                "control_rescaling_min",
                "control_rescaling_max",
            },
        )
        return pointwise_params, params_in_axes

    def inequality_constraints(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Returns inequality constraints g_l <= g(x) <= g_u.

        Args:
            states: (_horizon + 1, _num_state_variables) array
            controls: (_horizon + 1, _num_control_variables) array
            params: dictionary of parameters of the optimal control problem,
                (key=string, value=Any)

        Returns:
            g_value: value of g(x),
                (_horizon + 1, _num_inequality_constraints) array
            g_l: value of g_l,
                (_horizon + 1, _num_inequality_constraints) array
            g_u: value of g_u,
                (_horizon + 1, _num_inequality_constraints) array
        """
        pointwise_params, params_in_axes = self._prepare_pointwise_inequality_params(
            params
        )

        def _step_vals(state, control, params_t):
            return self.step_inequality_constraints(state, control, params_t)

        g_all, l_all, u_all = vmap(_step_vals, in_axes=(0, 0, params_in_axes))(
            states, controls, pointwise_params
        )
        if g_all.size == 0:
            return jnp.zeros((0,)), jnp.zeros((0,)), jnp.zeros((0,))
        return g_all, l_all, u_all

    def get_inequalities_linearized_matrices(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Linearize nonlinear inequality constraints and return solver QP blocks.

        This combines linearizations of step_inequality_constraints around
        (states, controls), including box bounds when enabled.

        Returns:
            A_ineq_blocks: (N+1, m, nx+nu) local blocks per timestep.
                If no constraints are active, returns shape (0, 0, nx+nu).
            l_ineq: (N+1, m)
            u_ineq: (N+1, m)
        """
        pointwise_params, params_in_axes = self._prepare_pointwise_inequality_params(
            params
        )

        def _linearize_step(state_t, control_t, params_t):
            g_t, l_t, u_t = self.step_inequality_constraints(
                state_t, control_t, params_t
            )
            state_control = jnp.concatenate([state_t, control_t], axis=0)

            def _g_only(sc):
                s = sc[: self.num_state_variables]
                c = sc[self.num_state_variables :]
                g_val, _, _ = self.step_inequality_constraints(s, c, params_t)
                return g_val

            g_val, jac = value_and_jacfwd(_g_only, state_control)
            Jx = jac[:, : self.num_state_variables]
            Ju = jac[:, self.num_state_variables :]
            offset = g_val - (Jx @ state_t + Ju @ control_t)
            l_lin = l_t - offset
            u_lin = u_t - offset
            return Jx, Ju, l_lin, u_lin

        Jx_all, Ju_all, l_all, u_all = vmap(
            _linearize_step, in_axes=(0, 0, params_in_axes)
        )(states, controls, pointwise_params)
        A_nl_blocks = jnp.concatenate([Jx_all, Ju_all], axis=-1)
        return A_nl_blocks, l_all, u_all

    def _get_dynamics_params_sequence(
        self, params: Dict[str, Any], num_steps: int
    ) -> Dict[str, jnp.ndarray]:
        """
        Build per-step dynamics params for dynamics.state_dot.

        If "dynamics_state_dot_params" is missing, returns {}.
        Scalars are broadcast to (num_steps,).
        1D arrays whose length matches num_steps or num_steps+1 are ambiguous and raise a value error.
        2D+ arrays with leading dimension num_steps/num_steps+1 are sliced to num_steps.
        Other arrays are broadcast across time.

        Warning: A constant matrix of size (dim1, dim2) with dim1=num_steps is assumed
        to be a time-varying vector, causing a (potentially silent) error downstream.
        In this case, pass it as a tensor of size (num_steps, dim1, dim2).
        """
        dyn_params = params.get("dynamics_state_dot_params")
        if dyn_params is None:
            return {}
        out: Dict[str, jnp.ndarray] = {}
        for key, value in dyn_params.items():
            if isinstance(value, jnp.ndarray):
                if value.ndim == 0:
                    out[key] = jnp.broadcast_to(value, (num_steps,))
                elif value.ndim == 1:
                    if value.shape[0] in (num_steps, num_steps + 1):
                        # length matches time axis => potential bug (can't distinguish between
                        # a vector of length horizon and scalar changing over the horizon)
                        raise ValueError(
                            f"Ambiguous 1D dynamics param '{key}': length"
                            f" {value.shape[0]} matches horizon ({num_steps}). Use 2D"
                            " shape (num_steps, dim) for time-varying vectors to avoid"
                            " ambiguities / potential bugs."
                        )
                    else:
                        # Treat 1D vectors as constants; time-varying vectors should be 2D.
                        out[key] = jnp.broadcast_to(value, (num_steps,) + value.shape)
                elif value.shape[0] == num_steps + 1:
                    out[key] = value[:num_steps]
                elif value.shape[0] == num_steps:
                    out[key] = value
                else:
                    out[key] = jnp.broadcast_to(value, (num_steps,) + value.shape)
            else:
                value_arr = jnp.asarray(value)
                if value_arr.ndim == 0:
                    out[key] = jnp.broadcast_to(value_arr, (num_steps,))
                else:
                    out[key] = jnp.broadcast_to(
                        value_arr, (num_steps,) + value_arr.shape
                    )
        # collapse scalar time-series encoded as (N, 1) into (N,)
        for key, value in out.items():
            if (
                isinstance(value, jnp.ndarray)
                and value.ndim == 2
                and value.shape[1] == 1
            ):
                out[key] = value[:, 0]
        return out

    def _get_control_rate_weights(self, params: Dict[str, Any]) -> jnp.ndarray:
        """
        Return control-rate penalty weights Rd as (N, nu, nu).

        Accepted shapes:
            - (nu,)
            - (nu, nu)
            - (N, nu)
            - (N, nu, nu)
        """
        N = self.horizon
        nu = self.num_control_variables
        rd = jnp.array(params["weights_penalization_control_rate"])

        if rd.ndim == 1 and rd.shape == (nu,):
            rd = jnp.diag(rd)
            rd = jnp.repeat(rd[None], repeats=N, axis=0)
        elif rd.ndim == 2 and rd.shape == (nu, nu):
            rd = jnp.repeat(rd[None], repeats=N, axis=0)
        elif rd.ndim == 2 and rd.shape == (N, nu):
            rd = vmap(jnp.diag)(rd)
        elif rd.ndim == 3 and rd.shape == (N, nu, nu):
            rd = rd
        else:
            raise ValueError(
                "weights_penalization_control_rate shape must be (nu,), (nu, nu), (N,"
                " nu), or (N, nu, nu)."
            )

        return rd

    def _cost_unsliced_keys(self):
        """Return keys passed unchanged to cost hooks.

        Some values, such as dynamics_state_dot_params, may contain
        time-varying leaves. They are not sliced here.
        """
        return {
            "horizon",
            "discretization_resolution",
            "discretization_scheme",
            "initial_state",
            "initial_guess_final_state",
            "initial_control",
            "state_rescaling_min",
            "state_rescaling_max",
            "control_rescaling_min",
            "control_rescaling_max",
            "state_min_bounds",
            "state_max_bounds",
            "control_min_bounds",
            "control_max_bounds",
            "dynamics_state_dot_params",
            "weights_penalization_control_rate",
        }

    def _get_cost_params(
        self, params: Dict[str, Any], dtype: jnp.dtype
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Return stage params, stage in_axes, and terminal params."""
        unsliced_keys = self._cost_unsliced_keys()

        def _is_constant_key(key):
            return str(key).startswith("_") or key in unsliced_keys

        def _slice(value, *, terminal: bool):
            if isinstance(value, dict):
                return {
                    key: (
                        val if _is_constant_key(key) else _slice(val, terminal=terminal)
                    )
                    for key, val in value.items()
                }
            if isinstance(value, tuple):
                return tuple(_slice(val, terminal=terminal) for val in value)
            if isinstance(value, list):
                return [_slice(val, terminal=terminal) for val in value]
            if hasattr(value, "ndim") and hasattr(value, "shape"):
                if value.ndim > 0 and value.shape[0] == self.horizon + 1:
                    return value[-1] if terminal else value[: self.horizon]
            return value

        stage_params = _slice(dict(params), terminal=False)
        stage_params["reference_state_trajectory"] = params[
            "reference_state_trajectory"
        ][:-1]
        stage_params["reference_control_trajectory"] = params[
            "reference_control_trajectory"
        ][:-1]
        stage_params["weights_penalization_reference_state_trajectory"] = _stage_weight(
            params["weights_penalization_reference_state_trajectory"], self.horizon
        )
        stage_params["weights_penalization_control_squared"] = _stage_weight(
            params["weights_penalization_control_squared"], self.horizon
        )
        stage_params["_control_rate_weight_matrix"] = self._get_control_rate_weights(
            params
        )

        stage_in_axes = self._get_param_in_axes(
            stage_params,
            self.horizon,
            unsliced_keys=unsliced_keys,
            time_varying_private_keys={"_control_rate_weight_matrix"},
        )

        terminal_params = _slice(dict(params), terminal=True)
        terminal_params["reference_state_trajectory"] = params[
            "reference_state_trajectory"
        ][-1]
        terminal_params["reference_control_trajectory"] = params[
            "reference_control_trajectory"
        ][-1]
        terminal_params["weights_penalization_reference_state_trajectory"] = (
            _terminal_weight(params["weights_penalization_reference_state_trajectory"])
        )
        terminal_params["weights_penalization_control_squared"] = _terminal_weight(
            params["weights_penalization_control_squared"]
        )
        terminal_params["weights_penalization_final_state"] = _terminal_weight(
            params["weights_penalization_final_state"]
        )
        terminal_params["weights_linear_penalization_final_state"] = _terminal_weight(
            params.get(
                "weights_linear_penalization_final_state",
                jnp.zeros((self.num_state_variables,), dtype=dtype),
            )
        )
        return stage_params, stage_in_axes, terminal_params

    def initial_guess(
        self, params: Dict[str, Any] = None
    ) -> Tuple[jnp.array, jnp.array]:
        """Returns an initial guess for the solution."""
        if params is None:
            params = self.params
        x_initial = params["initial_state"]
        x_final = params["initial_guess_final_state"]
        horizon = self.horizon

        # straight-line initial guess
        state_matrix = jnp.zeros((horizon + 1, self.num_state_variables))
        for t in range(horizon + 1):
            alpha1 = (horizon - t) / horizon
            alpha2 = t / horizon
            state_matrix = state_matrix.at[t].set(
                x_initial * alpha1 + x_final * alpha2 + 1e-6
            )
        # zero initial guess
        control_matrix = jnp.zeros((horizon + 1, self.num_control_variables)) + 1e-6
        return state_matrix, control_matrix

    def equality_constraints(
        self, states: jnp.array, controls: jnp.array, params: Dict[str, Any]
    ) -> jnp.array:
        """Returns equality constraints h(x) = 0.

        Args:
            states: (N + 1, nx) array
            controls: (N + 1, nu) array
            params: dictionary of parameters of the optimal control problem,
                (key=string, value=Any)

        Returns:
            h_value: value of h(x),
                (_num_equality_constraints, ) array
        """
        horizon = self.horizon
        # initial state is fixed
        initial_state_constraints = states[0] - params["initial_state"]

        # dynamics constraints
        dt = params["discretization_resolution"]
        if self.discretization_scheme == DiscretizationScheme.IMPLICIT:
            dynamics_params = self._get_dynamics_params_sequence(params, horizon + 1)
            states_dot = vmap(self.dynamics.state_dot)(
                states, controls, dynamics_params
            )
            next_states = states[:horizon] + 0.5 * dt * (
                states_dot[:-1] + states_dot[1:]
            )
            dynamics_constraints = states[1:] - next_states
        else:
            dynamics_params = self._get_dynamics_params_sequence(params, horizon)
            next_states = vmap(
                predict_next_state,
                in_axes=(None, None, None, 0, 0, 0, 0),
            )(
                self.dynamics,
                dt,
                self.discretization_scheme,
                dynamics_params,
                states[:horizon],
                controls[:horizon],
                controls[1:],
            )
            dynamics_constraints = next_states - states[1:]

        # all equality constraints
        constraints = jnp.concatenate(
            [initial_state_constraints[jnp.newaxis], dynamics_constraints], axis=0
        )

        if self._rescale_optimization_variables:
            _, _, _, _, state_diff, control_diff = self._get_rescaling_params(params)
            constraints = constraints / state_diff

        constraints = constraints.flatten()

        # initial control constraints
        u0_constraint = None
        if self._constrain_initial_control:
            init_control = params["initial_control"]
            if init_control.shape != (self.num_control_variables,):
                raise ValueError(
                    "initial_control has shape "
                    f"{init_control.shape}, expected ({self.num_control_variables},)"
                )
            u0_constraint = controls[0] - init_control
            if self._rescale_optimization_variables:
                u0_constraint = u0_constraint / control_diff
            constraints = jnp.concatenate([constraints, u0_constraint])

        return constraints

    def get_initial_equality_linearized_matrices(
        self, params: Dict[str, Any], dtype: jnp.dtype
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns terms for the stage-0 equality constraints
            A0 @ z0 = c0
        where z0 = [x0, u0].

        These rows encode the fixed initial-state constraint
            x0 = initial_state
        and, if enabled, the fixed initial-control constraint
            u0 = initial_control.

        Args:
            params: dictionary of parameters of the optimal control problem,
                (key=string, value=Any)
            dtype: dtype to use for the returned arrays

        Returns:
            (in scaled units if rescaling is enabled)

            A0: stage-0 equality matrix
                (n0, nx + nu) array
            c0: stage-0 equality right-hand side
                (n0,) array
        """
        nx = self.num_state_variables
        nu = self.num_control_variables
        initial_state = jnp.asarray(params["initial_state"], dtype=dtype)
        state_diff = control_diff = None
        if self._rescale_optimization_variables:
            _, _, _, _, state_diff, control_diff = self._get_rescaling_params(params)

        A0_rows = [
            jnp.concatenate(
                [jnp.eye(nx, dtype=dtype), jnp.zeros((nx, nu), dtype=dtype)], axis=1
            )
        ]
        c0_rows = [initial_state]

        if state_diff is not None:
            A0_rows[0] = A0_rows[0] * (1.0 / state_diff)[:, jnp.newaxis]
            c0_rows[0] = c0_rows[0] / state_diff

        if self._constrain_initial_control:
            if "initial_control" not in params:
                raise ValueError(
                    "initial_control must be provided when constrain_initial_control "
                    "is True."
                )
            initial_control = jnp.asarray(params["initial_control"], dtype=dtype)
            if initial_control.shape != (nu,):
                raise ValueError(
                    "initial_control has shape "
                    f"{initial_control.shape}, expected ({nu},)"
                )
            control_block = jnp.concatenate(
                [jnp.zeros((nu, nx), dtype=dtype), jnp.eye(nu, dtype=dtype)], axis=1
            )
            control_rhs = initial_control
            if control_diff is not None:
                control_block = control_block * (1.0 / control_diff)[:, jnp.newaxis]
                control_rhs = control_rhs / control_diff
            A0_rows.append(control_block)
            c0_rows.append(control_rhs)

        return jnp.concatenate(A0_rows, axis=0), jnp.concatenate(c0_rows, axis=0)

    def stage_cost(
        self,
        state: jnp.array,
        control: jnp.array,
        next_state: jnp.array,
        next_control: jnp.array,
        params: Dict[str, Any],
    ) -> float:
        """Returns pairwise stage cost."""
        reference_state = params["reference_state_trajectory"]
        reference_control = params["reference_control_trajectory"]
        weights_x_ref = params["weights_penalization_reference_state_trajectory"]
        weights_u_norm = params["weights_penalization_control_squared"]

        if self._penalize_control_reference:
            reference = jnp.concatenate([reference_state, reference_control], axis=-1)
        else:
            reference = jnp.concatenate([reference_state, jnp.zeros_like(control)])
        weights_ref = jnp.concatenate([weights_x_ref, weights_u_norm], axis=-1)

        state_control = jnp.concatenate([state, control], axis=-1)
        total_cost = weights_ref * (state_control - reference) ** 2
        total_cost = jnp.sum(total_cost)

        du = next_control - control
        rd = params["_control_rate_weight_matrix"]
        total_cost = total_cost + 0.5 * (du @ (rd @ du))
        return total_cost

    def terminal_cost(
        self, state: jnp.array, control: jnp.array, params: Dict[str, Any]
    ) -> float:
        """Returns terminal cost."""
        weights_x_ref = params["weights_penalization_reference_state_trajectory"]
        weights_x_final = params["weights_penalization_final_state"]
        weights_x_final_linear = params.get(
            "weights_linear_penalization_final_state", jnp.zeros_like(state)
        )
        weights_ref = weights_x_ref + weights_x_final
        total_cost = weights_ref * (state - params["reference_state_trajectory"]) ** 2
        total_cost = jnp.sum(total_cost)
        total_cost = total_cost + jnp.sum(weights_x_final_linear * state)

        weights_u_norm = params["weights_penalization_control_squared"]
        reference_control = params["reference_control_trajectory"]
        if self._penalize_control_reference:
            control_ref = reference_control
        else:
            control_ref = jnp.zeros_like(control)
        total_cost = total_cost + jnp.sum(weights_u_norm * (control - control_ref) ** 2)
        return total_cost

    def cost(
        self, states: jnp.array, controls: jnp.array, params: Dict[str, Any]
    ) -> float:
        """Returns total cost to minimize.

        Args:
            states: (N + 1, nx) array
            controls: (N + 1, nu) array
            params: dictionary of parameters of the optimal control problem,
                (key=string, value=Any)

        Returns:
            cost: value of the cost,
                (float)
        """
        stage_params, stage_in_axes, terminal_params = self._get_cost_params(
            params, states.dtype
        )

        stage_costs = vmap(
            self.stage_cost,
            in_axes=(0, 0, 0, 0, stage_in_axes),
        )(
            states[:-1],
            controls[:-1],
            states[1:],
            controls[1:],
            stage_params,
        )
        total_cost = jnp.sum(stage_costs) + self.terminal_cost(
            states[-1], controls[-1], terminal_params
        )
        return total_cost

    def get_cost_linearized_blocks(
        self,
        states: jnp.array,
        controls: jnp.array,
        params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Return the block-tridiagonal QP cost terms in true (unscaled) units.

        The returned blocks represent
            0.5 * z.T @ P @ z + q.T @ z
        where z stacks per-stage variables z_t = [x_t, u_t]. `D[t]` is the
        diagonal Hessian block for z_t, and `E[t]` is the lower off-diagonal
        block coupling z_{t+1} to z_t. The full Hessian therefore has
        P[t, t] = D[t], P[t + 1, t] = E[t], and P[t, t + 1] = E[t].T.

        TurboMPC and SQP-OSQP use the full (D, E, q).
        SQPDiffMPCSolver requires E == 0.

        Args:
            states: state trajectory,
                (N + 1, nx) array
            controls: control trajectory,
                (N + 1, nu) array
            params: dictionary of parameters of the optimal control problem,
                (key=string, value=Any)

        Returns:
            D: diagonal Hessian blocks,
                (N + 1, nx + nu, nx + nu) array
            E: lower off-diagonal Hessian blocks,
                (N, nx + nu, nx + nu) array
            q: linear cost blocks,
                (N + 1, nx + nu) array
        """
        N = self.horizon
        nx = self.num_state_variables
        nu = self.num_control_variables
        n = nx + nu

        stage_params, stage_in_axes, terminal_params = self._get_cost_params(
            params, states.dtype
        )

        x_blocks = jnp.concatenate([states, controls], axis=-1)
        x_pairs = jnp.concatenate([x_blocks[:-1], x_blocks[1:]], axis=-1)

        def stage_cost_from_pair(x_pair, stage_params_t):
            x_t = x_pair[:n]
            x_tp1 = x_pair[n:]
            state_t = x_t[:nx]
            control_t = x_t[nx:]
            state_tp1 = x_tp1[:nx]
            control_tp1 = x_tp1[nx:]
            return self.stage_cost(
                state_t,
                control_t,
                state_tp1,
                control_tp1,
                stage_params_t,
            )

        def terminal_cost_from_block(x_pair):
            state_t = x_pair[:nx]
            control_t = x_pair[nx:]
            return self.terminal_cost(state_t, control_t, terminal_params)

        stage_grad = jax.grad(stage_cost_from_pair)
        stage_hess = jax.hessian(stage_cost_from_pair)
        grads = vmap(stage_grad, in_axes=(0, stage_in_axes))(x_pairs, stage_params)
        hessians = vmap(stage_hess, in_axes=(0, stage_in_axes))(x_pairs, stage_params)

        H_minus = hessians[:, :n, :n]
        H_plus = hessians[:, n:, n:]
        H_pm = hessians[:, :n, n:]
        g_minus = grads[:, :n]
        g_plus = grads[:, n:]

        x_t = x_blocks[:-1]
        x_tp1 = x_blocks[1:]
        q_minus = g_minus - vmap(lambda Hm, Hpm, xt, xtp1: Hm @ xt + Hpm @ xtp1)(
            H_minus, H_pm, x_t, x_tp1
        )
        q_plus = g_plus - vmap(lambda Hp, Hpm, xt, xtp1: Hpm.T @ xt + Hp @ xtp1)(
            H_plus, H_pm, x_t, x_tp1
        )

        term_grad = jax.grad(terminal_cost_from_block)(x_blocks[-1])
        term_hess = jax.hessian(terminal_cost_from_block)(x_blocks[-1])
        qN_minus = term_grad - term_hess @ x_blocks[-1]

        D = jnp.zeros((N + 1, n, n), dtype=states.dtype)
        E = jnp.zeros((N, n, n), dtype=states.dtype)
        q = jnp.zeros((N + 1, n), dtype=states.dtype)

        D = D.at[0].set(H_minus[0])
        if N > 1:
            D = D.at[1:N].set(H_plus[:-1] + H_minus[1:])
        D = D.at[N].set(H_plus[-1] + term_hess)
        E = H_pm

        q = q.at[0].set(q_minus[0])
        if N > 1:
            q = q.at[1:N].set(q_plus[:-1] + q_minus[1:])
        q = q.at[N].set(q_plus[-1] + qN_minus)

        return D, E, q

    def get_dynamics_linearized_matrices(
        self, states: jnp.array, controls: jnp.array, params: Dict[str, Any]
    ) -> Tuple[jnp.ndarray]:
        """
        Returns terms for the initial state and dynamics equality constraints
        states[0] = Cs[0]
        As_next[t]@states[t+1] + As[t]@states[t] + Bs_next[t]@controls[t+1]
            + Bs[t]@controls[t] = Fs[t]
        where Cs = [Cs[0], Fs] and t = 0, ..., N-1
        corresponding to the linearization of dynamics constraints
            f_t(x_{t+1}, x_t, u_{t+1}, u_t) = 0
        around a (states, controls) trajectory.

        Dimensions: (N, nx, nu) = (horizon, num_states, num_controls)

        Args:
            states: state trajectory,
                (N + 1, nx) array
            controls: control trajectory,
                (N + 1, nu) array
            params: dictionary of parameters of the optimal control problem,
                (key=string, value=Any)

        Returns:
            (in scaled units if rescaling is enabled)

            As_next: dynamics matrices multiplying next states
                (N, nx, nx) array
            As: dynamics matrices multiplying states
                (N, nx, nx) array
            Bs_next: dynamics matrices multiplying next controls
                (N, nx, nu) array
            Bs: dynamics matrices multiplying controls
                (N, nx, nu) array
            Cs: initial state and dynamics vectors, Cs = (x0, Fs)
                (N+1, nx) array (initial state, dynamics) constraints

        """

        # Time-varying dynamics parameters should be shaped (horizon, ...)
        # for explicit schemes and (horizon+1, ...) for implicit schemes.
        def linearize_explicit_integrator(
            states: jnp.ndarray, controls: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            # x+ = f(x,u) ~= f(y,v) + ∇f(y,v) (x-y, u-v)
            # => -I x+ + ∇f(y,v)(x, u) = -(f(y,v) - ∇f(y,v) (y, v))
            dynamics_params = self._get_dynamics_params_sequence(params, self.horizon)

            def next_state(state_control, step_params):
                state = state_control[: self.num_state_variables]
                control = state_control[self.num_state_variables :]
                return predict_next_state(
                    self.dynamics,
                    params["discretization_resolution"],
                    self.discretization_scheme,
                    step_params,
                    state,
                    control,
                    control,
                )

            def next_state_and_gradient_dstate_dcontrol(state, control, step_params):
                state_control = jnp.concatenate([state, control])

                def next_state_fn(sc):
                    return next_state(sc, step_params)

                next_state_val, next_state_grad = value_and_jacfwd(
                    next_state_fn,
                    state_control,
                )
                return next_state_val, next_state_grad

            next_states, next_states_dstate_dcontrol = vmap(
                next_state_and_gradient_dstate_dcontrol
            )(
                states[: self.horizon],
                controls[: self.horizon],
                dynamics_params,
            )
            As = next_states_dstate_dcontrol[:, :, : self.num_state_variables]
            Bs = next_states_dstate_dcontrol[:, :, self.num_state_variables :]
            Cs = jnp.concatenate(
                [
                    params["initial_state"][jnp.newaxis],
                    -next_states
                    + vmap(lambda A, x: A @ x)(As, states[: self.horizon])
                    + vmap(lambda A, x: A @ x)(Bs, controls[: self.horizon]),
                ],
                axis=0,
            )
            As_next = jnp.repeat(
                -jnp.eye(self.num_state_variables)[jnp.newaxis],
                repeats=self.horizon,
                axis=0,
            )
            return As_next, As, Bs, Cs

        def linearize_implicit_integrator(
            states: jnp.ndarray, controls: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            nx = self.num_state_variables
            dt = params["discretization_resolution"]
            dynamics_params = self._get_dynamics_params_sequence(
                params, self.horizon + 1
            )

            def state_base_dot(state, control, step_params):
                return 0.5 * dt * self.dynamics.state_dot(state, control, step_params)

            def state_base_dot_and_gradient(state, control, step_params):
                state_control = jnp.concatenate([state, control])
                return value_and_jacfwd(
                    lambda sc: state_base_dot(sc[:nx], sc[nx:], step_params),
                    state_control,
                )

            states_base_dots, gradients = vmap(
                state_base_dot_and_gradient, in_axes=(0, 0, 0)
            )(states, controls, dynamics_params)
            dxddx = gradients[:, :, :nx]
            dxddu = gradients[:, :, nx:]
            I = jnp.eye(nx)
            As = I + dxddx[:-1]
            As_next = -I + dxddx[1:]
            Bs = dxddu[:-1]
            Bs_next = dxddu[1:]

            Fs = (
                vmap(lambda A, x: A @ x)(As, states[:-1])
                + vmap(lambda A, x: A @ x)(As_next, states[1:])
                + vmap(lambda B, u: B @ u)(Bs, controls[:-1])
                + vmap(lambda B, u: B @ u)(Bs_next, controls[1:])
                - states[:-1]
                - states_base_dots[:-1]
                - states_base_dots[1:]
                + states[1:]
            )
            Cs = jnp.concatenate([params["initial_state"][jnp.newaxis], Fs], axis=0)
            return As_next, Bs_next, As, Bs, Cs

        if self.discretization_scheme == DiscretizationScheme.IMPLICIT:
            As_next, Bs_next, As, Bs, Cs = linearize_implicit_integrator(
                states, controls
            )
        else:
            As_next, As, Bs, Cs = linearize_explicit_integrator(states, controls)
            Bs_next = jnp.zeros_like(Bs)
        if self._rescale_optimization_variables:
            _, _, _, _, state_diff, _ = self._get_rescaling_params(params)
            row_scale = 1.0 / state_diff
            As_next = As_next * row_scale[jnp.newaxis, :, jnp.newaxis]
            Bs_next = Bs_next * row_scale[jnp.newaxis, :, jnp.newaxis]
            As = As * row_scale[jnp.newaxis, :, jnp.newaxis]
            Bs = Bs * row_scale[jnp.newaxis, :, jnp.newaxis]
            Cs = Cs * row_scale[jnp.newaxis, :]
        return As_next, Bs_next, As, Bs, Cs

    def get_dynamics_lagrangian_hessian(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        params: Dict[str, Any],
        lambdas: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute dynamics Lagrangian Hessian contribution: ∇²[λᵀ c_t].

        For explicit integrators (Euler/Midpoint/RK4):
            c_t = x_t + dt*f(x_t, u_t) - x_{t+1}
            Only D_t is modified: D_t += dt * λ_{t+1}ᵀ ∇²f(x_t, u_t)

        For implicit (trapezoidal):
            c_t = x_t + 0.5*dt*(f(x_t,u_t) + f(x_{t+1},u_{t+1})) - x_{t+1}
            Both D_t and D_{t+1} are modified:
              D_t   += 0.5*dt * λ_{t+1}ᵀ ∇²f(x_t, u_t)           (from constraint t)
              D_{t+1} += 0.5*dt * λ_{t+1}ᵀ ∇²f(x_{t+1}, u_{t+1}) (from constraint t)

        Returns:
            (N+1, n, n) array of Hessian blocks to add to D.
        """
        N = self.horizon
        nx = self.num_state_variables
        nu = self.num_control_variables
        n = nx + nu
        dt = params["discretization_resolution"]

        if self.discretization_scheme == DiscretizationScheme.IMPLICIT:
            dt_scale = 0.5 * dt
        else:
            dt_scale = dt

        if self.discretization_scheme == DiscretizationScheme.IMPLICIT:
            dynamics_params_all = self._get_dynamics_params_sequence(params, N + 1)
        else:
            dynamics_params_all = self._get_dynamics_params_sequence(params, N)

        if self._rescale_optimization_variables:
            _, _, _, _, state_diff, _ = self._get_rescaling_params(params)
            row_scale = 1.0 / state_diff

            def state_dot(x, u, p):
                return row_scale * self.dynamics.state_dot(x, u, p)

        else:
            state_dot = self.dynamics.state_dot
            row_scale = 1.0

        def single_hessian(x, u, lam, step_params):
            """Hessian of lambda.T @ state_dot(x, u) for one state_dot call."""

            def scalar_fn(xu):
                return lam @ state_dot(xu[:nx], xu[nx:], step_params)

            xu = jnp.concatenate([x, u])
            return jax.hessian(scalar_fn)(xu)

        if self.discretization_scheme == DiscretizationScheme.IMPLICIT:
            # Implicit trapezoidal: constraint t involves f(x_t,u_t) AND f(x_{t+1},u_{t+1}).
            # H_t from f(x_t, u_t) side: contributes to D_t
            dynamics_params_t = jax.tree.map(lambda x: x[:N], dynamics_params_all)
            H_current = jax.vmap(single_hessian)(
                states[:N], controls[:N], lambdas, dynamics_params_t
            )  # (N, n, n)

            # H_{t+1} from f(x_{t+1}, u_{t+1}) side: contributes to D_{t+1}
            dynamics_params_tp1 = jax.tree.map(
                lambda x: x[1 : N + 1], dynamics_params_all
            )
            H_next = jax.vmap(single_hessian)(
                states[1 : N + 1], controls[1 : N + 1], lambdas, dynamics_params_tp1
            )  # (N, n, n)

            # Accumulate into (N+1, n, n) D blocks
            hess_blocks = jnp.zeros((N + 1, n, n), dtype=states.dtype)
            hess_blocks = hess_blocks.at[:N].add(
                dt_scale * H_current
            )  # D_t += from f(x_t,u_t)
            hess_blocks = hess_blocks.at[1 : N + 1].add(
                dt_scale * H_next
            )  # D_{t+1} += from f(x_{t+1},u_{t+1})
        else:
            # Explicit integrators: constraint t only involves predict_next_state(x_t, u_t), contributing to D_t.
            def single_hessian_discrete(x, u, lam, step_params):
                """∇²_{(x,u)} [λᵀ · predict_next_state(x, u)] for the configured integrator."""

                def scalar_fn(xu):
                    x_next = predict_next_state(
                        self.dynamics,
                        dt,
                        self.discretization_scheme,
                        step_params,
                        xu[:nx],
                        xu[nx:],
                        xu[nx:],
                    )
                    return lam @ (row_scale * x_next)

                return jax.hessian(scalar_fn)(jnp.concatenate([x, u]))

            H_blocks = jax.vmap(single_hessian_discrete)(
                states[:N], controls[:N], lambdas, dynamics_params_all
            )  # (N, n, n)

            terminal_block = jnp.zeros((1, n, n), dtype=states.dtype)
            hess_blocks = jnp.concatenate([H_blocks, terminal_block], axis=0)

        return hess_blocks


class SlackProblemAdapter:
    """Adapter that augments a problem to support slack variables.

    Expects params to include:
      - use_slack_variables: True
      - slack_penalization_weight: scalar penalty (gamma)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        params = self.params
        if "use_slack_variables" not in params:
            raise KeyError("use_slack_variables should be in params.")
        if "slack_penalization_weight" not in params:
            raise KeyError(
                "slack_penalization_weight should be in slack_penalization_weight."
            )
        self._use_slack_variables = bool(params["use_slack_variables"])
        if not self._use_slack_variables:
            raise ValueError("SlackProblemAdapter requires use_slack_variables=True.")

    def cost_with_slack(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slack: jnp.ndarray,
        params: Dict[str, Any],
    ) -> jnp.ndarray:
        """Return cost with 0.5 * gamma * ||slack||^2 added."""
        base_cost = self.cost(states, controls, params)
        return base_cost + 0.5 * params["slack_penalization_weight"] * jnp.sum(
            slack**2
        )

    def inequality_constraints_with_slack(
        self,
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slack: jnp.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return inequality constraints evaluated at g(x,u)+slack."""
        g, l, u = self.inequality_constraints(states, controls, params)
        return g + slack, l, u

    def optimal_slack(
        self,
        g: jnp.ndarray,
        l: jnp.ndarray,
        u: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return slack variables projecting g into [l,u]: proj(g) - g."""
        proj = jnp.minimum(jnp.maximum(g, l), u)
        return proj - g


def make_slack_problem(base_cls):
    class SlackProblem(SlackProblemAdapter, base_cls):
        pass

    SlackProblem.__name__ = f"{base_cls.__name__}Slack"
    return SlackProblem


OptimalControlProblemSlack = make_slack_problem(OptimalControlProblem)


__all__ = [
    "BaseOptimalControlProblem",
    "OptimalControlProblem",
    "OptimalControlProblemSlack",
    "SlackProblemAdapter",
    "make_slack_problem",
]
