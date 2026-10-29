"""Map actor outputs to bounded running and terminal cost weights."""

import jax
import jax.numpy as jnp

Q_KEY = "weights_penalization_reference_state_trajectory"
R_KEY = "weights_penalization_control_squared"


class CostMap:
    def __init__(
        self, horizon, config, *, default_q, default_r, default_q_terminal, nx, nu
    ):
        if config["quadratic_cost_mode"] != "linear":
            raise ValueError("racing uses quadratic_cost_mode=linear")
        self.horizon, self.nx, self.nu = horizon, nx, nu
        self.segments = int(config["running_cost_segments"])
        if not 1 <= self.segments <= horizon:
            raise ValueError("running_cost_segments must lie in [1, horizon]")
        self.learned_terminal_cost = bool(config["learned_terminal_cost"])
        self.terminal_multiplier = float(config["terminal_weight_multiplier"])
        if self.terminal_multiplier <= 0:
            raise ValueError("terminal_weight_multiplier must be positive")
        self.q_lower, self.q_upper = self._bounds(config, "Q", nx)
        self.r_lower, self.r_upper = self._bounds(config, "R", nu)
        self.default_q, self.default_r = default_q, default_r
        self.default_q_terminal = default_q_terminal
        base, extra = divmod(horizon, self.segments)
        self.node_segment = jnp.asarray(
            [s for s in range(self.segments) for _ in range(base + (s < extra))],
            dtype=jnp.int32,
        )
        # One [Q, R] block per running segment, followed by terminal Q.
        self.running_width = nx + nu
        self.output_dim = self.segments * self.running_width + nx

    @staticmethod
    def _bounds(config, name, width):
        lo, hi = (
            jnp.asarray(config[name + suffix], dtype=jnp.float64)
            for suffix in ("_lb", "_ub")
        )
        if lo.shape != (width,) or hi.shape != (width,):
            raise ValueError(f"{name} bounds must have width {width}")
        if not bool(
            jnp.all(jnp.isfinite(lo) & jnp.isfinite(hi) & (lo >= 0) & (hi >= lo))
        ):
            raise ValueError(f"invalid {name} bounds")
        return lo, hi

    @staticmethod
    def _initial(value, lo, hi):
        span = hi - lo
        gain = jnp.where(span > 0, (value - lo) / jnp.where(span > 0, span, 1.0), 0.5)
        if not bool(jnp.all(jnp.isfinite(gain) & (gain > 0) & (gain < 1))):
            raise ValueError(
                "initial learned costs must lie strictly inside their bounds"
            )
        return jnp.log(gain) - jnp.log1p(-gain)

    def initial_logits(self):
        q = self._initial(self.default_q, self.q_lower, self.q_upper)
        r = self._initial(self.default_r, self.r_lower, self.r_upper)
        qn = (
            self._initial(
                self.default_q_terminal / self.terminal_multiplier,
                self.q_lower,
                self.q_upper,
            )
            if self.learned_terminal_cost
            else jnp.zeros(self.nx)
        )
        running = jnp.concatenate([q, r])
        return jnp.concatenate([jnp.tile(running, self.segments), qn])

    def __call__(
        self,
        raw,
        *,
        reference_state_trajectory=None,
        reference_control_trajectory=None,
    ):
        if raw.shape[-1] != self.output_dim:
            raise ValueError(f"cost actor must output {self.output_dim} coordinates")
        running = raw[..., : self.segments * self.running_width].reshape(
            raw.shape[:-1] + (self.segments, self.running_width)
        )
        terminal = raw[..., self.segments * self.running_width :]
        q = self.q_lower + jax.nn.sigmoid(running[..., : self.nx]) * (
            self.q_upper - self.q_lower
        )
        r = self.r_lower + jax.nn.sigmoid(running[..., self.nx :]) * (
            self.r_upper - self.r_lower
        )
        qn = self.terminal_multiplier * (
            self.q_lower + jax.nn.sigmoid(terminal) * (self.q_upper - self.q_lower)
        )
        if not self.learned_terminal_cost:
            qn = jnp.zeros_like(qn)
        q = jnp.concatenate([q[..., self.node_segment, :], qn[..., None, :]], axis=-2)
        r = jnp.concatenate([r[..., self.node_segment, :], r[..., -1:, :]], axis=-2)
        return {
            Q_KEY: q,
            R_KEY: r,
            "reference_state_trajectory": (
                jnp.zeros((self.horizon + 1, self.nx))
                if reference_state_trajectory is None
                else reference_state_trajectory
            ),
            "reference_control_trajectory": (
                jnp.zeros((self.horizon + 1, self.nu))
                if reference_control_trajectory is None
                else reference_control_trajectory
            ),
        }
