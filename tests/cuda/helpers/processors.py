"""OCP and Solver processors for parameterized CUDA tests.

Design
------
Two frozen dataclass specs — OCPSpec, SolverSpec — cover EVERY parameter
that can vary in an OCP or solver configuration. Two processors —
OCPProcessor, SolverProcessor — build concrete instances from specs.
Invalid spec combinations raise ValueError with helpful error messages
that list valid alternatives.

Usage
-----
    @pytest.mark.parametrize("ocp_spec", OCPProcessor.parametrize(
        dynamics=["spacecraft", "linear"],
        horizons=[5, 10, 25],
    ))
    @pytest.mark.parametrize("solver_spec", SolverProcessor.parametrize(
        forward_backends=[ForwardBackend.ADMM_FUSED_PCG,
                          ForwardBackend.ADMM_FUSED_CUDSS],
    ))
    def test_something(ocp_spec, solver_spec):
        ocp, params = OCPProcessor.build(ocp_spec)
        solver = SolverProcessor.build(ocp, params, solver_spec)
        ...
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import product
from typing import Any, Optional

import jax.numpy as jnp
import pytest
from tests.helpers.backend_utils import backend_available
from tests.helpers.problem_fixtures import (
    make_drone_params,
    make_linear_params,
    make_spacecraft_params,
)
from tests.helpers.solver_fixtures import turbompc_solver_params
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.obstacle_avoidance import OptimalControlProblemObstacle
from turbompc.problems.optimal_control_problem import (
    OptimalControlProblem,
    OptimalControlProblemSlack,
)
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)

_VALID_OCP_VARIANTS = ("base", "obstacle", "slack", "obstacle_slack")
_VALID_DYNAMICS = ("linear", "spacecraft", "drone")
_VALID_DISCRETIZATION = ("euler", "midpoint", "rk4", "implicit")
_VALID_BOUNDS_MODE = ("both", "control_only", "state_only", "none")
_VALID_RESCALING_MODE = ("none", "unit", "linspace")

_DISCRETIZATION_CODE = {"euler": 0, "midpoint": 1, "rk4": 2, "implicit": 10}

# Default forward → backward mapping (mirrors _DEFAULT_BACKWARD in turbompc_solver.py)
_DEFAULT_BACKWARD: dict[ForwardBackend, BackwardBackend] = {
    ForwardBackend.ADMM_JAX_LOOP_PCG: BackwardBackend.ADMM_JAX_LOOP_PCG,
    ForwardBackend.ADMM_JAX_LOOP_PCG_FFI: BackwardBackend.ADMM_JAX_LOOP_PCG_FFI,
    ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI: BackwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
    ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE: BackwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
    ForwardBackend.ADMM_FUSED_PCG: BackwardBackend.DIRECT_CUDSS_FFI,
    ForwardBackend.ADMM_FUSED_CUDSS: BackwardBackend.DIRECT_CUDSS_FFI,
}


@dataclass(frozen=True)
class OCPSpec:
    # Problem structure
    ocp_variant: str = "base"  # "base", "obstacle", "slack", "obstacle_slack"
    dynamics: str = "spacecraft"  # "linear", "spacecraft", "drone"
    # Time / integration
    horizon: int = 10
    discretization: str = "euler"  # "euler", "midpoint", "rk4", "implicit"
    # Cost weights
    ref_weight: float = 1.0
    control_weight: float = 1.0
    rate_weight: float = 0.0
    final_weight: float = 0.0
    # Constraints
    bounds_mode: str = "both"  # "both", "control_only", "state_only", "none"
    state_bound: float = 10.0
    control_bound: float = 10.0
    constrain_initial_control: bool = False
    # Rescaling
    rescale: bool = False
    rescaling_mode: str = "none"  # "none", "unit", "linspace"
    # Slack
    slack_weight: float = 10.0
    # Obstacle (only valid for "obstacle" / "obstacle_slack")
    obstacle_centers: tuple = ()
    obstacle_radii: tuple = ()
    obstacle_dim: int = 2

    def __post_init__(self):
        _validate_ocp_spec(self)

    def pytest_id(self) -> str:
        parts = [self.dynamics, f"H{self.horizon}", self.discretization]
        if self.ocp_variant != "base":
            parts.append(self.ocp_variant)
        if self.bounds_mode != "both":
            parts.append(f"b-{self.bounds_mode}")
        if self.rescale:
            parts.append(f"rs-{self.rescaling_mode}")
        if self.constrain_initial_control:
            parts.append("ic")
        return "-".join(parts)


def _validate_ocp_spec(spec: OCPSpec) -> None:
    if spec.ocp_variant not in _VALID_OCP_VARIANTS:
        raise ValueError(
            f"Unknown ocp_variant={spec.ocp_variant!r}. "
            f"Valid: {list(_VALID_OCP_VARIANTS)}"
        )
    if spec.dynamics not in _VALID_DYNAMICS:
        raise ValueError(
            f"Unknown dynamics={spec.dynamics!r}. Valid: {list(_VALID_DYNAMICS)}"
        )
    if spec.discretization not in _VALID_DISCRETIZATION:
        raise ValueError(
            f"Unknown discretization={spec.discretization!r}. "
            f"Valid: {list(_VALID_DISCRETIZATION)}"
        )
    if spec.bounds_mode not in _VALID_BOUNDS_MODE:
        raise ValueError(
            f"Unknown bounds_mode={spec.bounds_mode!r}. "
            f"Valid: {list(_VALID_BOUNDS_MODE)}"
        )
    if spec.rescaling_mode not in _VALID_RESCALING_MODE:
        raise ValueError(
            f"Unknown rescaling_mode={spec.rescaling_mode!r}. "
            f"Valid: {list(_VALID_RESCALING_MODE)}"
        )
    if spec.horizon < 2:
        raise ValueError(f"horizon must be >= 2, got {spec.horizon}")
    if "obstacle" in spec.ocp_variant and spec.dynamics != "drone":
        raise ValueError(
            f"ocp_variant={spec.ocp_variant!r} requires dynamics='drone', "
            f"got dynamics={spec.dynamics!r}. "
            "Hint: set dynamics='drone' or use ocp_variant='base'/'slack'."
        )
    if "obstacle" in spec.ocp_variant and not spec.obstacle_centers:
        raise ValueError(
            f"ocp_variant={spec.ocp_variant!r} requires non-empty obstacle_centers. "
            "Hint: pass obstacle_centers=((-1.4, -0.1), (-0.7, 0.3)) and "
            "obstacle_radii=(0.3, 0.2)."
        )
    if spec.rescaling_mode != "none" and not spec.rescale:
        raise ValueError(
            f"rescaling_mode={spec.rescaling_mode!r} requires rescale=True. "
            "Hint: set rescale=True, or set rescaling_mode='none'."
        )
    if spec.rescale and spec.rescaling_mode == "none":
        raise ValueError(
            "rescale=True requires rescaling_mode != 'none'. "
            "Hint: set rescaling_mode='unit' or 'linspace'."
        )


@dataclass(frozen=True)
class SolverSpec:
    # Backends
    forward_backend: ForwardBackend = ForwardBackend.ADMM_JAX_LOOP_PCG
    backward_backend: Optional[BackwardBackend] = None  # auto-default via map
    use_full_hessian: bool = False
    # SQP outer loop
    num_sqp_iter: int = 10
    tol_convergence: float = 1e-4
    linesearch: bool = False
    # ADMM inner loop
    admm_sigma: float = 1e-6
    admm_max_iter: int = 500
    admm_eps_abs: float = 1e-4
    admm_eps_rel: float = 1e-4
    admm_rho_bar: float = 0.1
    admm_rho_min: float = 1e-6
    admm_rho_max: float = 1e6
    admm_rho_f_factor: float = 1000.0
    admm_alpha: float = 1.0
    admm_check_termination_every: int = 1
    admm_adapt_rho_every: int = 5
    admm_adaptive_rho_tolerance: float = 5.0
    # PCG
    pcg_max_iter: int = 200
    pcg_tol_epsilon: float = 1e-12

    def __post_init__(self):
        _validate_solver_spec(self)

    @property
    def effective_backward(self) -> BackwardBackend:
        return self.backward_backend or _DEFAULT_BACKWARD[self.forward_backend]

    def pytest_id(self) -> str:
        parts = [self.forward_backend.name]
        if (
            self.backward_backend is not None
            and self.backward_backend != _DEFAULT_BACKWARD[self.forward_backend]
        ):
            parts.append(f"bw-{self.backward_backend.name}")
        if self.use_full_hessian:
            parts.append("fh")
        if self.linesearch:
            parts.append("ls")
        return "-".join(parts)


def _validate_solver_spec(spec: SolverSpec) -> None:
    if not isinstance(spec.forward_backend, ForwardBackend):
        raise ValueError(
            "forward_backend must be ForwardBackend enum, got"
            f" {type(spec.forward_backend)}. Hint: import from"
            " turbompc.solvers.turbompc_solver."
        )
    if spec.backward_backend is not None and not isinstance(
        spec.backward_backend, BackwardBackend
    ):
        raise ValueError(
            "backward_backend must be BackwardBackend enum or None, got"
            f" {type(spec.backward_backend)}."
        )
    if spec.use_full_hessian:
        direct_bw = {BackwardBackend.DIRECT_JAX_DENSE, BackwardBackend.DIRECT_CUDSS_FFI}
        if spec.effective_backward not in direct_bw:
            raise ValueError(
                f"use_full_hessian=True requires backward_backend in {direct_bw}, "
                f"got effective backward={spec.effective_backward!r}. "
                "Hint: set backward_backend=BackwardBackend.DIRECT_CUDSS_FFI."
            )
    if spec.admm_sigma <= 0:
        raise ValueError(f"admm_sigma must be > 0, got {spec.admm_sigma}")
    if spec.admm_eps_abs <= 0 or spec.admm_eps_rel <= 0:
        raise ValueError(
            "admm_eps_abs and admm_eps_rel must be > 0, "
            f"got {spec.admm_eps_abs}, {spec.admm_eps_rel}"
        )
    if spec.admm_rho_min >= spec.admm_rho_max:
        raise ValueError(
            "admm_rho_min must be < admm_rho_max, "
            f"got {spec.admm_rho_min} >= {spec.admm_rho_max}"
        )
    if spec.admm_max_iter < 1:
        raise ValueError(f"admm_max_iter must be >= 1, got {spec.admm_max_iter}")
    if spec.num_sqp_iter < 1:
        raise ValueError(f"num_sqp_iter must be >= 1, got {spec.num_sqp_iter}")


class OCPProcessor:
    """Build OCP test instances from OCPSpec."""

    @staticmethod
    def build(spec: OCPSpec) -> tuple[Any, dict]:
        """Return (OptimalControlProblem, problem_params) ready for a solver."""
        state_bounds, control_bounds = _bounds_from_spec(spec)
        dynamics, params = _build_params(spec, state_bounds, control_bounds)
        params["discretization_scheme"] = _DISCRETIZATION_CODE[spec.discretization]
        params["constrain_initial_control"] = spec.constrain_initial_control
        if spec.constrain_initial_control:
            params.setdefault("initial_control", jnp.zeros((dynamics.num_controls,)))
        params["rescale_optimization_variables"] = spec.rescale

        if "slack" in spec.ocp_variant:
            params["use_slack_variables"] = True
            params["slack_penalization_weight"] = spec.slack_weight
            ocp_cls = OptimalControlProblemSlack
        else:
            params["use_slack_variables"] = False
            ocp_cls = OptimalControlProblem

        if "obstacle" in spec.ocp_variant:
            params["obstacles_centers"] = jnp.asarray(spec.obstacle_centers)
            params["obstacles_radii"] = jnp.asarray(spec.obstacle_radii)
            params["obstacles_dimension"] = spec.obstacle_dim
            ocp_cls = OptimalControlProblemObstacle  # overrides slack if both

        ocp = ocp_cls(dynamics=dynamics, params=params)
        return ocp, params

    @staticmethod
    def enumerate(**filters) -> list[OCPSpec]:
        """Cartesian product of filter lists (dict values). Scalars → single-element list.

        Example:
            enumerate(dynamics=["spacecraft", "linear"], horizon=[5, 10])
            → 4 OCPSpecs
        """
        return _cartesian_enumerate(OCPSpec, filters)

    @classmethod
    def parametrize(cls, **filters):
        """Return list[pytest.param(spec, id=...)] for @pytest.mark.parametrize."""
        return [
            pytest.param(spec, id=spec.pytest_id()) for spec in cls.enumerate(**filters)
        ]


class SolverProcessor:
    """Build TurboMPCSolver instances from SolverSpec."""

    @staticmethod
    def build(ocp, params, spec: SolverSpec) -> TurboMPCSolver:
        """Return a configured TurboMPCSolver.

        Raises FileNotFoundError if the requested backend's FFI library
        isn't built. Tests can call pytest.skip() on this.
        """
        if not backend_available(spec.forward_backend):
            raise OSError(f"{spec.forward_backend.name} not built/available")
        if not backend_available(spec.effective_backward):
            raise OSError(f"{spec.effective_backward.name} not built/available")

        sp = turbompc_solver_params(
            tol=spec.tol_convergence,
            sqp_iters=spec.num_sqp_iter,
            admm_max=spec.admm_max_iter,
        )
        sp["linesearch"] = spec.linesearch
        sp["admm"]["sigma"] = spec.admm_sigma
        sp["admm"]["eps_abs"] = spec.admm_eps_abs
        sp["admm"]["eps_rel"] = spec.admm_eps_rel
        sp["admm"]["rho"] = spec.admm_rho_bar
        sp["admm"]["rho_min"] = spec.admm_rho_min
        sp["admm"]["rho_max"] = spec.admm_rho_max
        sp["admm"]["rho_f_factor"] = spec.admm_rho_f_factor
        sp["admm"]["relaxation_parameter"] = spec.admm_alpha
        sp["admm"]["check_termination_every"] = spec.admm_check_termination_every
        sp["admm"]["adapt_rho_every"] = spec.admm_adapt_rho_every
        sp["admm"]["adaptive_rho_tolerance"] = spec.admm_adaptive_rho_tolerance
        sp["admm"]["pcg"]["max_iter"] = spec.pcg_max_iter
        sp["admm"]["pcg"]["tol_epsilon"] = spec.pcg_tol_epsilon

        return TurboMPCSolver(
            program=ocp,
            params=sp,
            forward_backend=spec.forward_backend,
            backward_backend=spec.effective_backward,
            use_full_hessian=spec.use_full_hessian,
        )

    @staticmethod
    def enumerate(**filters) -> list[SolverSpec]:
        return _cartesian_enumerate(SolverSpec, filters)

    @classmethod
    def parametrize(cls, **filters):
        """Return list[pytest.param(spec, id=...)].

        Note: backend availability is checked LAZILY inside the test
        (via build() raising FileNotFoundError/OSError) to avoid subprocess
        availability checks during test collection, which can spuriously
        fail when other GPU processes are running.
        """
        return [
            pytest.param(spec, id=spec.pytest_id()) for spec in cls.enumerate(**filters)
        ]


def _cartesian_enumerate(spec_cls, filters: dict) -> list:
    """Take a dict of lists, produce a list of spec instances via cartesian product.

    Filter keys map to spec field names. Values may be scalars (wrapped to 1-list)
    or iterables. Unspecified fields use the dataclass defaults.

    Aliases for user convenience: 'horizons' -> 'horizon',
    'forward_backends' -> 'forward_backend', etc.
    """
    # Accept plural aliases
    alias = {
        "dynamics_list": "dynamics",
        "horizons": "horizon",
        "discretizations": "discretization",
        "forward_backends": "forward_backend",
        "backward_backends": "backward_backend",
        "bounds_modes": "bounds_mode",
        "rescaling_modes": "rescaling_mode",
        "ocp_variants": "ocp_variant",
    }
    valid_fields = {f.name for f in fields(spec_cls)}
    normalized: dict[str, list] = {}
    for key, value in filters.items():
        key = alias.get(key, key)
        if key not in valid_fields:
            raise ValueError(
                f"Unknown filter {key!r} for {spec_cls.__name__}. "
                f"Valid fields: {sorted(valid_fields)}"
            )
        if isinstance(value, (list, tuple)) and not _is_tuple_field(spec_cls, key):
            normalized[key] = list(value)
        else:
            normalized[key] = [value]

    keys = list(normalized.keys())
    out = []
    for combo in product(*(normalized[k] for k in keys)):
        kwargs = dict(zip(keys, combo))
        out.append(spec_cls(**kwargs))
    return out


def _is_tuple_field(spec_cls, field_name: str) -> bool:
    """True if the field's declared type is tuple (so we should NOT expand)."""
    for f in fields(spec_cls):
        if f.name == field_name:
            # obstacle_centers, obstacle_radii, linesearch_alphas etc. are tuples
            return f.name in {"obstacle_centers", "obstacle_radii"}
    return False


def _bounds_from_spec(spec: OCPSpec) -> tuple[Optional[float], Optional[float]]:
    """Return (state_bound, control_bound) — either float or None (unbounded)."""
    if spec.bounds_mode == "both":
        return spec.state_bound, spec.control_bound
    if spec.bounds_mode == "control_only":
        return None, spec.control_bound
    if spec.bounds_mode == "state_only":
        return spec.state_bound, None
    # "none"
    return None, None


def _build_params(spec: OCPSpec, state_bound, control_bound):
    """Dispatch to existing fixture helpers and return (dynamics, params)."""
    if spec.dynamics == "spacecraft":
        # make_spacecraft_params expects rank-1 arrays, not scalars
        nx, nu = 3, 3
        state_bounds_arg = (
            jnp.ones((nx,)) * state_bound if state_bound is not None else None
        )
        control_bounds_arg = (
            jnp.ones((nu,)) * control_bound if control_bound is not None else None
        )
        params = make_spacecraft_params(
            horizon=spec.horizon,
            implicit=(spec.discretization == "implicit"),
            rate_weight=spec.rate_weight,
            control_weight=spec.control_weight,
            ref_weight=spec.ref_weight,
            final_weight=spec.final_weight,
            state_bounds=state_bounds_arg,
            control_bounds=control_bounds_arg,
        )
        return SpacecraftDynamics(), params

    if spec.dynamics == "linear":
        bounded = spec.bounds_mode in ("both", "state_only", "control_only")
        dynamics, params = make_linear_params(
            horizon=spec.horizon,
            implicit=(spec.discretization == "implicit"),
            bounded=bounded,
            rescale=spec.rescale,
            initial_control=spec.constrain_initial_control,
            rescaling=spec.rescaling_mode,
        )
        # Override cost weights
        nx = dynamics.num_states
        nu = dynamics.num_controls
        params["weights_penalization_reference_state_trajectory"] = (
            jnp.ones((nx,)) * spec.ref_weight
        )
        params["weights_penalization_control_squared"] = (
            jnp.ones((nu,)) * spec.control_weight
        )
        params["weights_penalization_control_rate"] = jnp.ones((nu,)) * spec.rate_weight
        params["weights_penalization_final_state"] = jnp.ones((nx,)) * spec.final_weight
        return dynamics, params

    if spec.dynamics == "drone":
        obs_centers = (
            jnp.asarray(spec.obstacle_centers)
            if spec.obstacle_centers
            else jnp.zeros((0, 2))
        )
        obs_radii = (
            jnp.asarray(spec.obstacle_radii) if spec.obstacle_radii else jnp.zeros((0,))
        )
        params, dynamics = make_drone_params(
            horizon=spec.horizon,
            obs_centers=obs_centers,
            obs_radii=obs_radii,
        )
        return dynamics, params

    raise ValueError(f"Unknown dynamics={spec.dynamics!r}")
