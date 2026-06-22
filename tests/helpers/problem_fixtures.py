from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from turbompc.dynamics.drone_dynamics import (
    DroneDynamics,
    drone_parameters,
    drone_state_dot_parameters,
)
from turbompc.dynamics.linear_dynamics import (
    LinearDynamics,
    default_state_dot_parameters,
)
from turbompc.utils.load_params import load_problem_params


def tile_spacecraft_inertia(problem_params: dict, horizon: int | None = None) -> dict:
    dyn = problem_params.get("dynamics_state_dot_params")
    if not dyn or "inertia_vector" not in dyn:
        return problem_params
    h = int(problem_params["horizon"] if horizon is None else horizon)
    inertia = jnp.asarray(dyn["inertia_vector"])
    if inertia.ndim == 1:
        dyn = dict(dyn)
        dyn["inertia_vector"] = jnp.repeat(inertia[None, :], repeats=h + 1, axis=0)
        problem_params["dynamics_state_dot_params"] = dyn
    return problem_params


def cost_blocks_from_qr(
    Qmat: jnp.ndarray,
    Rmat: jnp.ndarray,
    Rd: jnp.ndarray,
    qvec: jnp.ndarray,
    rvec: jnp.ndarray,
):
    N = Qmat.shape[0] - 1
    nx = Qmat.shape[1]
    nu = Rmat.shape[1]
    n = nx + nu
    D = jnp.zeros((N + 1, n, n), dtype=Qmat.dtype)
    E = jnp.zeros((N, n, n), dtype=Qmat.dtype)
    for t in range(N + 1):
        D = D.at[t, :nx, :nx].set(Qmat[t])
        D = D.at[t, nx:, nx:].set(Rmat[t])
        if t > 0:
            D = D.at[t, nx:, nx:].add(Rd[t - 1])
        if t < N:
            D = D.at[t, nx:, nx:].add(Rd[t])
    for t in range(N):
        E = E.at[t, nx:, nx:].set(-Rd[t])
    q = jnp.concatenate([qvec, rvec], axis=-1)
    return D, E, q


def _symmetric_bounds(bounds):
    if isinstance(bounds, tuple):
        return bounds
    bounds = jnp.array(bounds)
    return -bounds, bounds


def _interpolate_reference_trajectory(
    initial_state: jnp.ndarray,
    target_state: jnp.ndarray,
    horizon: int,
    *,
    eps: float = 0.0,
):
    alphas = jnp.linspace(0.0, 1.0, horizon + 1, dtype=initial_state.dtype)[:, None]
    ref = (1.0 - alphas) * initial_state[None, :] + alphas * target_state[None, :]
    if eps != 0.0:
        ref = ref + jnp.asarray(eps, dtype=initial_state.dtype)
    return ref


def make_spacecraft_params(
    horizon: int,
    implicit: bool = False,
    rate_weight: float = 0.0,
    control_weight: float = 1.0,
    ref_weight: float = 1.0,
    final_weight: float = 0.0,
    initial_state: jnp.ndarray | None = None,
    initial_guess_final_state: jnp.ndarray | None = None,
    state_bounds=None,
    control_bounds=None,
):
    params = dict(load_problem_params("spacecraft_constrained.yaml"))
    params["horizon"] = horizon
    tile_spacecraft_inertia(params, horizon=horizon)
    if implicit:
        params["discretization_scheme"] = 10

    if initial_state is not None:
        params["initial_state"] = initial_state
    if initial_guess_final_state is not None:
        params["initial_guess_final_state"] = initial_guess_final_state

    params["reference_state_trajectory"] = jnp.zeros((horizon + 1, 3))
    params["reference_control_trajectory"] = jnp.zeros((horizon + 1, 3))
    params["weights_penalization_reference_state_trajectory"] = (
        jnp.ones((3,)) * ref_weight
    )
    params["weights_penalization_final_state"] = jnp.ones((3,)) * final_weight
    params["weights_penalization_control_squared"] = jnp.ones((3,)) * control_weight
    params["weights_penalization_control_rate"] = jnp.ones((3,)) * rate_weight

    if state_bounds is not None:
        state_min, state_max = _symmetric_bounds(state_bounds)
        params["state_min_bounds"] = state_min
        params["state_max_bounds"] = state_max
    if control_bounds is not None:
        control_min, control_max = _symmetric_bounds(control_bounds)
        params["control_min_bounds"] = control_min
        params["control_max_bounds"] = control_max
    return params


def make_drone_params(
    horizon: int,
    obs_centers: jnp.ndarray,
    obs_radii: jnp.ndarray,
):
    params = dict(load_problem_params("drone.yaml"))
    params["horizon"] = horizon

    params["lqr_feedback"] = True
    params["lqr_feedback_around_nominal_traj"] = True
    horizon = params["horizon"]
    params["reference_state_trajectory"] = _interpolate_reference_trajectory(
        jnp.asarray(params["initial_state"]),
        jnp.asarray(params["initial_guess_final_state"]),
        horizon,
        eps=1.0e-6,
    )

    params["obstacles_centers"] = obs_centers
    params["obstacles_radii"] = obs_radii
    params["obstacles_dimension"] = 2
    params["rescale_optimization_variables"] = False

    dynamics = DroneDynamics(drone_parameters)
    params["dynamics_state_dot_params"] = {
        key: jnp.array([value] * horizon)
        for key, value in drone_state_dot_parameters.items()
    }
    params["reference_control_trajectory"] = jnp.zeros(
        (horizon + 1, dynamics.num_controls)
    )

    return params, dynamics


def make_linear_params(
    horizon: int,
    implicit: bool,
    bounded: bool,
    rescale: bool = False,
    initial_control: bool = False,
    *,
    rescaling: str = "linspace",
    initial_state: jnp.ndarray | None = None,
    initial_guess_final_state: jnp.ndarray | None = None,
    reference_mode: str = "interpolate",
    affine_drift: jnp.ndarray | None = None,
):
    dynamics = LinearDynamics()
    nx = dynamics.num_states
    nu = dynamics.num_controls
    bound = 10.0 if bounded else 1.0e3
    if initial_state is None:
        initial_state = jnp.array([0.2, -0.1, 0.05, 5.0])
    if initial_guess_final_state is None:
        initial_guess_final_state = jnp.zeros((nx,))
    if affine_drift is None:
        affine_drift = jnp.zeros((nx,))
    if rescale and rescaling == "none":
        raise ValueError("rescaling='none' requires rescale=False")
    if rescaling not in {"linspace", "unit", "none"}:
        raise ValueError(f"Unknown rescaling={rescaling!r}")
    if reference_mode not in {"interpolate", "zero"}:
        raise ValueError(f"Unknown reference_mode={reference_mode!r}")
    if rescaling == "linspace":
        state_rescaling_min = -np.linspace(0.1, 5.0, nx)
        state_rescaling_max = jnp.ones((nx,))
        control_rescaling_min = -np.linspace(0.2, 3.0, nu)
        control_rescaling_max = jnp.ones((nu,))
    elif rescaling == "unit":
        state_rescaling_min = -jnp.ones((nx,))
        state_rescaling_max = jnp.ones((nx,))
        control_rescaling_min = -jnp.ones((nu,))
        control_rescaling_max = jnp.ones((nu,))
    else:
        # Always populate required keys even when rescaling is disabled.
        state_rescaling_min = -jnp.ones((nx,))
        state_rescaling_max = jnp.ones((nx,))
        control_rescaling_min = -jnp.ones((nu,))
        control_rescaling_max = jnp.ones((nu,))
    if reference_mode == "interpolate":
        reference_state_trajectory = _interpolate_reference_trajectory(
            jnp.asarray(initial_state),
            jnp.asarray(initial_guess_final_state),
            horizon,
        )
    else:
        reference_state_trajectory = jnp.zeros((horizon + 1, nx))
    params = {
        "horizon": horizon,
        "discretization_resolution": 0.1,
        "discretization_scheme": 10 if implicit else 0,
        "initial_state": initial_state,
        "initial_guess_final_state": initial_guess_final_state,
        "reference_state_trajectory": reference_state_trajectory,
        "reference_control_trajectory": jnp.zeros((horizon + 1, nu)),
        "penalize_control_reference": False,
        "rescale_optimization_variables": rescale,
        "constrain_initial_control": initial_control,
        "initial_control": jnp.zeros((nu,)),
        "state_rescaling_min": state_rescaling_min,
        "state_rescaling_max": state_rescaling_max,
        "control_rescaling_min": control_rescaling_min,
        "control_rescaling_max": control_rescaling_max,
        "weights_penalization_reference_state_trajectory": jnp.ones((nx,)),
        "weights_penalization_final_state": jnp.zeros((nx,)),
        "weights_penalization_control_squared": jnp.ones((nu,)),
        "weights_penalization_control_rate": jnp.zeros((nu,)),
        "state_min_bounds": -jnp.ones((nx,)) * bound,
        "state_max_bounds": jnp.ones((nx,)) * bound,
        "control_min_bounds": -jnp.ones((nu,)) * bound,
        "control_max_bounds": jnp.ones((nu,)) * bound,
        "dynamics_state_dot_params": {
            "A": jnp.repeat(
                default_state_dot_parameters["A"][None, :, :],
                repeats=horizon + 1,
                axis=0,
            ),
            "B": jnp.repeat(
                default_state_dot_parameters["B"][None, :, :],
                repeats=horizon + 1,
                axis=0,
            ),
            "b": jnp.repeat(
                jnp.asarray(affine_drift)[None, :], repeats=horizon + 1, axis=0
            ),
        },
    }
    return dynamics, params


def make_linear_active_constraint_params(
    horizon: int,
    *,
    implicit: bool = False,
    control_limit: float = 0.05,
    active_steps: int = 1,
    initial_state: jnp.ndarray | None = None,
):
    dynamics, params = make_linear_params(
        horizon=horizon,
        implicit=implicit,
        bounded=True,
        initial_state=initial_state,
    )
    active_steps = int(np.clip(active_steps, 1, horizon + 1))
    lower = jnp.full((horizon + 1, dynamics.num_controls), -1.0e3)
    upper = jnp.full((horizon + 1, dynamics.num_controls), 1.0e3)
    lower = lower.at[:active_steps].set(-control_limit)
    upper = upper.at[:active_steps].set(control_limit)
    params["control_min_bounds"] = lower
    params["control_max_bounds"] = upper
    params["weights_penalization_reference_state_trajectory"] = jnp.array(
        [10.0, 10.0, 1.0, 1.0], dtype=params["initial_state"].dtype
    )
    return dynamics, params
