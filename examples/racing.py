"""Simplified minimum-time closed-loop racing example."""

# ruff: noqa: E402

from pathlib import Path
from time import perf_counter

from jax import config

config.update("jax_enable_x64", True)

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from turbompc.dynamics.vehicle import PathVehicleDynamics, TimeVehicleDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)
from turbompc.utils.load_params import load_problem_params, load_solver_params

MPC_PERIOD, SIM_STEPS = 0.1, 300
SIMULATOR_SUBSTEPS = 5
INITIAL_PATH_EVOLUTION = 37.4
VERBOSE = True
PROGRESS_PRINT_INTERVAL = 50

trajectory_csv = np.genfromtxt(
    Path(__file__).with_name("racing_trajectory.csv"), delimiter=",", names=True
)
TRAJECTORY = {
    name: jnp.asarray(trajectory_csv[name])
    for name in (
        "s",
        "k",
        "V",
        "r",
        "beta",
        "rear_wr",
        "delta_cmd",
        "engine_torque_cmd",
        "rear_brake_torque_cmd",
        "front_brake_torque_cmd",
        "E",
        "N",
        "phi",
        "e_min",
        "e_max",
    )
}


def interpolate_trajectory(path_evolution, name):
    clipped_path_evolution = jnp.clip(
        path_evolution,
        TRAJECTORY["s"][0],
        TRAJECTORY["s"][-1],
    )
    return jnp.interp(
        clipped_path_evolution,
        TRAJECTORY["s"],
        TRAJECTORY[name],
    )


def references_at(path_evolution_knots):
    zero_tracking_error = jnp.zeros_like(path_evolution_knots)
    zero_time_reference = jnp.zeros_like(path_evolution_knots)
    states = jnp.stack(
        [
            interpolate_trajectory(path_evolution_knots, "r"),
            interpolate_trajectory(path_evolution_knots, "V"),
            interpolate_trajectory(path_evolution_knots, "beta"),
            interpolate_trajectory(path_evolution_knots, "rear_wr"),
            zero_tracking_error,  # lateral deviation from the racing line
            zero_tracking_error,  # heading error from the racing line
            zero_time_reference,  # time evolution is minimized, not tracked
        ],
        axis=-1,
    )
    controls = jnp.stack(
        [
            interpolate_trajectory(path_evolution_knots, "delta_cmd"),
            interpolate_trajectory(path_evolution_knots, "engine_torque_cmd")
            + interpolate_trajectory(path_evolution_knots, "rear_brake_torque_cmd"),
            interpolate_trajectory(path_evolution_knots, "front_brake_torque_cmd"),
        ],
        axis=-1,
    )
    return states, controls


def state_bounds(path_evolution, minimum, maximum):
    minimum = jnp.broadcast_to(minimum, (path_evolution.size, minimum.size))
    maximum = jnp.broadcast_to(maximum, (path_evolution.size, maximum.size))
    minimum = minimum.at[:, 4].set(interpolate_trajectory(path_evolution, "e_min"))
    maximum = maximum.at[:, 4].set(interpolate_trajectory(path_evolution, "e_max"))
    return minimum, maximum


simulator_model = TimeVehicleDynamics()
mpc_model = PathVehicleDynamics()
problem_params = load_problem_params("racing.yaml")
HORIZON = int(problem_params["horizon"])
PATH_STEP = float(problem_params["discretization_resolution"])
VEHICLE_PARAMETERS = problem_params["dynamics_state_dot_params"]


def dynamics_parameters_at(path_evolution):
    curvature = interpolate_trajectory(path_evolution, "k")
    if curvature.ndim:
        curvature = curvature[:, None]
    return {
        **VEHICLE_PARAMETERS,
        "curvature": curvature,
    }


initial_path_evolution_knots = INITIAL_PATH_EVOLUTION + PATH_STEP * jnp.arange(
    HORIZON + 1
)
initial_reference, initial_controls = references_at(initial_path_evolution_knots)
initial_simulator_state = initial_reference[0].at[-1].set(INITIAL_PATH_EVOLUTION)
initial_minimum, initial_maximum = state_bounds(
    initial_path_evolution_knots,
    problem_params["state_min_bounds"],
    problem_params["state_max_bounds"],
)
problem_params.update(
    initial_state=initial_reference[0],
    initial_guess_final_state=initial_reference[-1],
    initial_control=initial_controls[0],
    reference_state_trajectory=initial_reference,
    reference_control_trajectory=initial_controls,
    state_min_bounds=initial_minimum,
    state_max_bounds=initial_maximum,
    dynamics_state_dot_params=dynamics_parameters_at(initial_path_evolution_knots),
)

solver_params = load_solver_params("turbompc.yaml")
solver_params["num_sqp_iteration_max"] = 1
solver_params["linesearch"]["method"] = "merit"
solver_params["tol_convergence"] = 1e-3
solver_params["admm"].update(
    max_iter=10_000, eps_abs=1e-3, eps_rel=1e-3, check_termination_every=5
)
problem = OptimalControlProblem(mpc_model, problem_params)
solver = TurboMPCSolver(
    problem,
    solver_params,
    forward_backend=ForwardBackend.ADMM_JAX_LOOP_EIGEN,
    backward_backend=BackwardBackend.ADMM_JAX_LOOP_EIGEN,
)


def mpc_parameters(simulator_state, previous_control):
    path_evolution_knots = simulator_state[-1] + PATH_STEP * jnp.arange(HORIZON + 1)
    state_reference, control_reference = references_at(path_evolution_knots)
    minimum, maximum = state_bounds(
        path_evolution_knots,
        problem_params["state_min_bounds"][0],
        problem_params["state_max_bounds"][0],
    )
    return {
        **problem_params,
        "initial_state": jnp.concatenate([simulator_state[:6], jnp.zeros(1)]),
        "initial_control": previous_control,
        "reference_state_trajectory": state_reference,
        "reference_control_trajectory": control_reference,
        "state_min_bounds": minimum,
        "state_max_bounds": maximum,
        "dynamics_state_dot_params": dynamics_parameters_at(path_evolution_knots),
    }, path_evolution_knots


def simulate(simulator_state, planned_controls, control_path_evolution_knots):
    """Simulate one MPC period while interpolating the planned controls."""

    def state_dot(_, state, args):
        control_plan, path_evolution_knots = args
        current_path_evolution = state[-1]
        control = jax.vmap(jnp.interp, in_axes=(None, None, 0))(
            current_path_evolution, path_evolution_knots, control_plan.T
        )
        return simulator_model.state_dot(
            state,
            control,
            dynamics_parameters_at(current_path_evolution),
        )

    return diffrax.diffeqsolve(
        diffrax.ODETerm(state_dot),
        diffrax.Tsit5(),
        t0=0.0,
        t1=MPC_PERIOD,
        dt0=MPC_PERIOD / SIMULATOR_SUBSTEPS,
        y0=simulator_state,
        args=(planned_controls, control_path_evolution_knots),
        saveat=diffrax.SaveAt(t1=True),
    ).ys[0]


def shift_knots(knots, path_delta):
    source = jnp.arange(HORIZON + 1) + path_delta / PATH_STEP
    source = jnp.clip(source, 0.0, HORIZON)
    lower = jnp.floor(source).astype(int)
    upper = jnp.minimum(lower + 1, HORIZON)
    fraction = (source - lower)[:, None]
    return knots[lower] + fraction * (knots[upper] - knots[lower])


def rollout():
    def mpc_step(carry, step):
        states, controls, simulator_state, previous_path = carry
        path_delta = jnp.maximum(simulator_state[-1] - previous_path, 0.0)
        states = shift_knots(states, path_delta)
        controls = shift_knots(controls, path_delta)
        params, control_path_evolution_knots = mpc_parameters(
            simulator_state, controls[0]
        )
        guess = solver.initial_guess(params)._replace(
            states=states.at[0].set(params["initial_state"]),
            controls=controls,
        )
        solution = solver.solve(guess, params, {})
        next_state = simulate(
            simulator_state, solution.controls, control_path_evolution_knots
        )
        admm_iterations = jnp.sum(solution.admm_iters)

        def print_progress(_):
            jax.debug.print(
                (
                    "MPC step {step}/{total}: path={path:.1f} m, "
                    "lateral deviation={deviation:.2f} m, ADMM iterations={iterations}"
                ),
                step=step + 1,
                total=SIM_STEPS,
                path=next_state[-1],
                deviation=next_state[4],
                iterations=admm_iterations,
            )
            return 0

        if VERBOSE:
            jax.lax.cond(
                (step + 1) % PROGRESS_PRINT_INTERVAL == 0,
                print_progress,
                lambda _: 0,
                operand=None,
            )
        log = next_state, solution.controls[0], admm_iterations
        carry = solution.states, solution.controls, next_state, simulator_state[-1]
        return carry, log

    carry = (
        initial_reference,
        initial_controls,
        initial_simulator_state,
        initial_simulator_state[-1],
    )
    return jax.lax.scan(mpc_step, carry, jnp.arange(SIM_STEPS))


def track_coordinates(path_evolution, lateral_deviation):
    heading = interpolate_trajectory(path_evolution, "phi")
    east = interpolate_trajectory(path_evolution, "E")
    north = interpolate_trajectory(path_evolution, "N")
    return (
        east - lateral_deviation * jnp.cos(heading),
        north - lateral_deviation * jnp.sin(heading),
    )


def plot_results(states, controls, admm_iterations):
    output_directory = Path(__file__).with_name("racing_output")
    output_directory.mkdir(exist_ok=True)
    first = np.searchsorted(TRAJECTORY["s"], states[0, -1])
    last = np.searchsorted(TRAJECTORY["s"], states[-1, -1])
    track_progress = TRAJECTORY["s"][first : last + 1]
    center_east, center_north = track_coordinates(track_progress, 0.0)
    minimum_east, minimum_north = track_coordinates(
        track_progress, interpolate_trajectory(track_progress, "e_min")
    )
    maximum_east, maximum_north = track_coordinates(
        track_progress, interpolate_trajectory(track_progress, "e_max")
    )
    vehicle_east, vehicle_north = track_coordinates(states[:, -1], states[:, 4])

    figure, axes = plt.subplots(4, 3, figsize=(15, 12))
    top_view = axes.flat[0]
    top_view.plot(center_east, center_north, "k--", label="reference")
    top_view.plot(minimum_east, minimum_north, color="0.5", label="track edges")
    top_view.plot(maximum_east, maximum_north, color="0.5")
    top_view.plot(vehicle_east, vehicle_north, label="closed loop")
    top_view.set_aspect("equal")
    top_view.set_xlabel("east [m]")
    top_view.set_ylabel("north [m]")
    top_view.legend()
    top_view.grid(alpha=0.3)

    time = MPC_PERIOD * np.arange(1, SIM_STEPS + 1)
    histories = [
        *zip(states.T, simulator_model.names_states),
        *zip(controls.T, simulator_model.names_controls),
        (admm_iterations, "ADMM iterations"),
    ]
    for axis, (values, name) in zip(axes.flat[1:], histories):
        axis.plot(time, values)
        axis.set_ylabel(name)
        axis.set_xlabel("time [s]")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_directory / "results.png", dpi=150)
    plt.close("all")
    print(f"plot saved to {output_directory / 'results.png'}")


if __name__ == "__main__":
    print(
        f"Starting {SIM_STEPS}-step racing rollout "
        f"({HORIZON + 1} knots, {PATH_STEP:.1f} m spacing, one SQP iteration)"
    )
    print("JIT-compiling and running the closed loop...")
    start_time = perf_counter()
    _, (states, controls, admm_iterations) = jax.jit(rollout)()
    jax.block_until_ready(states)
    print(f"Closed loop finished in {perf_counter() - start_time:.1f} s")

    states = np.asarray(states)
    controls = np.asarray(controls)
    admm_iterations = np.asarray(admm_iterations)
    minimum_deviation = interpolate_trajectory(states[:, -1], "e_min")
    maximum_deviation = interpolate_trajectory(states[:, -1], "e_max")
    track_violation = np.maximum(
        minimum_deviation - states[:, 4],
        states[:, 4] - maximum_deviation,
    )
    print(
        f"progress: {states[-1, -1]:.1f} m, "
        f"max |lateral deviation|: {np.max(np.abs(states[:, 4])):.2f} m, "
        f"max track violation: {max(0.0, np.max(track_violation)):.2f} m"
    )
    print("Saving the combined results figure...")
    plot_results(states, controls, admm_iterations)
