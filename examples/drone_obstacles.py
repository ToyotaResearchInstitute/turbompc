"""Drone obstacle-avoidance optimal control script."""

from time import time

import matplotlib.pyplot as plt
import numpy as np
from jax import config
from matplotlib.patches import Circle

config.update("jax_enable_x64", True)  # use double precision

import jax

from examples.drone_utils import OBS_CENTERS, OBS_RADII, make_drone_config
from turbompc.solvers.turbompc_solver import (
    BACKWARD_BACKEND_CHOICES,
    FORWARD_BACKEND_CHOICES,
    TurboMPCSolver,
    parse_backward_backend,
    parse_forward_backend,
)

# Available solver backends
print("Forward backends:")
for b in FORWARD_BACKEND_CHOICES:
    print(f"  {b}")

print("\nBackward backends:")
for b in BACKWARD_BACKEND_CHOICES:
    print(f"  {b}")

print(f"\nJAX devices: {jax.devices()}\n")


# OCP dimensions
horizon = 50  # MPC prediction horizon
dt = 1.0  # discretisation time step (s)

# Discretization discretization_scheme: 0 = Euler, 2 = RK4, 10 = implicit trapezoidal
discretization_scheme = 2

# Cost weights
rd_weight = 0.01  # control-rate weight r_d
slack_weight = 10.0  # slack penalty λ (only used in the slack case)

# Backend selection
backend_names = (
    "admm_jax_loop_eigen",
    "admm_jax_loop_pcg",
)

max_sqp_iter = 20
sqp_tol = 1e-4
cfg_open_loop = make_drone_config(
    use_slack=True,
    slack_weight=slack_weight,
    rd_weight=rd_weight,
    discretization_scheme=discretization_scheme,
    dt=dt,
    horizon=horizon,
    sqp_iter=max_sqp_iter,
    admm_tol=1e-5,
    admm_max_iter=5000,
    sqp_tol=sqp_tol,
)

problem = cfg_open_loop.problem_class(
    dynamics=cfg_open_loop.dynamics,
    params=cfg_open_loop.problem_params,
)
weights = {k: cfg_open_loop.problem_params[k] for k in cfg_open_loop.weight_keys}
results = {}

for backend_name in backend_names:
    solver = TurboMPCSolver(
        program=problem,
        params=cfg_open_loop.solver_params,
        forward_backend=parse_forward_backend(backend_name),
        backward_backend=parse_backward_backend(backend_name),
    )
    solve = jax.jit(solver.solve)
    initial_guess = solver.initial_guess(cfg_open_loop.problem_params)

    warmup = solve(initial_guess, cfg_open_loop.problem_params, weights)
    jax.block_until_ready(warmup)

    start = time()
    solution = solve(initial_guess, cfg_open_loop.problem_params, weights)
    jax.block_until_ready(solution)
    elapsed = time() - start
    results[backend_name] = solution, elapsed

    print(f"\n{backend_name}:")
    print(f"  solve elapsed: {elapsed:.6f} s")
    print(f"  SQP iterations: {int(solution.num_iter)} (max = {max_sqp_iter})")
    print(f"  ADMM iterations: {solution.admm_iters}")
    print(
        f"  convergence_error: {float(solution.convergence_error):.3e}"
        f" (max = {sqp_tol})"
    )

solution, eigen_time = results["admm_jax_loop_eigen"]
_, pcg_time = results["admm_jax_loop_pcg"]
print("\nRuntime comparison:")
print(f"  Eigen: {eigen_time:.6f} s")
print(f"  PCG:   {pcg_time:.6f} s")
print(f"  PCG/Eigen runtime ratio: {pcg_time / eigen_time:.3f}")

# Plot the solution.
states = np.asarray(solution.states)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(states[:, 0], states[:, 1], "o-", ms=3, label="Eigen open-loop plan")
ax.plot(states[0, 0], states[0, 1], "k^", ms=7, label="start")
ax.plot(0.0, 0.0, "kx", ms=8, label="goal")
for center, radius in zip(OBS_CENTERS, OBS_RADII):
    ax.add_patch(Circle(center, radius, color="tab:red", alpha=0.25))
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x position")
ax.set_ylabel("y position")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
plt.show()
