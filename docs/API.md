# TurboMPC API

TurboMPC provides a differentiable MPC solver with a JAX API:
`TurboMPCSolver`.

## Quick Start

```python
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)  # double precision

from turbompc.dynamics.drone_dynamics import (
    DroneDynamics,
    drone_parameters,
    drone_state_dot_parameters
)
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers import BackwardBackend, ForwardBackend, TurboMPCSolver
from turbompc.utils.load_params import load_problem_params, load_solver_params

dynamics = DroneDynamics(drone_parameters)
params = load_problem_params("drone.yaml")
N = int(params["horizon"])
nx = dynamics.num_states
nu = dynamics.num_controls

params["initial_state"] = 0.1 * jnp.ones(nx)
params["initial_guess_final_state"] = jnp.zeros(nx)
params["reference_state_trajectory"] = jnp.zeros((N + 1, nx))
params["reference_control_trajectory"] = jnp.zeros((N + 1, nu))
params["dynamics_state_dot_params"] = drone_state_dot_parameters
params["control_min_bounds"] = -10.0 * jnp.ones(nu)
params["control_max_bounds"] = 10.0 * jnp.ones(nu)
problem = OptimalControlProblem(dynamics=dynamics, params=params)

solver_params = load_solver_params("turbompc.yaml")
solver = TurboMPCSolver(problem, params=solver_params)

solution = solver.solve(solver.initial_guess(params), problem_params=params)
u0 = solution.controls[0]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(states[:, 0], states[:, 1], "o-", ms=3, label="open-loop plan")
ax.plot(states[0, 0], states[0, 1], "k^", ms=7, label="start")
ax.plot(0.0, 0.0, "kx", ms=8, label="goal")
ax.legend(loc="best")
plt.show()
```

## Solver API

Works also on the CPU:
```python
solver = TurboMPCSolver(
    program=problem,
    params=load_solver_params("turbompc.yaml"),
    forward_backend="admm_jax_loop_pcg",
    backward_backend="admm_jax_loop_pcg",
    use_full_hessian=False,
)
```
Recommended on the GPU (default parameters):
```python
solver = TurboMPCSolver(
    program=problem,
    params=load_solver_params("turbompc.yaml"),
    forward_backend="admm_fused_cudss",
    backward_backend="direct_cudss_ffi",
    use_full_hessian=True,
)
```

Constructor arguments:

| Argument | Meaning |
|---|---|
| `program` | `OptimalControlProblem` or compatible problem adapter |
| `params` | Solver parameter dict; defaults to `turbompc.yaml` |
| `name` | Optional display name |
| `forward_backend` | Backend for forward QP solves |
| `backward_backend` | Backend for gradients through the solve |
| `use_full_hessian` | Include dynamics Hessian terms in gradient calculations |

Common solver params:

| Key | Meaning |
|---|---|
| `num_sqp_iteration_max` | Max SQP iterations |
| `tol_convergence` | SQP stopping tolerance |
| `convergence_criterion` | `"first_order"` or `"step"` |
| `linesearch` | Enable backtracking line search |
| `linesearch_alphas` | Candidate line-search step sizes |
| `admm.rho`, `admm.sigma` | ADMM penalty and regularization values |
| `admm.max_iter` | Max ADMM iterations per SQP step |
| `admm.eps_abs`, `admm.eps_rel` | ADMM stopping tolerances |
| `admm.relaxation_parameter` | ADMM relaxation parameter |
| `admm.pcg.max_iter`, `admm.pcg.tol_epsilon` | PCG settings |

Backend strings:

| Backend | Forward | Backward |
|---|---:|---:|
| `admm_jax_loop_pcg` | yes | yes |
| `admm_jax_loop_pcg_ffi` | yes | yes |
| `admm_jax_loop_cudss_ffi` | yes | yes |
| `admm_jax_loop_jax_dense` | yes | yes |
| `admm_fused_pcg` | yes | yes |
| `admm_fused_cudss` | yes | yes |
| `direct_jax_dense` | no | yes |
| `direct_cudss_ffi` | no | yes |

CUDA FFI backends require the CUDA
extensions described in [cuda-ffi-backends.md](cuda-ffi-backends.md).

## Solving

```python
guess = solver.initial_guess(problem_params)
solution = solver.solve(guess, problem_params=problem_params)
```

`initial_guess(params)` returns a `TurboMPCSolution` containing a
straight-line state trajectory, near-zero controls, zero slack, and empty
warm-start state.

`solve(initial_guess, problem_params, weights={})` returns a
`TurboMPCSolution`:

| Field | Shape | Meaning |
|---|---:|---|
| `states` | `(N+1, nx)` | Planned states |
| `controls` | `(N+1, nu)` | Planned controls; apply `controls[0]` |
| `slack` | `(N+1, m)` | Inequality slack values |
| `status` | scalar | `0` on normal return |
| `num_iter` | scalar | SQP iterations used |
| `convergence_error` | scalar | Final convergence metric |
| `admm_iters` | `(num_sqp_iteration_max,)` | ADMM iterations per SQP step |
| `linesearch_alphas` | optional array | Accepted line-search steps |
| `admm_state` | `ADMMState` | Internal state for warm-starting |
| `solver_stats` | `SolverStats` | Per-SQP iteration stats |
| `kkt_state` | `KKTState` | Solver metadata used for differentiation |


## Closed-Loop Warm Starts

For receding-horizon MPC, either reuse the previous MPC solution as an initial guess, or shift the previous solution and reuse it:

```python
guess = solution._replace(
    states=jnp.concatenate([solution.states[1:], solution.states[-1:]], axis=0),
    controls=jnp.concatenate([solution.controls[1:], solution.controls[-1:]], axis=0),
    slack=jnp.concatenate([solution.slack[1:], solution.slack[-1:]], axis=0),
)

solution = solver.solve(
    guess,
    problem_params,
    {"initial_state": current_state},
)
u0 = solution.controls[0]
```

## Gradients

The `weights` argument is a pytree of parameters merged into
`problem_params` before solving. Use it for learned costs, references,
slack weights, initial state (for closed-loop multi-steps rollouts) or other differentiable MPC parameters.

```python
weights = {
    "weights_penalization_reference_state_trajectory": learned_Q,
    "weights_penalization_control_squared": learned_R,
}
solution = solver.solve(solver.initial_guess(params), params, weights)
```

Gradients flow from `solution.states`, `solution.controls`, and
`solution.slack` to the leaves in `weights`.

```python
import jax

def loss(weights):
    sol = solver.solve(solver.initial_guess(params), params, weights)
    return jnp.sum(sol.states[:, :3] ** 2) + 1e-3 * jnp.sum(sol.controls**2)

value, grads = jax.value_and_grad(loss)(weights)
```

## Constraints and Slack

Box constraints are enabled by finite bound arrays in `problem_params`.
Potentially-active constraints are determined when the `OptimalControlProblem` is
created: recreate the problem and solver if the constraints change.

For soft constraints, use a slack-enabled problem class or adapter, set
`use_slack_variables=True` and provide `slack_penalization_weight`.


## Problem API

Use `OptimalControlProblem(dynamics, params)` for standard tracking MPC.
`BaseOptimalControlProblem` documents the lower-level API for custom problem
families.

Trajectory conventions:

| Quantity | Shape | Meaning |
|---|---:|---|
| `states` | `(N+1, nx)` | State trajectory from stage `0` through terminal stage `N` |
| `controls` | `(N+1, nu)` | Control trajectory; includes terminal control |

Dynamics and control-rate costs use `t = 0, ..., N-1`; terminal costs use
stage `N`.

`dynamics` must provide:

| Name | Meaning |
|---|---|
| `num_states` | State dimension `nx` |
| `num_controls` | Control dimension `nu` |
| `state_dot(x, u, params)` | Continuous-time dynamics |

Core `params`:

| Key | Shape | Meaning |
|---|---:|---|
| `horizon` | scalar | MPC horizon `N` |
| `discretization_resolution` | scalar | Time step |
| `discretization_scheme` | scalar | `0` Euler, `1` midpoint, `2` RK4, `10` implicit trapezoid |
| `initial_state` | `(nx,)` | Fixed first state |
| `initial_guess_final_state` | `(nx,)` | End point for the default initial guess |
| `reference_state_trajectory` | `(N+1, nx)` | State reference |
| `reference_control_trajectory` | `(N+1, nu)` | Control reference |
| `weights_penalization_reference_state_trajectory` | `(nx,)` or `(N+1, nx)` | State tracking weights |
| `weights_penalization_final_state` | `(nx,)` or `(N+1, nx)` | Extra terminal state weights |
| `weights_penalization_control_squared` | `(nu,)` or `(N+1, nu)` | Control effort weights |
| `weights_penalization_control_rate` | `(nu,)`, `(nu,nu)`, `(N,nu)`, or `(N,nu,nu)` | Control-rate weights |
| `state_min_bounds`, `state_max_bounds` | `(nx,)` or `(N+1, nx)` | Optional state bounds |
| `control_min_bounds`, `control_max_bounds` | `(nu,)` or `(N+1, nu)` | Optional control bounds |
| `dynamics_state_dot_params` | pytree | Optional dynamics parameters |

For custom inequality constraints, subclass `OptimalControlProblem`
and override `step_inequality_constraints(state, control, params)`.
Return `(g, lower, upper)` for one time step. In trajectory-level calls,
array leaves in `params` with leading dimension `N+1` are sliced before
`step_inequality_constraints` is called; other values are passed unchanged.
The built-in obstacle-avoidance variant is available from
`turbompc.problems.obstacle_avoidance`.

For custom costs, override `stage_cost` or `terminal_cost`.
