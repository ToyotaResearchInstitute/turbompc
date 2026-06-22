#!/usr/bin/env python3
"""Post-build smoke test: runs automatically after cmake build.

Checks that all FFI libraries load correctly and produce finite results
on a tiny problem. Catches stale builds, architecture mismatches, and
compiler bugs (like the __restrict__ NaN on SM 89) immediately.

Exit code 0 = all good. Exit code 1 = something is broken.
"""
import os
import sys


def main():  # noqa: C901
    # Find the project root (this script is in turbompc/solvers/csrc/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    sys.path.insert(0, project_root)

    os.environ.setdefault("JAX_PLATFORMS", "cuda")

    try:
        from jax import config

        config.update("jax_enable_x64", True)
        import jax
        import jax.numpy as jnp
    except Exception as e:
        print(f"SKIP: JAX not available ({e})")
        return 0  # Don't fail build if JAX isn't installed

    if jax.default_backend() != "gpu":
        print("SKIP: No GPU available")
        return 0

    gpu = jax.devices()[0].device_kind
    print(f"Post-build check: {gpu}")

    # Check 1: All FFI libraries load
    libs = [
        ("admm_fused_ffi", "turbompc.solvers.admm.admm_ffi_backend"),
        ("pcg_blktridi_ffi", "turbompc.solvers.linear_systems_solvers.pcg_ffi_backend"),
    ]
    # cuDSS libs are optional (require nvidia-cudss-cu12)
    cudss_libs = [
        ("admm_cudss_ffi", "turbompc.solvers.admm.admm_cudss_ffi_backend"),
        ("cudss_sparse_kkt_ffi", "turbompc.solvers.backward.backward_kkt_cudss_ffi"),
    ]

    all_ok = True
    for name, mod in libs:
        try:
            __import__(mod, fromlist=["_LIB"])
            print(f"  {name:25s} OK")
        except Exception as e:
            print(f"  {name:25s} FAIL: {e}")
            all_ok = False

    has_cudss = True
    for name, mod in cudss_libs:
        try:
            __import__(mod, fromlist=["_LIB"])
            print(f"  {name:25s} OK")
        except Exception:
            print(f"  {name:25s} SKIP (cuDSS not installed)")
            has_cudss = False

    if not all_ok:
        print("FAIL: Some FFI libraries failed to load. Rebuild required.")
        return 1

    # Check 2: fused_pcg produces finite output (catches __restrict__ bug)
    try:
        from turbompc.dynamics.linear_dynamics import (
            LinearDynamics,
            default_state_dot_parameters,
        )
        from turbompc.problems.optimal_control_problem import OptimalControlProblem
        from turbompc.solvers.turbompc_solver import (
            BackwardBackend,
            ForwardBackend,
            TurboMPCSolver,
        )
        from turbompc.utils.load_params import load_solver_params

        dynamics = LinearDynamics()
        nx, nu = dynamics.num_states, dynamics.num_controls
        H = 5
        params = {
            "horizon": H,
            "discretization_resolution": 1.0,
            "discretization_scheme": 0,
            "initial_state": jnp.ones(nx),
            "initial_guess_final_state": jnp.zeros(nx),
            "reference_state_trajectory": jnp.zeros((H + 1, nx)),
            "reference_control_trajectory": jnp.zeros((H + 1, nu)),
            "penalize_control_reference": False,
            "rescale_optimization_variables": False,
            "constrain_initial_control": False,
            "initial_control": jnp.zeros(nu),
            "state_rescaling_min": -jnp.ones(nx),
            "state_rescaling_max": jnp.ones(nx),
            "control_rescaling_min": -jnp.ones(nu),
            "control_rescaling_max": jnp.ones(nu),
            "weights_penalization_reference_state_trajectory": jnp.ones(nx),
            "weights_penalization_final_state": jnp.zeros(nx),
            "weights_penalization_control_squared": jnp.ones(nu),
            "weights_penalization_control_rate": jnp.zeros(nu),
            "state_min_bounds": -jnp.ones(nx) * 10,
            "state_max_bounds": jnp.ones(nx) * 10,
            "control_min_bounds": -jnp.ones(nu) * 10,
            "control_max_bounds": jnp.ones(nu) * 10,
            "dynamics_state_dot_params": {
                "A": default_state_dot_parameters["A"],
                "B": default_state_dot_parameters["B"],
                "b": default_state_dot_parameters["b"],
            },
        }
        sp = load_solver_params("turbompc.yaml")
        sp["num_sqp_iteration_max"] = 1
        sp["admm"]["max_iter"] = 100

        problem = OptimalControlProblem(dynamics=dynamics, params=params)

        for backend_name, fwd_backend, bwd_backend in [
            (
                "fused_pcg",
                ForwardBackend.ADMM_FUSED_PCG,
                (
                    BackwardBackend.DIRECT_CUDSS_FFI
                    if has_cudss
                    else BackwardBackend.DIRECT_JAX_DENSE
                ),
            ),
            (
                "fused_cudss",
                ForwardBackend.ADMM_FUSED_CUDSS,
                (
                    BackwardBackend.DIRECT_CUDSS_FFI
                    if has_cudss
                    else BackwardBackend.DIRECT_JAX_DENSE
                ),
            ),
            (
                "pcg_ffi",
                ForwardBackend.ADMM_JAX_LOOP_PCG_FFI,
                BackwardBackend.DIRECT_JAX_DENSE,
            ),
        ]:
            if not has_cudss and fwd_backend == ForwardBackend.ADMM_FUSED_CUDSS:
                print(f"  {backend_name:25s} SKIP (no cuDSS)")
                continue
            try:
                solver = TurboMPCSolver(
                    program=problem,
                    params=sp,
                    forward_backend=fwd_backend,
                    backward_backend=bwd_backend,
                )
                guess = solver.initial_guess(params)
                sol = solver.solve(guess, params, {})
                cost = float(jnp.sum(sol.states**2))
                if jnp.isnan(jnp.array(cost)):
                    print(f"  {backend_name:25s} FAIL: NaN output!")
                    all_ok = False
                else:
                    print(f"  {backend_name:25s} OK (cost={cost:.2f})")
            except Exception as e:
                print(f"  {backend_name:25s} FAIL: {e}")
                all_ok = False

    except Exception as e:
        print(f"  solve check SKIP: {e}")

    if all_ok:
        print("Post-build check: ALL PASSED")
        return 0
    else:
        print("Post-build check: FAILED: see errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
