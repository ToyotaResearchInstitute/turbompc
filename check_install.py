#!/usr/bin/env python3
"""Verify turbompc installation: Python, JAX, CUDA, FFI kernels."""
import sys


def main():
    ok = True

    print(
        f"Python: {sys.executable} ({sys.version_info.major}.{sys.version_info.minor})"
    )

    try:
        import jax

        print(f"JAX: {jax.__version__}, devices: {jax.devices()}")
    except ImportError:
        print("JAX: NOT INSTALLED")
        ok = False

    try:
        import jax.numpy as jnp

        x = jnp.linalg.solve(jnp.eye(3, dtype=jnp.float64), jnp.ones(3))
        assert float(jnp.sum(x)) == 3.0
        print("cuSolver: OK")
    except Exception as e:
        print(f"cuSolver: FAILED ({e})")
        ok = False

    try:
        from turbompc.solvers.turbompc_solver import TurboMPCSolver  # noqa: F401

        print("turbompc: OK")
    except ImportError as e:
        print(f"turbompc: FAILED ({e})")
        ok = False

    for name, import_path in [
        ("fused_pcg FFI", "turbompc.solvers.admm.admm_ffi_backend"),
        ("fused_cudss FFI", "turbompc.solvers.admm.admm_cudss_ffi_backend"),
        ("pcg FFI", "turbompc.solvers.linear_systems_solvers.pcg_ffi_backend"),
        ("cudss FFI", "turbompc.solvers.linear_systems_solvers.cudss_ffi_backend"),
        ("cudss KKT FFI", "turbompc.solvers.backward.backward_kkt_cudss_ffi"),
    ]:
        try:
            mod = __import__(import_path, fromlist=["_find_lib"])
            mod._find_lib()
            print(f"{name}: OK")
        except (ImportError, FileNotFoundError):
            print(f"{name}: NOT BUILT (run: make cuda)")
        except Exception as e:
            print(f"{name}: ERROR ({e})")

    print()
    if ok:
        print("All checks passed")
    else:
        print("Some checks FAILED — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
