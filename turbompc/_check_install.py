#!/usr/bin/env python3
"""
Verify turbompc installation: Python, JAX, and native extensions.
"""
import sys


def main():
    ok = True
    cuda_device = False

    print(
        f"Python: {sys.executable} ({sys.version_info.major}.{sys.version_info.minor})"
    )

    try:
        import jax

        devices = jax.devices()
        cuda_device = any(d.platform in ("gpu", "cuda") for d in devices)
        print(f"JAX: {jax.__version__}, devices: {devices}")
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

    for name, import_path, check_name, option in [
        (
            "fused_pcg FFI",
            "turbompc.solvers.admm.admm_ffi_backend",
            "_LIB",
            "TURBOMPC_CUDA",
        ),
        (
            "fused_cudss FFI",
            "turbompc.solvers.admm.admm_cudss_ffi_backend",
            "_LIB",
            "TURBOMPC_CUDA",
        ),
        (
            "pcg FFI",
            "turbompc.solvers.linear_systems_solvers.pcg_ffi_backend",
            "_LIB",
            "TURBOMPC_CUDA",
        ),
        (
            "cudss FFI",
            "turbompc.solvers.linear_systems_solvers.cudss_ffi_backend",
            "_LIB",
            "TURBOMPC_CUDA",
        ),
        (
            "cudss KKT FFI",
            "turbompc.solvers.backward.backward_kkt_cudss_ffi",
            "_LIB",
            "TURBOMPC_CUDA",
        ),
        (
            "Eigen backend",
            "turbompc.solvers.linear_systems_solvers.eigen_backend",
            "_extension",
            "TURBOMPC_EIGEN",
        ),
    ]:
        # CUDA backends cannot run without a device, so their absence only
        # fails the check when one is visible.
        required = cuda_device if option == "TURBOMPC_CUDA" else True
        try:
            mod = __import__(import_path, fromlist=[check_name])
            attr = getattr(mod, check_name)
            if callable(attr):
                attr()
            print(f"{name}: OK")
        except (ImportError, FileNotFoundError):
            suffix = "" if required else ". No CUDA device, not required"
            print(
                f"{name}: NOT BUILT (reinstall with {option}=ON python -m pip install"
                f" .){suffix}"
            )
            if required:
                ok = False
        except Exception as e:
            print(f"{name}: ERROR ({e})")
            ok = False

    print()
    if ok:
        print("All checks passed")
    else:
        print("Some checks FAILED (see above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
