import os
import subprocess
import sys

import pytest
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.turbompc_solver import BackwardBackend, ForwardBackend

_AVAIL_CACHE: dict[str, bool] = {}


def _has_gpu() -> bool:
    import jax

    return any(d.platform in {"gpu", "cuda"} for d in jax.devices())


def _smoke_check_subprocess(code: str) -> bool:
    """Run a tiny backend smoke check in a subprocess.

    Some CUDA FFI failures abort the process (cannot be caught in-process), so
    we probe availability in a subprocess to keep pytest collection safe.
    """
    env = os.environ.copy()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _check_ffi_available(import_code: str, key: str) -> bool:
    if key in _AVAIL_CACHE:
        return _AVAIL_CACHE[key]
    if not _has_gpu():
        _AVAIL_CACHE[key] = False
        return False
    ok = _smoke_check_subprocess(import_code)
    _AVAIL_CACHE[key] = ok
    return ok


def backend_available(backend) -> bool:
    try:
        if isinstance(backend, SchurSolverBackend):
            return _schur_backend_available(backend)
        if isinstance(backend, (ForwardBackend, BackwardBackend)):
            return _forward_backend_available(backend)
        raise TypeError(f"Unknown backend type: {type(backend)}")
    except Exception:
        return False


def _schur_backend_available(backend: SchurSolverBackend) -> bool:
    if backend == SchurSolverBackend.PCG_FFI:
        return _check_ffi_available(
            "\n".join(
                [
                    "import jax; import jax.numpy as jnp",
                    (
                        "from turbompc.solvers.linear_systems_solvers.pcg_ffi_backend"
                        " import pcg_ffi_solve"
                    ),
                    "T, n = 2, 1",
                    (
                        "S = jnp.array([[[0.0, 1.0, 0.1]], [[0.1, 1.0, 0.0]]],"
                        " dtype=jnp.float32)"
                    ),
                    "Phiinv = jnp.zeros_like(S)",
                    "rhs = jnp.ones((T, n), dtype=jnp.float32)",
                    (
                        "x, _ = pcg_ffi_solve(S, Phiinv, rhs, jnp.zeros_like(rhs),"
                        " eps=1e-6, max_iters=2)"
                    ),
                    "jax.block_until_ready(x)",
                ]
            ),
            "pcg_ffi",
        )
    if backend == SchurSolverBackend.CUDSS_FFI:
        return _check_ffi_available(
            "\n".join(
                [
                    "import jax; import jax.numpy as jnp",
                    (
                        "from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend"
                        " import cudss_ffi_solve"
                    ),
                    "T, n = 2, 1",
                    (
                        "S = jnp.array([[[0.0, 1.0, 0.1]], [[0.1, 1.0, 0.0]]],"
                        " dtype=jnp.float32)"
                    ),
                    "rhs = jnp.ones((T, n), dtype=jnp.float32)",
                    "x = cudss_ffi_solve(S, rhs)",
                    "jax.block_until_ready(x)",
                ]
            ),
            "cudss_ffi",
        )
    # Pure-JAX backends always available
    return True


def _forward_backend_available(backend) -> bool:
    """Check availability for ForwardBackend or BackwardBackend."""
    # val = int(backend)
    name = backend.name

    # Pure JAX backends always available
    if "JAX_LOOP_PCG" in name and "FFI" not in name:
        return True
    if "JAX_DENSE" in name:
        return True

    # FFI backends
    if "PCG_FFI" in name:
        return _schur_backend_available(SchurSolverBackend.PCG_FFI)
    if "CUDSS_FFI" in name and "FUSED" not in name:
        return _schur_backend_available(SchurSolverBackend.CUDSS_FFI)

    # Fused backends need their own smoke check
    if "FUSED_PCG" in name:
        return _check_ffi_available(
            "\n".join(
                [
                    "import jax; import jax.numpy as jnp",
                    "from turbompc.solvers.admm.admm_ffi_backend import _find_lib",
                    "_find_lib()",
                ]
            ),
            "fused_pcg",
        )
    if "FUSED_CUDSS" in name:
        return _check_ffi_available(
            "\n".join(
                [
                    "import jax; import jax.numpy as jnp",
                    (
                        "from turbompc.solvers.admm.admm_cudss_ffi_backend import"
                        " _find_lib"
                    ),
                    "_find_lib()",
                ]
            ),
            "fused_cudss",
        )
    if "DIRECT_CUDSS_FFI" in name:
        return _check_ffi_available(
            "\n".join(
                [
                    "import jax; import jax.numpy as jnp",
                    (
                        "from turbompc.solvers.backward.backward_kkt_cudss_ffi import"
                        " _find_lib"
                    ),
                    "_find_lib()",
                ]
            ),
            "direct_cudss_ffi",
        )

    return True


def backend_param(backend, marks=()):
    """Return a `pytest.param` for a backend, skipping when unavailable."""
    param_marks = list(marks)
    needs_check = False

    if isinstance(backend, SchurSolverBackend):
        if backend in {SchurSolverBackend.PCG_FFI, SchurSolverBackend.CUDSS_FFI}:
            needs_check = True
    elif isinstance(backend, (ForwardBackend, BackwardBackend)):
        name = backend.name
        if any(k in name for k in ("FFI", "FUSED", "DIRECT")):
            needs_check = True
    else:
        raise TypeError(f"Expected backend enum, got {type(backend)}")

    if needs_check:
        param_marks.append(
            pytest.mark.skipif(
                not backend_available(backend),
                reason=f"{backend.name} not built/available",
            )
        )
    return pytest.param(backend, marks=param_marks, id=backend.name)


RECOMMENDED_BACKEND_COMBOS = [
    (ForwardBackend.ADMM_JAX_LOOP_PCG, BackwardBackend.ADMM_JAX_LOOP_PCG, "pure JAX"),
    (
        ForwardBackend.ADMM_FUSED_PCG,
        BackwardBackend.DIRECT_CUDSS_FFI,
        "fused_pcg/direct_cudss",
    ),
    (
        ForwardBackend.ADMM_FUSED_CUDSS,
        BackwardBackend.DIRECT_CUDSS_FFI,
        "fused_cudss/direct_cudss",
    ),
]


def combo_param(fwd, bwd, desc):
    """Return a pytest.param for a (forward, backward) combo with auto-skip."""
    marks = []
    if not backend_available(fwd) or not backend_available(bwd):
        marks.append(pytest.mark.skip(reason=f"{desc}: backend not available"))
    return pytest.param(fwd, bwd, marks=marks, id=desc)


BACKEND_COMBO_PARAMS = [combo_param(f, b, d) for f, b, d in RECOMMENDED_BACKEND_COMBOS]
