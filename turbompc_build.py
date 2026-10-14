"""Allows building the CMakeLists.txt with cuda support from Python."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

CUDA_BUILD_REQUIRES = ["nvidia-cudss-cu12>=0.8"]
ENV_VAR = "TURBOMPC_CUDA"
_TRUTHY = frozenset({"1", "TRUE", "YES", "ON"})
_FALSY = frozenset({"0", "FALSE", "NO", "OFF"})


def normalize(value: str) -> str:
    """Map a value onto AUTO, ON, or OFF."""
    value = value.strip().upper()
    if value in _TRUTHY:
        return "ON"
    if value in _FALSY:
        return "OFF"
    return value


def cuda_setting() -> str:
    """Resolve `TURBOMPC_CUDA` value from the environment.

    Returns:
        One of `"AUTO"`, `"ON"`, or `"OFF"`.
    """
    value = os.environ.get(ENV_VAR)
    return normalize(value) if value else "AUTO"


def have_cuda_compiler() -> bool:
    """Report whether CMake's `check_language(CUDA)` will succeed."""
    for var in ("CUDACXX", "CMAKE_CUDA_COMPILER"):
        candidate = os.environ.get(var)
        if candidate and Path(candidate).exists():
            return True
    if shutil.which("nvcc"):
        return True
    for var in ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT"):
        root = os.environ.get(var)
        if root and (Path(root) / "bin" / "nvcc").exists():
            return True
    return False


def cuda_build_requires() -> list[str]:
    """Return the extra build requirements for a CUDA-enabled build."""
    setting = cuda_setting()
    if setting == "OFF":
        return []
    if setting == "AUTO" and not have_cuda_compiler():
        return []
    return list(CUDA_BUILD_REQUIRES)


def _requires(hook: str, config_settings: dict | None) -> list[str]:
    """Delegate to scikit-build-core, adding CUDA build deps when needed."""
    import scikit_build_core.build as skbuild

    requires = list(getattr(skbuild, hook)(config_settings))
    return requires + cuda_build_requires()


def get_requires_for_build_wheel(config_settings: dict | None = None) -> list[str]:
    return _requires("get_requires_for_build_wheel", config_settings)


def get_requires_for_build_editable(config_settings: dict | None = None) -> list[str]:
    return _requires("get_requires_for_build_editable", config_settings)


def __getattr__(name: str):
    """Forward every other PEP 517 hook to scikit-build-core."""
    if name.startswith("_"):
        raise AttributeError(name)
    import scikit_build_core.build as skbuild

    return getattr(skbuild, name)
