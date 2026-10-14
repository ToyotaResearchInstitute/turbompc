"""Shared discovery and loading of the CUDA FFI libraries."""
from __future__ import annotations

import ctypes
import functools
import os
import site
import sysconfig
from pathlib import Path

_SOLVERS_DIR = Path(__file__).resolve().parent
_FFI_PACKAGE_SUBDIR = "_ffi"
_PACKAGE_PARTS = tuple(__package__.split("."))
_NVIDIA_LIB_STEMS = (
    "libnvJitLink",
    "libcudart",
    "libcublasLt",
    "libcublas",
    "libcusparse",
    "libcudss",
)

_preloaded = False


def _nvidia_lib_dirs() -> list[Path]:
    """Return ``lib/`` directories of installed pip ``nvidia-*`` wheels."""
    try:
        import nvidia
    except ImportError:
        return []
    dirs: list[Path] = []
    for base in nvidia.__path__:
        dirs.extend(sorted(Path(base).glob("*/lib")))
    return dirs


def _load_first(dirs: list[Path], stem: str) -> None:
    for lib_dir in dirs:
        for candidate in sorted(lib_dir.glob(f"{stem}.so*")):
            try:
                ctypes.CDLL(str(candidate))
                return
            except OSError:
                continue


def preload_nvidia_libs() -> None:
    """Load CUDA runtime libraries from pip nvidia wheels into the process."""
    global _preloaded
    if _preloaded:
        return
    _preloaded = True
    dirs = _nvidia_lib_dirs()
    for stem in _NVIDIA_LIB_STEMS:
        _load_first(dirs, stem)


@functools.lru_cache(maxsize=None)
def _installed_ffi_dirs() -> list[Path]:
    """Return ``_ffi`` directories inside installed copies of this package."""
    paths = sysconfig.get_paths()
    roots = [paths.get("platlib"), paths.get("purelib")]
    try:
        roots.append(site.getusersitepackages())
    except (AttributeError, OSError):  # pragma: no cover - unusual interpreters
        pass
    dirs: list[Path] = []
    for root in roots:
        if not root:
            continue
        candidate = Path(root, *_PACKAGE_PARTS, _FFI_PACKAGE_SUBDIR)
        if candidate not in dirs:
            dirs.append(candidate)
    return dirs


def _search_paths(name: str) -> list[Path]:
    """Return every location searched for ``name``, in precedence order."""
    candidates = [
        _SOLVERS_DIR / _FFI_PACKAGE_SUBDIR / name,  # installed, non-editable
        *(directory / name for directory in _installed_ffi_dirs()),  # editable
    ]
    return list(dict.fromkeys(candidates))


def find_ffi_lib(name: str, env_var: str | None = None) -> Path:
    """Locate an FFI shared library.

    Args:
        name: Library file name, e.g. ``libadmm_fused_ffi.so``.
        env_var: Optional environment variable naming an override path.

    Returns:
        Path to the shared library.
    """
    if env_var:
        env_lib = os.environ.get(env_var)
        if env_lib:
            path = Path(env_lib).resolve()
            if path.exists():
                return path
            raise FileNotFoundError(f"{env_var}={env_lib} does not exist")
    candidates = _search_paths(name)
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(f"  {path.parent}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find {name}. Reinstall with CUDA enabled: "
        "`TURBOMPC_CUDA=ON pip install .` (add -e for an editable install).\n"
        f"Searched:\n{searched}"
    )


def load_ffi_lib(name: str, env_var: str | None = None) -> ctypes.CDLL:
    """Preload pip CUDA libraries, then load the FFI library ``name``."""
    preload_nvidia_libs()
    return ctypes.cdll.LoadLibrary(str(find_ffi_lib(name, env_var)))
