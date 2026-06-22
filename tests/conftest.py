import glob
import os
import shutil
import subprocess

import pytest


def _has_visible_cuda_device() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        return False
    if glob.glob("/dev/nvidia[0-9]*"):
        return True
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


if not _has_visible_cuda_device():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def pytest_addoption(parser):
    parser.addoption(
        "--run-extended",
        action="store_true",
        default=False,
        help="run tests marked as extended",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-extended"):
        return

    skip_extended = pytest.mark.skip(reason="need --run-extended to run")
    for item in items:
        if "extended" in item.keywords:
            item.add_marker(skip_extended)


def pytest_configure(config):
    import jax

    jax.config.update("jax_enable_x64", True)
