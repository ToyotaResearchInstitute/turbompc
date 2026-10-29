"""Test coverage for the custom ``turbompc_build.py`` backend."""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[3] / "turbompc_build.py"

if not _MODULE_PATH.is_file():  # pragma: no cover - source checkouts only
    pytest.skip("build backend not present", allow_module_level=True)

_spec = importlib.util.spec_from_file_location(
    "turbompc_build_under_test", _MODULE_PATH
)
backend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backend)

EXPECTED = backend.CUDA_BUILD_REQUIRES


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Start each test from a clean environment."""
    for var in (
        "TURBOMPC_CUDA",
        "CUDACXX",
        "CMAKE_CUDA_COMPILER",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDAToolkit_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def no_cuda_compiler(monkeypatch):
    """Make compiler detection report that no CUDA compiler is available."""
    monkeypatch.setattr(backend.shutil, "which", lambda _name: None)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("ON", "ON"),
        ("on", "ON"),
        (" 1 ", "ON"),
        ("TRUE", "ON"),
        ("yes", "ON"),
        ("OFF", "OFF"),
        ("0", "OFF"),
        ("false", "OFF"),
        ("no", "OFF"),
        ("AUTO", "AUTO"),
        ("auto", "AUTO"),
    ],
)
def test_normalize_tristate_spellings(value, expected):
    assert backend.normalize(value) == expected


def test_unrecognized_value_passes_through_for_cmake_to_reject():
    assert backend.normalize("maybe") == "MAYBE"


def test_defaults_to_auto():
    assert backend.cuda_setting() == "AUTO"


def test_env_var_is_honoured_and_stripped(monkeypatch):
    monkeypatch.setenv("TURBOMPC_CUDA", " off ")

    assert backend.cuda_setting() == "OFF"


def test_off_requires_nothing_even_with_a_compiler(monkeypatch):
    monkeypatch.setenv("TURBOMPC_CUDA", "OFF")

    assert backend.cuda_build_requires() == []


def test_on_requires_cudss_even_without_a_compiler(monkeypatch, no_cuda_compiler):
    """ON must still request cuDSS; CMake is what reports the missing nvcc."""
    monkeypatch.setenv("TURBOMPC_CUDA", "ON")

    assert backend.cuda_build_requires() == EXPECTED


def test_auto_requires_nothing_without_a_compiler(no_cuda_compiler):
    assert backend.cuda_build_requires() == []


def test_auto_requires_cudss_when_nvcc_is_on_path(monkeypatch):
    monkeypatch.setattr(backend.shutil, "which", lambda _name: "/usr/bin/nvcc")

    assert backend.cuda_build_requires() == EXPECTED


def test_cudacxx_env_var_counts_as_a_compiler(monkeypatch, tmp_path, no_cuda_compiler):
    nvcc = tmp_path / "nvcc"
    nvcc.write_bytes(b"")
    monkeypatch.setenv("CUDACXX", str(nvcc))

    assert backend.have_cuda_compiler() is True


def test_cuda_home_counts_as_a_compiler(monkeypatch, tmp_path, no_cuda_compiler):
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir()
    nvcc.write_bytes(b"")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))

    assert backend.have_cuda_compiler() is True


def test_nonexistent_cudacxx_is_not_a_compiler(monkeypatch, tmp_path, no_cuda_compiler):
    monkeypatch.setenv("CUDACXX", str(tmp_path / "missing"))

    assert backend.have_cuda_compiler() is False
