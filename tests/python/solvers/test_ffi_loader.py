"""
Tests for FFI shared library discovery.
"""

import pytest

from turbompc.solvers import ffi_loader

LIB_NAME = "libadmm_fused_ffi.so"


def _touch(path):
    """Create an empty file, including any missing parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


@pytest.fixture
def package_ffi_dir(tmp_path, monkeypatch):
    """Redirect the loader's package-relative root into a temporary tree."""
    solvers_dir = tmp_path / "turbompc" / "solvers"
    monkeypatch.setattr(ffi_loader, "_SOLVERS_DIR", solvers_dir)
    monkeypatch.setattr(ffi_loader, "_installed_ffi_dirs", lambda: [])
    return solvers_dir / ffi_loader._FFI_PACKAGE_SUBDIR


@pytest.fixture
def editable_site_packages(tmp_path, monkeypatch):
    """Simulate an editable install."""
    installed = tmp_path / "site-packages" / "turbompc" / "solvers" / "_ffi"
    monkeypatch.setattr(ffi_loader, "_installed_ffi_dirs", lambda: [installed])
    return installed


def test_finds_library_in_package_dir(package_ffi_dir):
    _touch(package_ffi_dir / LIB_NAME)

    assert ffi_loader.find_ffi_lib(LIB_NAME) == package_ffi_dir / LIB_NAME


def test_finds_library_installed_by_editable_install(
    package_ffi_dir, editable_site_packages
):
    """An editable install redirects imports to the source tree, so the
    site-packages copy is the only one that exists."""
    _touch(editable_site_packages / LIB_NAME)

    assert ffi_loader.find_ffi_lib(LIB_NAME) == editable_site_packages / LIB_NAME


def test_package_dir_wins_over_editable_site_packages(
    package_ffi_dir, editable_site_packages
):
    _touch(package_ffi_dir / LIB_NAME)
    _touch(editable_site_packages / LIB_NAME)

    assert ffi_loader.find_ffi_lib(LIB_NAME) == package_ffi_dir / LIB_NAME


def test_missing_library_error_lists_every_searched_directory(
    package_ffi_dir, editable_site_packages
):
    """The message has to be actionable: this exact failure was diagnosed by
    hand because the searched paths were not reported."""
    with pytest.raises(FileNotFoundError, match="TURBOMPC_CUDA=ON") as excinfo:
        ffi_loader.find_ffi_lib(LIB_NAME)

    message = str(excinfo.value)
    for directory in (package_ffi_dir, editable_site_packages):
        assert str(directory) in message


def test_search_paths_are_deduplicated(package_ffi_dir, monkeypatch):
    """A non-editable install's platlib entry duplicates the first entry."""
    monkeypatch.setattr(ffi_loader, "_installed_ffi_dirs", lambda: [package_ffi_dir])

    paths = ffi_loader._search_paths(LIB_NAME)

    assert len(paths) == len(set(paths))


def test_installed_ffi_dirs_targets_this_package():
    """Guards against the package layout and the CMake install dir desyncing."""
    for directory in ffi_loader._installed_ffi_dirs():
        assert directory.parts[-3:] == ("turbompc", "solvers", "_ffi")


def test_env_var_overrides_all_search_paths(package_ffi_dir, tmp_path, monkeypatch):
    _touch(package_ffi_dir / LIB_NAME)
    override = tmp_path / "elsewhere" / "libadmm_fused_profile_ffi.so"
    _touch(override)
    monkeypatch.setenv("ADMM_FFI_LIB", str(override))

    found = ffi_loader.find_ffi_lib(LIB_NAME, env_var="ADMM_FFI_LIB")

    assert found == override.resolve()


def test_env_var_pointing_at_missing_file_raises(
    package_ffi_dir, tmp_path, monkeypatch
):
    _touch(package_ffi_dir / LIB_NAME)
    monkeypatch.setenv("ADMM_FFI_LIB", str(tmp_path / "nope.so"))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        ffi_loader.find_ffi_lib(LIB_NAME, env_var="ADMM_FFI_LIB")


def test_unset_env_var_falls_back_to_search_paths(package_ffi_dir, monkeypatch):
    _touch(package_ffi_dir / LIB_NAME)
    monkeypatch.delenv("ADMM_FFI_LIB", raising=False)

    found = ffi_loader.find_ffi_lib(LIB_NAME, env_var="ADMM_FFI_LIB")

    assert found == package_ffi_dir / LIB_NAME
