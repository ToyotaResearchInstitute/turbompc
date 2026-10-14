# Build Guide

TurboMPC is installed from the repository with pip.

The install builds the Python package, Eigen extension, and CUDA code (on GPU platforms).

## Prerequisites

TurboMPC supports Python 3.10+ and is tested on Ubuntu 22.04.

Install the common
build tools and Eigen headers with:

```bash
sudo apt install build-essential cmake libeigen3-dev python3-dev python3-venv
```

A GPU installation requires (check `nvcc --version` and `nvidia-smi`):

- CUDA Toolkit 12.x, including `nvcc`,
- An NVIDIA driver compatible with the toolkit.

`pip` handles Python dependencies.

## Install

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/ToyotaResearchInstitute/turbompc.git
cd turbompc
python -m venv .venv
source .venv/bin/activate
```

For an NVIDIA GPU installation:

```bash
TURBOMPC_CUDA=ON python -m pip install -e ".[cuda12]"
```

For a CPU-only installation:

```bash
TURBOMPC_CUDA=OFF python -m pip install -e .
```

Verify the installation:

```bash
turbompc-check
```

Python edits take effect immediately.

After changing C++ or CUDA source files,
rerun the same install command to rebuild. Compiled CUDA
code is installed under `turbompc/solvers/_ffi` and discovered
automatically.

### Build controls

The `[cuda12]` extra installs the CUDA dependencies. The
`TURBOMPC_CUDA` environment variable controls whether CUDA libraries are
compiled:

| Value | Behavior |
|---|---|
| `ON` | Require CUDA libraries. Fails if the CUDA toolkit is unavailable. |
| `OFF` | Do not build CUDA libraries. |
| `AUTO` (default) | Build CUDA libraries when a toolkit is found. Skip otherwise. |

`AUTO` is the default when the variable is omitted. Set the variable before an

`TURBOMPC_EIGEN` uses the same `ON`, `OFF`, and `AUTO` values for the Eigen
extension. Its default is also `AUTO`.

### GPU architecture

With CMake 3.24+, TurboMPC builds for the GPU on the current machine. Older
CMake versions query `nvidia-smi` and fall back to different architectures
if detection fails.

To build for a specific cuda architecture, pass it through `CMAKE_ARGS`:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" \
  TURBOMPC_CUDA=ON python -m pip install -e ".[cuda12]"
```

The selected value is recorded in `build/skbuild/*/CMakeCache.txt`.

### Direct CMake builds

Use CMake directly only for one-off targets that pip does not install, such as
the profiling library:

```bash
cmake -S . -B build/profile \
  -DCMAKE_BUILD_TYPE=Release \
  -DTURBOMPC_CUDA=ON \
  -DPython3_EXECUTABLE="$(pwd)/.venv/bin/python" \
  -DPython3_FIND_VIRTUALENV=ONLY
cmake --build build/profile -j --target admm_fused_profile_ffi
ADMM_FFI_LIB=build/profile/libadmm_fused_profile_ffi.so python ...
```


## Troubleshooting

### Force a clean rebuild

If a native library is stale or reports an undefined symbol, remove the cached
build and reinstall:

```bash
rm -rf build/skbuild
rm -f turbompc/solvers/linear_systems_solvers/_eigen_block_tridi*.so
TURBOMPC_CUDA=ON python -m pip install -e ".[cuda12]"
```

Use the CPU install command instead on a CPU-only system.

### JAX does not see the GPU

Check the driver and the devices visible to JAX:

```bash
nvidia-smi
python -c "import jax; print(jax.devices())"
```

JAX's CUDA 12 package includes the CUDA runtime but still requires a compatible
host driver. See the
[JAX GPU installation guide](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).

### Unsupported GPU architecture

If `nvcc` reports an unsupported GPU architecture, upgrade the CUDA Toolkit or
build for an architecture supported by the installed toolkit:

```bash
rm -rf build/skbuild
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80" \
  TURBOMPC_CUDA=ON python -m pip install -e ".[cuda12]"
```

Find the CUDA architecture for your GPU with:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

The command prints a value such as `8.0` or `8.9`. Remove the decimal point
when passing it to CMake: use `80` for `8.0` or `89` for `8.9`.

### cuDSS is not found

The CUDA install normally provides cuDSS through the `nvidia-cudss-cu12`
package. Confirm it is installed, then rebuild:

```bash
python -m pip install "nvidia-cudss-cu12>=0.8"
TURBOMPC_CUDA=ON python -m pip install -e ".[cuda12]"
```

For a system installation, set `CUDSS_DIR` to the directory containing its
`include` and `lib` directories.

### A backend is reported as `NOT BUILT`

Rebuild with verbose output:

```bash
rm -rf build/skbuild
TURBOMPC_CUDA=ON python -m pip install -e ".[cuda12]" -v \
  2>&1 | tee /tmp/turbompc-build.log
```

Setting `TURBOMPC_CUDA=ON` turns a missing CUDA compiler into an installation
error instead of silently producing a CPU-only build.

If the Eigen backend is missing, install its headers and rebuild:

```bash
sudo apt install libeigen3-dev
TURBOMPC_EIGEN=ON python -m pip install -e ".[cuda12]"
```

### Unable to load cuSPARSE

When using TurboMPC on the GPU, `RuntimeError: Unable to load cuSPARSE. Is it installed?`
can be resolved by running:

```bash
unset LD_LIBRARY_PATH
```
`python -c "import jax; print(jax.devices())"` should then show a GPU device.

## See also

- [CUDA backends](cuda_backends.md) explains the available GPU backends.
- [`pyproject.toml`](../pyproject.toml) and [`CMakeLists.txt`](../CMakeLists.txt)
  define the build.
- `turbompc-check` validates the installed package and libraries.
