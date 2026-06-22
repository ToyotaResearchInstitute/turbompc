# Build Guide

Install of turbompc (venv + Python deps + CUDA FFI kernels +
post-install check).

## Prerequisites

| Tool | Version | How to check |
|---|---|---|
| Python | 3.10+ | `python3 --version` |
| CUDA Toolkit | 12.x | `nvcc --version` |
| CMake | 3.20+ | `cmake --version` |
| NVIDIA driver | supports your CUDA Toolkit | `nvidia-smi` |
| C++ compiler | supports C++17 (GCC 9+, Clang 10+) | `g++ --version` |


## Fresh Install

```bash
git clone git@github.com:ToyotaResearchInstitute/turbompc.git
cd turbompc
make install
source .venv/bin/activate
```

Expected `make check` output on a fully-built install:

```
Python: /.../.venv/bin/python (3.13)
JAX: 0.10.x, devices: [CudaDevice(id=0)]
cuSolver: OK
turbompc: OK
fused_pcg FFI: OK
fused_cudss FFI: OK
pcg FFI: OK
cudss FFI: OK
cudss KKT FFI: OK

All checks passed
```

### Install with an existing venv

The build defaults to an in-repo `.venv/`. To use a different one: e.g.
a shared workspace venv: pass `VENV` (relative or absolute) to any
target:

```bash
make install VENV=/path/to/shared/.venv   # or: make cuda check VENV=../.venv
```

### Details

`make install` runs four targets in order:

1. `make venv`: creates `.venv/` via `python3 -m venv`.
2. `make pip`: upgrades pip, installs `jax[cuda12]`, `nvidia-cudss-cu12`,
   and the project itself in editable mode with the `dev` extra.
   Verifies JAX can see the GPU.
3. `make cuda`: runs CMake against `turbompc/solvers/csrc/`, building
   the five FFI shared libraries (see
   [cuda-ffi-backends.md](cuda-ffi-backends.md) for the list).
4. `make check`: runs `check_install.py`, which imports each
   FFI wrapper and calls `_find_lib()` to confirm the `.so` is loadable.

Summary of commands:

| Target | What it does | When to use |
|---|---|---|
| `make venv` | Create `.venv/` if missing | First time, or after `rm -rf .venv/` |
| `make pip` | Install / reinstall Python deps | After editing `pyproject.toml` or pulling new deps |
| `make cuda` | Compile FFI libs via CMake | After editing `.cu` / `.cc` / `.cuh` files |
| `make check` | Verify install end-to-end | After `make cuda`, or to debug a stale install |
| `make torch-env` | Create isolated Torch CUDA env in `venv-torch/` | When you need PyTorch without changing JAX CUDA deps |
| `make test` | Default test suite | CI / regression |
| `make test-extended` | Default + extended tests | Broad coverage |
| `make clean` | Remove `build/ffi*` | Force a full CUDA rebuild |

## Separate PyTorch environment

If you use both JAX CUDA wheels and PyTorch CUDA wheels, keep them in
separate virtual environments to avoid pip resolver conflicts on
`nvidia-*` packages.
Create a dedicated Torch env:

```bash
make torch-env
source venv-torch/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`make torch-env` installs `torch==2.10.0` from the CUDA 12.8 wheel index
in `venv-torch/`, leaving the main turbompc JAX env untouched.

## Manual build (without `make`)

If you need finer control, reproduce what `make install` does step by
step:

```bash
# 1. venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Python deps
pip install --upgrade pip
pip install "jax[cuda12]"
pip install nvidia-cudss-cu12
pip install -e ".[dev]"

# 3. Verify JAX + GPU
python -c "import jax; assert any(d.platform in ('gpu','cuda') for d in jax.devices())"

# 4. CUDA FFI kernels
cmake -S turbompc/solvers/csrc -B build/ffi -DCMAKE_BUILD_TYPE=Release \
      -DPython3_EXECUTABLE=$(pwd)/.venv/bin/python \
      -DPython3_FIND_VIRTUALENV=ONLY
cmake --build build/ffi -j

# 5. Verify FFI imports
python check_install.py
```

### GPU architecture selection

CMake auto-detects the GPU via `nvidia-smi --query-gpu=compute_cap`
(on cmake < 3.24 it falls back to `native`, and if that fails, to
`75;80;86;89;90;120`: a broad fat-binary). To cross-compile for a
specific arch, pass `-DCMAKE_CUDA_ARCHITECTURES="89"` (Ada) or
`"80;90"` (Ampere + Ada) to the first `cmake -S ... -B ...` invocation.

Verify what CMake actually used:
```bash
grep CMAKE_CUDA_ARCHITECTURES build/ffi/CMakeCache.txt
```

## Troubleshooting



### `undefined symbol` error
Run
```
make clean && make cuda
```
and do a full rebuild.


### `make pip` fails with `No GPU found!`

JAX's CUDA 12 wheel ships its own CUDA runtime, but it needs a matching
NVIDIA driver on the host. Check:
```bash
nvidia-smi        # driver version must support CUDA 12
python -c "import jax; print(jax.devices())"
```
If `jax.devices()` shows only CPU, see the
[JAX GPU install guide](https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu).

### `make cuda` fails with `nvcc fatal : Unsupported gpu architecture`

Your CUDA Toolkit is older than the auto-detected architecture target.
Two options:
- Upgrade the CUDA Toolkit to one that supports your GPU's compute
  capability.
- Override with an older arch: `make clean && CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80" make cuda`
  (replace `80` with whichever arch your toolkit supports).

### `make check` shows `NOT BUILT` for some FFI backends

One or more `.so` files didn't compile. Re-run `make cuda` with
verbose output:
```bash
make clean
cmake --build build/ffi -j --verbose 2>&1 | tee /tmp/cuda_build.log
```
Scan for `error:` in the log. Common causes:
- Missing cuDSS headers: `pip install nvidia-cudss-cu12` didn't run.
- `cudss.h: No such file`: the cuDSS Python wheel's include dir isn't
  on the CMake search path. CMake picks it up from the venv's
  `site-packages/nvidia/cu12/include/`; confirm the files are there.

### `NOT_FOUND: No FFI handler registered for ...`

The Python wrapper couldn't find its `.so` at import time. Verify:
```bash
ls -lh build/ffi/lib*.so
python -c "from turbompc.solvers.admm.admm_ffi_backend import _find_lib; print(_find_lib())"
```
FFI libraries are machine-specific (compiled for a specific CUDA
version + GPU arch). Moving a `build/` directory between machines or
changing GPUs requires `make clean && make cuda`.

### Old shared libraries after branch switch

Git never tracks `build/`. Switching between branches that changed
`.cu` files can leave you with a stale `build/ffi/` that doesn't match
the current sources. Run
```bash
make clean && make cuda
```
and rebuild.


## See also

- [cuda-ffi-backends.md](cuda-ffi-backends.md): what each backend does and how to pick one.
- [`Makefile`](../Makefile): the install definition.
- [`check_install.py`](../check_install.py): for validation.
