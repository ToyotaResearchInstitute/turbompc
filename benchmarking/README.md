# Benchmarking

## Installation
Separate environments are used for external solvers to avoid dependency
conflicts. Installation instructions for each supported solver are below.

### turbompc
From the repo root, run the standard install (GPU):
```bash
make install
source .venv/bin/activate
```

For a CPU-only machine (no GPU required):
```bash
make install-cpu
source .venv/bin/activate
```

### mpc.pytorch
**GPU machine:**
```bash
git clone https://github.com/diffmpc/mpc.pytorch.git benchmarking/mpc.pytorch
python -m venv .mpcpt-venv && source .mpcpt-venv/bin/activate
pip install torch
cd benchmarking/mpc.pytorch
python -m pip install -e .
cd ../..
```

**CPU-only machine** (installs a lightweight CPU-only PyTorch, ~250 MB):
```bash
git clone https://github.com/diffmpc/mpc.pytorch.git benchmarking/mpc.pytorch
python -m venv .mpcpt-venv && source .mpcpt-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
cd benchmarking/mpc.pytorch
python -m pip install -e .
cd ../..
```

After installation, import the module in Python as:
```python
import mpc
```

### acados
The acados benchmark environment uses a Docker container. This requires Docker
and the NVIDIA Container Toolkit.

```bash
cd benchmarking/acados_env
./build.sh
./run.sh
```

## Running Benchmarks
Linear-system benchmark scripts are in `benchmarking/linear-system`.

Activate the environment for the solver being evaluated, then run the
corresponding benchmark script.

### turbompc
```bash
source .venv/bin/activate
python benchmarking/linear-system/benchmark_turbompc_constrained.py
```

### mpc.pytorch
```bash
source .mpcpt-venv/bin/activate
python benchmarking/linear-system/benchmark_mpcpytorch_constrained.py
```

### acados
Run this inside the acados container started from `benchmarking/acados_env/run.sh`:

```bash
python benchmarking/linear-system/benchmark_acados_constrained.py
```
