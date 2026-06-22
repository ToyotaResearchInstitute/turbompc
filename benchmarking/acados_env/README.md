# acados Environment for Benchmarking

This directory contains a Docker environment for running acados benchmarks against turbompc.

## Setup

### 1. Build the Docker image

```bash
cd benchmarking/acados_env
./build.sh
```

This will:
- Clone acados v0.5.3 inside the container
- Build acados with QPOASES support
- Install Python interface
- Set up CUDA 12.8 environment

### 2. Download Zenodo benchmark code (optional)

If you want to reproduce results from the paper:

1. Download from: https://zenodo.org/records/17832359
2. Extract to `benchmarking/acados-benchmark/`

### 3. Run the environment

```bash
./run.sh
```

This launches an interactive shell with:
- acados v0.5.3 installed
- GPU support enabled
- Access to your workspace at `/workspace`

## Usage

Inside the container:
```bash
# Run your benchmark scripts
python3 benchmark_acados.py

# Or run the Zenodo reproduction code
cd /workspace/benchmarking/acados-benchmark
python3 their_script.py
```

## Notes

- acados is built inside the container at `/opt/acados`
- The container isolates acados dependencies from your host system
- No need to add acados as a git submodule
