#!/bin/bash
set -e

# Run the acados environment
docker run --rm -it \
    -v "$(pwd)/../..:/workspace" \
    -w /workspace/benchmarking/linear_system \
    turbompc-acados:latest \
    bash
