#!/bin/bash
set -e

# Run the acados environment
docker run --rm -it \
    -v "$(pwd)/../..:/workspace" \
    -w /workspace/benchmarking/linear-system \
    turbompc-acados:latest \
    bash
