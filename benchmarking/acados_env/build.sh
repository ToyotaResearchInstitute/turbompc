#!/bin/bash
set -euo pipefail

docker build \
  --build-arg ACADOS_REF=v0.6.0 \
  -t turbompc-acados:v0.6.0 \
  -t turbompc-acados:latest \
  -f Dockerfile \
  .

echo "✓ acados Docker image built successfully"
