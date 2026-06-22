#!/bin/bash
set -e

docker build -t turbompc-acados:latest -f Dockerfile .
echo "✓ acados Docker image built successfully"
