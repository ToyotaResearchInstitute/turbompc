#!/bin/bash
export PYTHONPATH="/opt/differentiable_nmpc:${PYTHONPATH}"
exec "$@"
