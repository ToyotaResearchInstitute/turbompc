#!/bin/bash
# Gradient accuracy sweep for TurboMPC.
#
# Evaluates cosine similarity between AD gradients and finite-difference
# gradients over 100 problem instances per solver tolerance.
#
# Nominal settings (matching run_sweeps_oat_turbompc.sh):
#   batch=64, horizon=40, n_state=8, n_ctrl=4, umax=1.0
#   fwd=admm_cudss_loop, bwd=direct_cudss_ffi, alpha=1.0
#   admm_max_iter=1000, sim_steps=50, use_full_hessian
#
# Tolerances swept: 1e-1, 1e-3, 1e-5, 1e-7, 1e-9

set -uo pipefail
unset JAX_PLATFORMS
export PYTORCH_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$SCRIPT_DIR/../../.venv/bin/python}"

# ---------- nominal settings ----------
BATCH=64
HORIZON=40
NSTATE=8
NCTRL=4
UMAX=1.0
ALPHA=1.0
ADMM_MAX_ITER=1000
SIM_STEPS=50
N_SEEDS=100
FD_EPS=1e-5

FWD_BACKEND="admm_fused_cudss"
BWD_BACKEND="direct_cudss_ffi"

echo "Nominal: batch=${BATCH}, horizon=${HORIZON}, nx=${NSTATE}, nu=${NCTRL}, umax=${UMAX}"
echo "Seeds per tolerance: ${N_SEEDS}"
echo ""

echo "======== TurboMPC Gradient Accuracy Sweep ========="

"$PY" "$SCRIPT_DIR/benchmark_turbompc_gradient_accuracy.py" \
    --batch_size      "$BATCH"          \
    --horizon         "$HORIZON"        \
    --sim_steps       "$SIM_STEPS"      \
    --n_state         "$NSTATE"         \
    --n_ctrl          "$NCTRL"          \
    --umax            "$UMAX"           \
    --alpha           "$ALPHA"          \
    --admm_max_iter   "$ADMM_MAX_ITER"  \
    --n_seeds         "$N_SEEDS"        \
    --fd_eps          "$FD_EPS"         \
    --fd_solver_tol   1e-9              \
    --fd_admm_max_iter 1000             \
    --tolerances 1e-1 1e-3 1e-5 1e-7 1e-9 \
    --linear_fwd_backend "$FWD_BACKEND" \
    --linear_bwd_backend "$BWD_BACKEND" \
    --use_full_hessian                  \
    --save_results

echo ""
echo "======== Gradient accuracy sweeps complete ========"
