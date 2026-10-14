#!/bin/bash
# OAT (One-At-a-Time) sweeps for TurboMPC constrained benchmark.
#
# Three sweeps, each varying one axis while the others are fixed at the nominal:
#   Nominal: batch=64, horizon=40, n_state=8, n_ctrl=4, umax=1
#
# Sweep A: batch size  (horizon=40, dim=8/4)
# Sweep B: horizon     (batch=64,   dim=8/4)
# Sweep C: dimension   (batch=64,   horizon=40)
#
# Each sweep is run at umax in {1, 10}.

set -uo pipefail
unset JAX_PLATFORMS
export PYTORCH_ALLOC_CONF=expandable_segments:True


FAILED_RUNS=()

# Use the repo's .venv so jax/jaxlib match the FFI .so. Bare `python3` resolves
# to conda's python with a different jax version, which trips XLA compiler
# bugs at batch=1 + --use_full_hessian.
PY="${PY:-$(cd "$(dirname "$0")/../.." && pwd)/.venv/bin/python}"

# ---------- shared settings ----------
FWD_BACKEND="admm_fused_cudss"
BWD_BACKEND="direct_cudss_ffi"
NUM_REPEATS=10
ALPHA=1.0
SIM_STEPS=50

# ---------- nominal ----------
NOM_BATCH=64
NOM_HORIZON=40
NOM_NSTATE=8
NOM_NCTRL=4

UMAXES=(1.0 10.0)
TOLERANCES=(1e-3 1e-5 1e-7)
ADMM_ITERS=(50 1000)

cleanup_caches() {
    "$PY" << 'EOF'
import gc, shutil, os
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
except ImportError:
    pass
jax_cache = os.path.expanduser("~/.cache/jax")
if os.path.exists(jax_cache):
    shutil.rmtree(jax_cache); os.makedirs(jax_cache, exist_ok=True)
gc.collect()
print("Caches cleared")
EOF
}

run_turbompc() {
    local batch="$1" horizon="$2" nstate="$3" nctrl="$4" umax="$5" tol="$6" admm="$7"
    echo ">>> TurboMPC  batch=$batch  horizon=$horizon  dim=${nstate}+${nctrl}  umax=$umax  tol=$tol  admm=$admm"
    if "$PY" "benchmark_turbompc_constrained.py" \
        --batch_size   "$batch"   \
        --horizon      "$horizon" \
        --n_state      "$nstate"  \
        --n_ctrl       "$nctrl"   \
        --num_repeats  "$NUM_REPEATS" \
        --umax         "$umax"    \
        --tol     "$tol"     \
        --admm_max_iter "$admm"   \
        --cold_start           \
        --alpha        "$ALPHA"   \
        --linear_fwd_backend "$FWD_BACKEND" \
        --linear_bwd_backend "$BWD_BACKEND" \
        --use_full_hessian \
        --save_results; then
        echo "    OK"
    else
        echo "    FAILED — batch=$batch horizon=$horizon dim=${nstate}+${nctrl} umax=$umax tol=$tol admm=$admm" >&2
        FAILED_RUNS+=("turbompc:batch=${batch},horizon=${horizon},dim=${nstate}+${nctrl},umax=${umax},tol=${tol},admm=${admm}")
    fi
    cleanup_caches
    read -t 20 -p "Done. Continuing in 20 s (Enter to skip)..." 2>/dev/null || true
}

# ============================================================
# Sweep A — batch size
# ============================================================
echo "======== Sweep A: batch size ========"
BATCH_SIZES=(1 8 16 32 64 128 256 512 1024)
for UMAX in "${UMAXES[@]}"; do
    for TOL in "${TOLERANCES[@]}"; do
        for ADMM in "${ADMM_ITERS[@]}"; do
            for BS in "${BATCH_SIZES[@]}"; do
                run_turbompc "$BS" "$NOM_HORIZON" "$NOM_NSTATE" "$NOM_NCTRL" "$UMAX" "$TOL" "$ADMM"
            done
        done
    done
done

# ============================================================
# Sweep B — horizon
# ============================================================
echo "======== Sweep B: horizon ========"
HORIZONS=(10 20 40 80 160 320)
for UMAX in "${UMAXES[@]}"; do
    for TOL in "${TOLERANCES[@]}"; do
        for ADMM in "${ADMM_ITERS[@]}"; do
            for H in "${HORIZONS[@]}"; do
                run_turbompc "$NOM_BATCH" "$H" "$NOM_NSTATE" "$NOM_NCTRL" "$UMAX" "$TOL" "$ADMM"
            done
        done
    done
done

# ============================================================
# Sweep C — dimension (n_state=2*n_ctrl assumed)
# ============================================================
echo "======== Sweep C: dimension ========"
# (n_state, n_ctrl) pairs
DIM_PAIRS=("4 2" "8 4" "16 8" "32 16" "64 32")
for UMAX in "${UMAXES[@]}"; do
    for TOL in "${TOLERANCES[@]}"; do
        for ADMM in "${ADMM_ITERS[@]}"; do
            for PAIR in "${DIM_PAIRS[@]}"; do
                read -r NS NC <<< "$PAIR"
                run_turbompc "$NOM_BATCH" "$NOM_HORIZON" "$NS" "$NC" "$UMAX" "$TOL" "$ADMM"
            done
        done
    done
done

echo "======== All TurboMPC OAT sweeps complete ========"

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: ${#FAILED_RUNS[@]} run(s) failed:"
    for r in "${FAILED_RUNS[@]}"; do echo "  $r"; done
    exit 1
fi
