#!/bin/bash
# OAT (One-At-a-Time) sweeps for Acados constrained benchmark.
#
# Same three sweeps as run_sweeps_oat_turbompc.sh for direct comparison:
#   Nominal: batch=64, horizon=40, n_state=8, n_ctrl=4, umax=1
#
# Sweep A: batch size  (horizon=40, dim=8/4)
# Sweep B: horizon     (batch=64,   dim=8/4)
# Sweep C: dimension   (batch=64,   horizon=40)
#
# Each sweep is run at umax in {1, 10}.

set -uo pipefail


FAILED_RUNS=()

# ---------- shared settings ----------
NUM_REPEATS=10
SIM_STEPS=50

# ---------- nominal ----------
NOM_BATCH=64
NOM_HORIZON=40
NOM_NSTATE=8
NOM_NCTRL=4

UMAXES=(1.0 10.0)
TOLERANCES=(1e-3 1e-5 1e-7)

cleanup_caches() {
    python3 << 'EOF'
import gc
gc.collect()
print("Caches cleared")
EOF
}

run_acados() {
    local batch="$1" horizon="$2" nstate="$3" nctrl="$4" umax="$5" tol="$6"
    echo ">>> Acados  batch=$batch  horizon=$horizon  dim=${nstate}+${nctrl}  umax=$umax  tol=$tol"
    if python3 "benchmark_acados_constrained.py" \
        --batch_size  "$batch"   \
        --horizon     "$horizon" \
        --n_state     "$nstate"  \
        --n_ctrl      "$nctrl"   \
        --sim_steps   "$SIM_STEPS" \
        --num_repeats "$NUM_REPEATS" \
        --umax        "$umax"    \
        --tol         "$tol"     \
        --save_results; then
        echo "    OK"
    else
        echo "    FAILED — batch=$batch horizon=$horizon dim=${nstate}+${nctrl} umax=$umax tol=$tol" >&2
        FAILED_RUNS+=("acados:batch=${batch},horizon=${horizon},dim=${nstate}+${nctrl},umax=${umax},tol=${tol}")
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
        for BS in "${BATCH_SIZES[@]}"; do
            run_acados "$BS" "$NOM_HORIZON" "$NOM_NSTATE" "$NOM_NCTRL" "$UMAX" "$TOL"
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
        for H in "${HORIZONS[@]}"; do
            run_acados "$NOM_BATCH" "$H" "$NOM_NSTATE" "$NOM_NCTRL" "$UMAX" "$TOL"
        done
    done
done

# ============================================================
# Sweep C — dimension (n_state=2*n_ctrl assumed)
# ============================================================
echo "======== Sweep C: dimension ========"
DIM_PAIRS=("4 2" "8 4" "16 8" "32 16" "64 32")
for UMAX in "${UMAXES[@]}"; do
    for TOL in "${TOLERANCES[@]}"; do
        for PAIR in "${DIM_PAIRS[@]}"; do
            read -r NS NC <<< "$PAIR"
            run_acados "$NOM_BATCH" "$NOM_HORIZON" "$NS" "$NC" "$UMAX" "$TOL"
        done
    done
done

echo "======== All Acados OAT sweeps complete ========"

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: ${#FAILED_RUNS[@]} run(s) failed:"
    for r in "${FAILED_RUNS[@]}"; do echo "  $r"; done
    exit 1
fi
