"""Compare TurboMPC and acados d(cost)/d(Q_diag, R_diag) gradients.

Usage (by parameters — recommended):
    python compare_gradients.py --batch_size 1 --horizon 40 --sim_steps 50 \
      --n_state 8 --n_ctrl 4 --umax 10.0 [--atol 0.01]

Usage (by explicit file paths):
  python compare_gradients.py --turbompc_grad_Q <f> --turbompc_grad_R <f> \
      --acados_grad_Q <f> --acados_grad_R <f> [--atol 0.01]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

# Allow imports from parent directory (utils, benchmark_naming, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from benchmark_naming import acados_dirname, turbompc_dirname


def compare(name: str, a: np.ndarray, b: np.ndarray, atol: float, rtol: float):
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    print(f"\n--- {name} ---")
    print(f"  TurboMPC shape : {a.shape}  acados shape: {b.shape}")
    if a.shape != b.shape:
        print("  SHAPE MISMATCH - cannot compare values")
        return

    abs_diff = np.abs(a - b)
    rel_diff = abs_diff / (np.abs(b) + 1e-12)
    max_abs = abs_diff.max()
    max_rel = rel_diff.max()
    passed = np.allclose(a, b, atol=atol, rtol=rtol)

    print(f"  TurboMPC : {a}")
    print(f"  acados  : {b}")
    print(f"  abs_diff: {abs_diff}")
    print(f"  rel_diff: {rel_diff}")
    print(f"  max |diff| = {max_abs:.3e}   max rel = {max_rel:.3e}")
    print(f"  allclose(atol={atol}, rtol={rtol}): {'PASS' if passed else 'FAIL'}")


def _collapse_repeated_gradients(arr: np.ndarray, num_repeats: int) -> np.ndarray:
    """Average over a leading repeat axis when gradients are saved per repeat."""
    arr = np.asarray(arr)
    if arr.ndim > 1 and arr.shape[0] == num_repeats:
        return arr.mean(axis=0)
    return arr


def _find_grad_files(
    results_dir: str, dirname: str, prefix: str, backend_subdir: str | None = None
):
    """Return (grad_Q_path, grad_R_path) by searching for matching files."""
    def _search(base_pattern: str):
        for base in sorted(glob.glob(base_pattern)):
            # Device name varies (e.g. TFRT_CPU_0, cpu) — glob for it.
            q_matches = sorted(glob.glob(os.path.join(base, f"{prefix}_*_grad_Q.npy")))
            r_matches = sorted(glob.glob(os.path.join(base, f"{prefix}_*_grad_R.npy")))
            if q_matches and r_matches:
                return q_matches[0], r_matches[0]
        return None

    exact_base = (
        os.path.join(results_dir, backend_subdir, dirname)
        if backend_subdir
        else os.path.join(results_dir, dirname)
    )
    exact = _search(exact_base)
    if exact:
        return exact

    # If the requested sim_steps directory is absent, fall back to any matching
    # steps=* directory so existing benchmark outputs can still be compared.
    wildcard_base = None
    if "_steps=" in dirname:
        dirname_prefix = dirname.rsplit("_steps=", 1)[0]
        wildcard_base = (
            os.path.join(results_dir, backend_subdir, f"{dirname_prefix}_steps=*")
            if backend_subdir
            else os.path.join(results_dir, f"{dirname_prefix}_steps=*")
        )
        fallback = _search(wildcard_base)
        if fallback:
            return fallback

    searched = [exact_base]
    if wildcard_base is not None:
        searched.append(wildcard_base)
    raise FileNotFoundError(
        "No gradient files found. Searched: " + ", ".join(searched)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare TurboMPC vs acados gradients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- shared parameters (used to auto-construct paths) ---
    params = parser.add_argument_group("benchmark parameters (auto-construct paths)")
    params.add_argument("--batch_size", type=int, default=1)
    params.add_argument("--horizon", type=int, default=40)
    params.add_argument("--sim_steps", type=int, default=50)
    params.add_argument("--n_state", type=int, default=8)
    params.add_argument("--n_ctrl", type=int, default=4)
    params.add_argument("--num_repeats", type=int, default=1)
    params.add_argument("--umax", type=float, default=10.0)
    params.add_argument("--results_dir", type=str, default="timing_results")

    # --- turbompc-specific naming ---
    dm = parser.add_argument_group("turbompc naming (used with auto-construct)")
    dm.add_argument("--pcg_eps", type=float, default=1e-12)
    dm.add_argument("--alpha", type=float, default=1.6)
    dm.add_argument("--warm_start", action="store_true", default=True)
    dm.add_argument(
        "--tol", type=float, default=1e-7, help="TurboMPC solver tolerance (SQP + ADMM)"
    )
    dm.add_argument("--admm_max_iter", type=int, default=100)
    dm.add_argument("--linear_fwd_backend", type=str, default="admm_jax_loop_pcg")
    dm.add_argument("--linear_bwd_backend", type=str, default="admm_jax_loop_pcg")

    # --- acados-specific naming ---
    ac = parser.add_argument_group("acados naming (used with auto-construct)")
    ac.add_argument(
        "--acados_tol",
        type=float,
        default=None,
        help="Acados solver tolerance (defaults to --tol)",
    )

    # --- explicit file paths (override auto-construct) ---
    explicit = parser.add_argument_group(
        "explicit file paths (override auto-construct)"
    )
    explicit.add_argument("--turbompc_grad_Q", type=str, default=None)
    explicit.add_argument("--turbompc_grad_R", type=str, default=None)
    explicit.add_argument("--acados_grad_Q", type=str, default=None)
    explicit.add_argument("--acados_grad_R", type=str, default=None)

    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    # Resolve paths
    if (
        args.turbompc_grad_Q
        and args.turbompc_grad_R
        and args.acados_grad_Q
        and args.acados_grad_R
    ):
        dQ_path, dR_path = args.turbompc_grad_Q, args.turbompc_grad_R
        aQ_path, aR_path = args.acados_grad_Q, args.acados_grad_R
    else:
        dims = args.n_state + args.n_ctrl
        dm_dir = turbompc_dirname(
            args.batch_size,
            args.horizon,
            dims,
            args.num_repeats,
            constrained=True,
            warm_start=args.warm_start,
            pcg_eps=args.pcg_eps,
            alpha=args.alpha,
            umax=args.umax,
            tol=args.tol,
            admm_max_iter=args.admm_max_iter,
            sim_steps=args.sim_steps,
        )
        ac_dir = acados_dirname(
            args.batch_size,
            args.horizon,
            dims,
            args.num_repeats,
            constrained=True,
            umax=args.umax,
            tol=args.tol if args.acados_tol is None else args.acados_tol,
            sim_steps=args.sim_steps,
        )
        backend_subdir = f"fwd={args.linear_fwd_backend}_bwd={args.linear_bwd_backend}"

        dQ_path, dR_path = _find_grad_files(
            args.results_dir, dm_dir, "turbompc", backend_subdir
        )
        aQ_path, aR_path = _find_grad_files(args.results_dir, ac_dir, "acados")

    dQ = np.load(dQ_path)
    dR = np.load(dR_path)
    aQ = np.load(aQ_path)
    aR = np.load(aR_path)

    dQ = _collapse_repeated_gradients(dQ, args.num_repeats)
    dR = _collapse_repeated_gradients(dR, args.num_repeats)
    aQ = _collapse_repeated_gradients(aQ, args.num_repeats)
    aR = _collapse_repeated_gradients(aR, args.num_repeats)

    print("Loaded files:")
    print(f"  turbompc_grad_Q: {dQ_path}  shape={dQ.shape}")
    print(f"  turbompc_grad_R: {dR_path}  shape={dR.shape}")
    print(f"  acados_grad_Q : {aQ_path}   shape={aQ.shape}")
    print(f"  acados_grad_R : {aR_path}   shape={aR.shape}")

    compare(
        "d(reward)/d(Q_diag)  [shape (nx,)]",
        dQ.flatten(),
        aQ.flatten(),
        args.atol,
        args.rtol,
    )
    compare(
        "d(reward)/d(R_diag)  [shape (nu,)]",
        dR.flatten(),
        aR.flatten(),
        args.atol,
        args.rtol,
    )


if __name__ == "__main__":
    main()
