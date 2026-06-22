"""OAT (One-At-a-Time) sweep figure: forward/backward timing vs batch, horizon, dim.

Combines data from multiple result directories per solver (acados, TurboMPC, mpc.pytorch).

Usage:
    python3 plot_results/plot_fig_linear_system.py \
        --acados-root     <path/to/acados_results> \
        --turbompc-root    <path/to/turbompc_results/fwd=..._bwd=...> \
        --mpcpytorch-root <path/to/mpcpytorch_results> \
        --x-axes batch horizon dim --line-axis umax \
        --fixed-tol 1e-3 --fixed-admm 50 \
        --output-dir figs --save
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from load_results import load_dataframe
from plot_utils import (
    SOLVER_ORDER,
    SWEEP,
    annotate_speedup,
    configure_xaxis,
    draw_speedup_row,
    draw_sweep,
    finalize_figure,
    float_eq,
    legend_handles,
)

OAT_AXES = ("batch", "horizon", "dim")


def _load_multi(acados_roots, turbompc_roots, mpcpytorch_roots):
    dfs = []
    for ar in acados_roots or []:
        dfs.append(load_dataframe(results_root="__no_scan__", acados_root=ar))
    for dr in turbompc_roots or []:
        dfs.append(load_dataframe(results_root="__no_scan__", turbompc_root=dr))
    for mr in mpcpytorch_roots or []:
        dfs.append(load_dataframe(results_root="__no_scan__", mpcpytorch_root=mr))
    if not dfs:
        return pd.DataFrame()
    all_cols = list(dict.fromkeys(c for d in dfs for c in d.columns))
    dfs = [d.reindex(columns=all_cols) for d in dfs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return pd.concat(dfs, ignore_index=True)


def plot_figure(df, x_axes, line_axis, line_vals, solvers, oat_nominals, args):
    n_cols = len(x_axes)
    speedup_row = getattr(args, "speedup_row", False)
    n_rows = 3 if speedup_row else 2
    row_heights = [1.5, 1.5, 1.0] if speedup_row else [1.5, 1.5]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.16, sum(row_heights)),
        gridspec_kw={"height_ratios": row_heights},
        squeeze=False,
    )
    pass_row_label = {0: "Fwd.", 1: "Bwd."}

    oat_in_figure = [k for k in OAT_AXES if k in x_axes]

    for col, x_key in enumerate(x_axes):
        sdef = SWEEP[x_key]
        x_col = sdef["col"]

        col_fixed = {}
        if len(oat_in_figure) > 1 and x_key in oat_in_figure:
            for oat_key in oat_in_figure:
                if oat_key != x_key:
                    nom = oat_nominals.get(oat_key)
                    if nom is not None:
                        col_fixed[SWEEP[oat_key]["col"]] = nom

        col_df = df.copy()
        for fc, fv in col_fixed.items():
            col_df = col_df[col_df[fc] == fv]
        xvals = sorted(col_df[x_col].dropna().unique())

        for row, pass_type in enumerate(("fwd", "bwd")):
            ax = axes[row, col]
            draw_sweep(
                ax,
                df,
                x_col,
                pass_type,
                solvers,
                line_axis,
                line_vals,
                col_fixed=col_fixed,
            )
            if getattr(args, "speedup", False):
                annotate_speedup(
                    ax,
                    df,
                    x_col,
                    pass_type,
                    solvers,
                    line_axis,
                    line_vals,
                    col_fixed=col_fixed,
                )
            ax.set_yscale("log")
            bottom = (row == 1) and not speedup_row
            configure_xaxis(ax, x_col, xvals, bottom_row=bottom, sdef=sdef)
            if col == 0:
                ax.set_ylabel(f"{pass_row_label[row]} Time (s)", fontsize=7)

        if speedup_row:
            ax = axes[2, col]
            draw_speedup_row(
                ax, df, x_col, solvers, line_axis, line_vals, col_fixed=col_fixed
            )
            configure_xaxis(ax, x_col, xvals, bottom_row=True, sdef=sdef)
            if col == 0:
                ax.set_ylabel("Speedup\nvs TurboMPC", fontsize=7)

    axes_tag = "_".join(x_axes)
    line_tag = f"_line={line_axis}" if line_axis else ""
    args._output_stem = f"fig3b_{axes_tag}{line_tag}"
    finalize_figure(fig, axes, legend_handles(solvers, line_axis, line_vals), args)


def main():
    parser = argparse.ArgumentParser(
        description="OAT sweep figure: acados / TurboMPC / mpc.pytorch (multi-subset)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data sources — each accepts multiple paths for multi-subset combining
    parser.add_argument(
        "--acados-root",
        nargs="+",
        default=None,
        help="Dir(s) containing acados_* result folders",
    )
    parser.add_argument(
        "--turbompc-root",
        nargs="+",
        default=None,
        help=(
            "Dir(s) containing turbompc_* result folders "
            "(the fwd=.../bwd=... dirs themselves)"
        ),
    )
    parser.add_argument(
        "--mpcpytorch-root",
        nargs="+",
        default=None,
        help="Dir(s) containing mpcpytorch_* result folders",
    )

    # Sweep axes
    parser.add_argument(
        "--x-axes",
        nargs="+",
        default=["batch", "horizon", "dim"],
        choices=list(SWEEP),
        help="OAT axis per column (default: batch horizon dim)",
    )
    parser.add_argument(
        "--line-axis",
        default=None,
        choices=list(SWEEP),
        help="Secondary dimension as linestyle (auto-detected if omitted)",
    )

    # OAT nominals (used to fix non-sweep columns)
    parser.add_argument(
        "--nom-batch",
        type=int,
        default=64,
        help="Nominal batch size for OAT cross-column pinning (default: 64)",
    )
    parser.add_argument(
        "--horizon", type=int, default=40, help="Nominal / fixed horizon (default: 40)"
    )
    parser.add_argument(
        "--dim", type=int, default=12, help="Nominal / fixed dimension (default: 12)"
    )

    # Fixed filters for non-OAT axes
    parser.add_argument("--fixed-umax", type=float, default=None)
    parser.add_argument("--fixed-tol", type=float, default=None)
    parser.add_argument(
        "--fixed-admm",
        type=int,
        default=None,
        help="Fix TurboMPC admm_max_iter; NaN rows (other solvers) pass through",
    )

    # Explicit value lists for line-axis
    parser.add_argument("--umaxes", type=float, nargs="+", default=None)
    parser.add_argument("--tols", type=float, nargs="+", default=None)

    # Output
    parser.add_argument("--output-dir", default="figs")
    parser.add_argument("--save", action="store_true")
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="Annotate the last x-point with speedup ratio vs TurboMPC",
    )
    parser.add_argument(
        "--speedup-row",
        action="store_true",
        help="Add a 3rd row showing time_other/time_turbompc ratio",
    )

    args = parser.parse_args()

    df = _load_multi(args.acados_root, args.turbompc_root, args.mpcpytorch_root)
    if df.empty:
        print(
            "No data. Provide at least one of: --acados-root, --turbompc-root, "
            "--mpcpytorch-root."
        )
        sys.exit(1)

    # Resolve line axis
    line_axis = args.line_axis
    if line_axis is None:
        for candidate in ("umax", "tol"):
            if (
                candidate not in args.x_axes
                and getattr(args, f"fixed_{candidate}", None) is None
            ):
                line_axis = candidate
                break

    x_cols = {SWEEP[k]["col"] for k in args.x_axes}
    line_col = SWEEP[line_axis]["col"] if line_axis else None

    def _is_free(key):
        return key in args.x_axes or key == line_axis

    # Global fixed filters (OAT dims handled per-column inside plot_figure)
    if not _is_free("umax") and args.fixed_umax is not None:
        df = df[df["umax"] == args.fixed_umax]
    if not _is_free("tol") and args.fixed_tol is not None:
        df = df[float_eq(df["tol"], args.fixed_tol, "tol")]
    if (
        not _is_free("admm")
        and args.fixed_admm is not None
        and "admm_max_iter" in df.columns
    ):
        df = df[df["admm_max_iter"].isna() | (df["admm_max_iter"] == args.fixed_admm)]
    # For non-OAT x-axes that are fixed (e.g. horizon when x_axes=[batch, dim])
    if (
        "horizon" not in x_cols
        and line_col != "horizon"
        and "horizon" not in [k for k in OAT_AXES if k in args.x_axes]
    ):
        if args.horizon is not None:
            df = df[df["horizon"] == args.horizon]
    if (
        "dimensions" not in x_cols
        and line_col != "dimensions"
        and "dim" not in [k for k in OAT_AXES if k in args.x_axes]
    ):
        if args.dim is not None:
            df = df[df["dimensions"] == args.dim]

    if df.empty:
        print("No data after filtering. Check parameter values and data roots.")
        sys.exit(1)

    # Line-axis values
    if line_axis is None:
        line_vals = []
    elif line_axis == "tol":
        line_vals = sorted(args.tols or df["tol"].dropna().unique())
    elif line_axis == "umax":
        line_vals = sorted(args.umaxes or df["umax"].dropna().unique())
    else:
        line_vals = sorted(df[line_col].dropna().unique())

    oat_nominals = {
        "batch": args.nom_batch,
        "horizon": args.horizon,
        "dim": args.dim,
    }

    present = set(df["solver"].unique())
    solvers = [s for s in SOLVER_ORDER if s in present]
    plot_figure(df, args.x_axes, line_axis, line_vals, solvers, oat_nominals, args)


if __name__ == "__main__":
    main()
