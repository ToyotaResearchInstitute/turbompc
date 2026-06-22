"""Shared visual constants and drawing helpers for benchmark figures."""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.labelpad": 2.0,
        }
    )


SOLVER_COLORS = {
    "acados": "#c0392b",  # red
    "mpcpytorch": "#1a9850",  # green
    "turbompc": "#2166ac",  # blue
}
SOLVER_MARKERS = {"acados": "s", "mpcpytorch": "^", "turbompc": "o"}
SOLVER_ORDER = ["acados", "mpcpytorch", "turbompc"]
SOLVER_DISPLAY = {
    "acados": "acados",
    "mpcpytorch": "mpc.pytorch",
    "turbompc": "TurboMPC",
}

LINE_LS = ["-", "--", ":"]
LINE_ALPHA = [1.0, 0.70, 0.50]

SWEEP: dict[str, dict] = {
    "batch": dict(col="batch_size", label="Batch size", log_x=True, base2=True),
    "horizon": dict(col="horizon", label="Horizon", log_x=True, base2=False),
    "dim": dict(col="dimensions", label="State+ctrl dim", log_x=True, base2=False),
    "umax": dict(col="umax", label="$u_{\\max}$", log_x=True, base2=False),
    "tol": dict(col="tol", label="Tolerance", log_x=True, base2=False),
    "admm": dict(col="admm_max_iter", label="ADMM max iter", log_x=True, base2=False),
}


def stats_group(df_sub: pd.DataFrame, x_col: str) -> pd.DataFrame:
    rows = []
    for xv, g in df_sub.groupby(x_col):
        a = g["time_s"].values
        rows.append(
            {
                x_col: xv,
                "mean": np.mean(a),
                "p25": np.percentile(a, 25),
                "p75": np.percentile(a, 75),
            }
        )
    if not rows:
        return pd.DataFrame(columns=[x_col, "mean", "p25", "p75"])
    out = pd.DataFrame(rows).sort_values(x_col).reset_index(drop=True)
    out[x_col] = out[x_col].astype(float)
    return out


def float_eq(series: pd.Series, val: float, axis_key: str) -> pd.Series:
    if axis_key == "tol":
        return np.isclose(series.astype(float), val, rtol=1e-3)
    return series == val


def fmt_val(axis_key: str, val) -> str:
    if axis_key == "tol":
        return f"tol={val:.0e}"
    if axis_key == "umax":
        return f"$u_{{\\max}}$={int(val) if val == int(val) else val}"
    if axis_key == "batch":
        return f"batch={int(val)}"
    if axis_key == "admm":
        return f"admm={int(val)}"
    return f"{axis_key}={val}"


def draw_sweep(
    ax,
    df: pd.DataFrame,
    x_col: str,
    pass_type: str,
    solvers: list[str],
    line_axis: str | None,
    line_vals: list,
    col_fixed: dict | None = None,
) -> list:
    line_col = SWEEP[line_axis]["col"] if line_axis else None
    sub = df[df["pass_type"] == pass_type].copy()
    if col_fixed:
        for fc, fv in col_fixed.items():
            sub = sub[sub[fc] == fv]

    handles = []
    for solver in solvers:
        color = SOLVER_COLORS.get(solver, "grey")
        marker = SOLVER_MARKERS.get(solver, "o")
        sv = sub[sub["solver"] == solver]

        if line_col is None:
            if sv.empty:
                continue
            st = stats_group(sv, x_col)
            lbl = SOLVER_DISPLAY.get(solver, solver)
            (h,) = ax.plot(
                st[x_col],
                st["mean"],
                color=color,
                linestyle="-",
                linewidth=1.8,
                marker=marker,
                markersize=5,
                label=lbl,
                zorder=3,
            )
            ax.fill_between(st[x_col], st["p25"], st["p75"], color=color, alpha=0.22)
            handles.append(h)
        else:
            for li, lv in enumerate(line_vals):
                ls = LINE_LS[li % len(LINE_LS)]
                alpha = LINE_ALPHA[li % len(LINE_ALPHA)]
                lv_sub = sv[float_eq(sv[line_col], lv, line_axis)]
                if lv_sub.empty:
                    continue
                st = stats_group(lv_sub, x_col)
                lbl = f"{SOLVER_DISPLAY.get(solver, solver)}, {fmt_val(line_axis, lv)}"
                (h,) = ax.plot(
                    st[x_col],
                    st["mean"],
                    color=color,
                    linestyle=ls,
                    linewidth=1.8,
                    marker=marker,
                    markersize=5,
                    alpha=alpha,
                    label=lbl,
                    zorder=3,
                )
                ax.fill_between(
                    st[x_col], st["p25"], st["p75"], color=color, alpha=0.22 * alpha
                )
                handles.append(h)

    return handles


def annotate_speedup(
    ax,
    df: pd.DataFrame,
    x_col: str,
    pass_type: str,
    solvers: list[str],
    line_axis: str | None,
    line_vals: list,
    col_fixed: dict | None = None,
) -> None:
    if "turbompc" not in solvers:
        return

    line_col = SWEEP[line_axis]["col"] if line_axis else None
    sub = df[df["pass_type"] == pass_type].copy()
    if col_fixed:
        for fc, fv in col_fixed.items():
            sub = sub[sub[fc] == fv]

    iters = [(0, None)] if line_axis is None else list(enumerate(line_vals))

    _offsets = [8, -8, 14, -14]

    ann_idx = 0
    for solver in solvers:
        if solver == "turbompc":
            continue
        color = SOLVER_COLORS.get(solver, "grey")
        sv = sub[sub["solver"] == solver]
        sv_d = sub[sub["solver"] == "turbompc"]

        for li, lv in iters:
            alpha = LINE_ALPHA[li % len(LINE_ALPHA)]
            if lv is not None:
                mask = float_eq(sv[line_col], lv, line_axis)
                sv_lv = sv[mask]
                mask_d = float_eq(sv_d[line_col], lv, line_axis)
                sv_d_lv = sv_d[mask_d]
            else:
                sv_lv = sv
                sv_d_lv = sv_d

            if sv_lv.empty or sv_d_lv.empty:
                continue

            st = stats_group(sv_lv, x_col)
            st_d = stats_group(sv_d_lv, x_col)
            if st.empty or st_d.empty:
                continue

            # Use the last common x-value
            common_x = sorted(set(st[x_col]) & set(st_d[x_col]))
            if not common_x:
                continue
            x_last = common_x[-1]

            y_other = float(st.loc[st[x_col] == x_last, "mean"].values[0])
            y_turbompc = float(st_d.loc[st_d[x_col] == x_last, "mean"].values[0])
            if y_turbompc == 0:
                continue

            ratio = y_other / y_turbompc
            if ratio >= 1.0:
                label = f"{ratio:.1f}×"
            else:
                label = f"1/{1/ratio:.1f}×"

            xyoff = _offsets[ann_idx % len(_offsets)]
            ax.annotate(
                label,
                xy=(x_last, y_other),
                xytext=(4, xyoff),
                textcoords="offset points",
                fontsize=5,
                color=color,
                alpha=alpha,
                va="center",
                arrowprops=dict(arrowstyle="-", color=color, alpha=alpha * 0.5, lw=0.6),
            )
            ann_idx += 1


def draw_speedup_row(
    ax,
    df: pd.DataFrame,
    x_col: str,
    solvers: list[str],
    line_axis: str | None,
    line_vals: list,
    col_fixed: dict | None = None,
) -> None:
    if "turbompc" not in solvers:
        return

    line_col = SWEEP[line_axis]["col"] if line_axis else None
    sub = df.copy()
    if col_fixed:
        for fc, fv in col_fixed.items():
            sub = sub[sub[fc] == fv]

    sub = (
        sub.groupby(
            ["solver", x_col]
            + ([line_col] if line_col else [])
            + (["admm_max_iter"] if "admm_max_iter" in sub.columns else []),
            dropna=False,
        )["time_s"]
        .mean()
        .reset_index()
    )

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", zorder=1)
    ax.grid(alpha=0.3, linestyle="--")

    iters = [(0, None)] if line_axis is None else list(enumerate(line_vals))

    for solver in solvers:
        if solver == "turbompc":
            continue
        color = SOLVER_COLORS.get(solver, "grey")
        marker = SOLVER_MARKERS.get(solver, "o")
        sv = sub[sub["solver"] == solver]
        sv_d = sub[sub["solver"] == "turbompc"]

        for li, lv in iters:
            ls = LINE_LS[li % len(LINE_LS)]
            alpha = LINE_ALPHA[li % len(LINE_ALPHA)]
            if lv is not None:
                sv_lv = sv[float_eq(sv[line_col], lv, line_axis)]
                sv_d_lv = sv_d[float_eq(sv_d[line_col], lv, line_axis)]
            else:
                sv_lv, sv_d_lv = sv, sv_d

            if sv_lv.empty or sv_d_lv.empty:
                continue

            st = stats_group(sv_lv, x_col)
            st_d = stats_group(sv_d_lv, x_col)
            common_x = sorted(set(st[x_col].values) & set(st_d[x_col].values))
            if not common_x:
                continue

            st = st[st[x_col].isin(common_x)].set_index(x_col)
            st_d = st_d[st_d[x_col].isin(common_x)].set_index(x_col)
            ratio = st["mean"] / st_d["mean"]
            ax.plot(
                common_x,
                ratio.values,
                color=color,
                linestyle=ls,
                linewidth=1.8,
                marker=marker,
                markersize=5,
                alpha=alpha,
                zorder=3,
            )


def configure_xaxis(ax, x_col: str, xvals, bottom_row: bool, sdef: dict) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xticks(sorted(xvals))
    if sdef["base2"]:
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=2))
    elif x_col == "tol":
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
    else:
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.3, linestyle="--")
    if bottom_row:
        ax.set_xlabel(sdef["label"], fontsize=7)
    else:
        ax.tick_params(labelbottom=False)


def legend_handles(solvers: list[str], line_axis: str | None, line_vals: list) -> list:
    handles = []
    for solver in solvers:
        color = SOLVER_COLORS.get(solver, "grey")
        marker = SOLVER_MARKERS.get(solver, "o")
        iters = [(0, None)] if line_axis is None else enumerate(line_vals)
        for li, lv in iters:
            ls = LINE_LS[li % len(LINE_LS)]
            alpha = LINE_ALPHA[li % len(LINE_ALPHA)]
            display = SOLVER_DISPLAY.get(solver, solver)
            lbl = display if lv is None else f"{display}, {fmt_val(line_axis, lv)}"
            h = mlines.Line2D(
                [],
                [],
                color=color,
                linestyle=ls,
                linewidth=1.6,
                marker=marker,
                markersize=5,
                alpha=alpha,
                label=lbl,
            )
            handles.append(h)
    return handles


def finalize_figure(fig, axes, legend_hdl: list, args) -> None:
    fig.legend(
        handles=legend_hdl,
        loc="upper center",
        ncol=min(len(legend_hdl), 6),
        fontsize=7,
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    for ax in axes[-1, :]:  # bottom row
        lbls = ax.get_xticklabels()
        if lbls:
            lbls[-1].set_ha("left")

    if args.save:
        import os

        os.makedirs(args.output_dir, exist_ok=True)
        stem = getattr(args, "_output_stem", "fig_out")
        base = os.path.join(args.output_dir, stem)
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        print(f"Saved {base}.pdf/.png")
    else:
        import matplotlib.pyplot as plt

        plt.show()
