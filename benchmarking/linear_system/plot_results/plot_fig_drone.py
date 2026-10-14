"""Drone obstacle-avoidance trajectory figure (w/o slack vs w/ slack).

Usage:
    python3 plot_results/plot_fig_drone.py --traj_dir z_drone/trajectories/subset_a \
        --out figs/drone --no_show
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

_HERE = os.path.dirname(os.path.abspath(__file__))
_Z_DRONE = os.path.join(_HERE, "..", "z_drone")
sys.path.insert(0, _Z_DRONE)

from benchmark_drone_params import DRONE_X0_BASE, OBS_CENTERS, OBS_RADII  # noqa: E402
from plot_utils import apply_ieee_style  # noqa: E402

apply_ieee_style()

_C_NOSLACK = "#2E6DAD"
_C_SLACK = "#D4602A"


def _draw_obstacles(ax):
    for (cx, cy), r in zip(OBS_CENTERS, OBS_RADII):
        ax.add_patch(Circle((cx, cy), r, fc="#DDDDDD", ec="#888888", lw=0.6, zorder=2))
    ax.plot(DRONE_X0_BASE[0], DRONE_X0_BASE[1], "k^", ms=5, zorder=8, clip_on=False)
    ax.plot(0.0, 0.0, "*", color="#1CA832", ms=7, zorder=8, ls="none")


def _style_env(ax):
    ax.set_xlim(-2.25, 0.25)
    ax.set_ylim(-0.50, 0.70)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel(r"$y$ (m)", fontsize=6, labelpad=2)
    ax.set_xlabel(r"$x$ (m)", fontsize=6, labelpad=2)
    ax.tick_params(labelsize=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, lw=0.3, color="#CCCCCC", alpha=0.5, zorder=0)


def _plot_trajectories(
    ax,
    traj_dir,
    scheme="rk4",
    init="line",
    dt=1.0,
    sqp=8,
    admm=50,
    rd_noslack=0.0,
    rd_slack=0.0,
    tol=0.001,
):
    _draw_obstacles(ax)
    for slack, ls, color, rd in [
        ("noslack", "-", _C_NOSLACK, rd_noslack),
        ("slack", "--", _C_SLACK, rd_slack),
    ]:
        pattern = os.path.join(
            traj_dir,
            (
                f"turbompc_drone_traj_{slack}_{scheme}_{init}_dt{dt}"
                f"_sqp{sqp}_admm{admm}_rd{rd}_tol{tol}_s*.npy"
            ),
        )
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"Warning: no files matched {pattern}")
        for i, fp in enumerate(files):
            t = np.load(fp)
            ax.plot(
                t[:, 0],
                t[:, 1],
                color=color,
                ls=ls,
                lw=1.4 if i == 0 else 0.5,
                alpha=0.90 if i == 0 else 0.18,
                zorder=4 + (i == 0),
            )
    _style_env(ax)


def _legend_handles():
    return [
        mpatches.Patch(fc="#DDDDDD", ec="#888888", lw=0.6, label="Obstacle"),
        Line2D([0], [0], color="k", marker="^", ls="none", ms=4, label="Start"),
        Line2D([0], [0], color="#1CA832", marker="*", ls="none", ms=5, label="Goal"),
        Line2D([0], [0], color=_C_NOSLACK, lw=1.2, ls="-", label="w/o slack"),
        Line2D([0], [0], color=_C_SLACK, lw=1.2, ls="--", label="w/ slack"),
    ]


def _save_or_show(fig, out_prefix, tag, show):
    if out_prefix:
        os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
        for ext in ("png", "pdf"):
            path = f"{out_prefix}_{tag}.{ext}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            print(f"Saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--traj_dir", default=os.path.join(_Z_DRONE, "trajectories", "subset_a")
    )
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--sqp", type=int, default=8)
    ap.add_argument("--admm", type=int, default=50)
    ap.add_argument("--rd_noslack", type=float, default=0.0)
    ap.add_argument("--rd_slack", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=0.001)
    ap.add_argument("--out", default=None, help="Output prefix (no extension)")
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    _plot_trajectories(
        ax,
        args.traj_dir,
        dt=args.dt,
        sqp=args.sqp,
        admm=args.admm,
        rd_noslack=args.rd_noslack,
        rd_slack=args.rd_slack,
        tol=args.tol,
    )
    ax.legend(
        handles=_legend_handles(),
        fontsize=5,
        loc="upper left",
        ncol=2,
        framealpha=0.85,
        handlelength=1.8,
        borderpad=0.4,
        labelspacing=0.25,
        columnspacing=0.8,
    )
    fig.tight_layout(pad=0.4)
    _save_or_show(fig, args.out, "drone", not args.no_show)


if __name__ == "__main__":
    main()
