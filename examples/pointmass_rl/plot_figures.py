"""IEEE-styled paper figures for the point-mass APG example.

Imports the OCP/solver/rollout setup from ``pointmass_rl.py`` (no training is
triggered; the training loop lives behind a ``__main__`` guard) and reuses
``apply_ieee_style()`` and the TurboMPC blue from the linear-system benchmark
figures so all paper figures look uniform.

Outputs (overwrites in place):
  outputs/pointmass_apg_500_curve.{png,pdf}     learning curve
  outputs/pointmass_trajectory_3d.{png,pdf}     3D closed-loop trajectory

Run:  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. \\
        .venv/bin/python examples/pointmass_rl/plot_figures.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Make repo root importable for turbompc + the example modules.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, str(REPO_ROOT / "benchmarking" / "linear-system" / "plot_results"))
from plot_utils import SOLVER_COLORS, SOLVER_MARKERS, apply_ieee_style  # noqa: E402

LEARNED_COLOR = SOLVER_COLORS["turbompc"]  # "#2166ac" — TurboMPC blue
LEARNED_MARKER = SOLVER_MARKERS["turbompc"]  # "o"
REFERENCE_COLOR = "0.20"  # near-black
UNTRAINED_COLOR = "#c0392b"  # red
TIME_CMAP = "viridis"  # time gradient along the learned trajectory

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"


# ────────────────────────────────────────────────────────────────────────────
# Figure 1: learning curve
# ────────────────────────────────────────────────────────────────────────────

def make_learning_curve(
    out_stem: Path = OUTPUT_DIR / "pointmass_apg_500_curve",
) -> None:
    from examples.pointmass_rl.pointmass_rl import LOG_PATH

    apply_ieee_style()

    steps, rmses = [], []
    with open(LOG_PATH) as f:
        for r in csv.DictReader(f):
            if r["eval_rmse"]:
                steps.append(int(r["step"]))
                rmses.append(float(r["eval_rmse"]))
    steps, rmses = np.array(steps), np.array(rmses)

    # IEEE journal column ≈ 3.5 in; subfloat at 0.48\columnwidth ≈ 1.68 in.
    fig, ax = plt.subplots(figsize=(1.7, 1.35))
    # axhline first so "untrained" sits on top of the legend.
    ax.axhline(rmses[0], color=UNTRAINED_COLOR, ls="-", lw=0.8, label="untrained")
    ax.plot(steps, rmses, "-", color=LEARNED_COLOR, lw=1.0,
            marker=LEARNED_MARKER, markersize=2.5, label="learned",
            markevery=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("RMSE")
    ax.set_xlim(0, 500)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.4)
    ax.legend(loc="upper right", framealpha=0.9, handlelength=1.5,
              borderpad=0.3, labelspacing=0.25)

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_stem.with_suffix('.pdf').name}  +  .png")


# ────────────────────────────────────────────────────────────────────────────
# Figure 2: 3D closed-loop trajectory
# ────────────────────────────────────────────────────────────────────────────

def _load_theta(npz_path: Path, template):
    import jax
    import jax.numpy as jnp

    leaves, treedef = jax.tree.flatten(template)
    data = np.load(npz_path)
    return jax.tree.unflatten(
        treedef, [jnp.asarray(data[f"leaf_{i}"]) for i in range(len(leaves))]
    )


def make_trajectory_3d(
    out_stem: Path = OUTPUT_DIR / "pointmass_trajectory_3d",
) -> None:
    from examples.pointmass_rl import pointmass_rl as pm
    from examples.pointmass_rl.policy import init_policy

    import jax
    import jax.numpy as jnp

    apply_ieee_style()

    s0 = jnp.concatenate([pm.pos_target_at_step(0), jnp.zeros(3)])
    tgt = np.stack([np.asarray(pm.pos_target_at_step(i)) for i in range(pm.N_ROLL)])

    th_init = init_policy(jax.random.PRNGKey(0), obs_dim=6, out_dim=pm.H_STAGES * 9)
    st_init, _ = pm.rollout(th_init, s0, 0, pm.N_ROLL)
    st_init = np.asarray(jax.block_until_ready(st_init))
    th = _load_theta(pm.THETA_PATH, th_init)
    st_trained, _ = pm.rollout(th, s0, 0, pm.N_ROLL)
    st_trained = np.asarray(jax.block_until_ready(st_trained))

    fig = plt.figure(figsize=(2.1, 1.7))
    ax = fig.add_subplot(111, projection="3d")
    ref_h, = ax.plot(tgt[:, 0], tgt[:, 1], tgt[:, 2], "-o",
                     color=REFERENCE_COLOR, lw=0.9, ms=1.6,
                     markevery=pm.SEGMENT_STEPS, label="reference")
    unt_h, = ax.plot(st_init[1:, 0], st_init[1:, 1], st_init[1:, 2], "-",
                     color=UNTRAINED_COLOR, lw=0.9, label="untrained")

    # Learned trajectory as a per-segment time gradient.
    pts = st_trained[1:, :3]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=TIME_CMAP,
                          norm=plt.Normalize(0, len(pts) - 1), linewidths=1.3)
    lc.set_array(np.arange(len(pts) - 1))
    ax.add_collection3d(lc)

    start_h = ax.scatter(*np.asarray(pm.pos_target_at_step(0)),
                         color="black", s=14, marker="s",
                         facecolors="none", linewidth=0.9, label="start")

    allp = np.concatenate([tgt, st_init[1:, :3], pts], axis=0)
    ax.auto_scale_xyz(allp[:, 0], allp[:, 1], allp[:, 2])

    # Text scaled ~12% up so it prints the same size as the learning-curve
    # panel when both are included at equal widths (this crop is ~12% wider).
    fs_ticks, fs_legend, fs_cb_label, fs_cb_ticks = 6.75, 7.9, 6.75, 6.2

    learned_h = Line2D([], [], color=matplotlib.colormaps[TIME_CMAP](0.5),
                       lw=1.3, label="learned")
    # 2x2 legend above the plot so it never occludes the trajectories.
    ax.legend(handles=[ref_h, unt_h, learned_h, start_h], ncols=2,
              loc="lower center", bbox_to_anchor=(0.5, 0.80),
              bbox_transform=fig.transFigure, framealpha=0.9,
              handlelength=1.5, borderpad=0.25, labelspacing=0.25,
              columnspacing=1.0, fontsize=fs_legend)

    cb = fig.colorbar(lc, ax=ax, orientation="horizontal",
                      fraction=0.05, pad=0.14, shrink=0.65)
    cb.set_label("time step", fontsize=fs_cb_label)
    cb.ax.tick_params(labelsize=fs_cb_ticks, width=0.5)
    cb.outline.set_linewidth(0.5)
    # Embed the gradient as one raster patch in vector output: some PDF
    # viewers overpaint the quad-mesh cell edges, bleeding color past the
    # outline. Outline/ticks/label stay vector.
    cb.solids.set_rasterized(True)
    cb.solids.set_edgecolor("face")

    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect(None, zoom=1.18)  # shrink mplot3d's default margins
    ax.tick_params(axis="x", pad=-2, labelsize=fs_ticks)
    ax.tick_params(axis="y", pad=-2, labelsize=fs_ticks)
    ax.tick_params(axis="z", pad=-2, labelsize=fs_ticks)
    # 0.1-spaced ticks on x/y (drop the 0.05 labels) to reduce clutter.
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(alpha=0.3)

    # Tight bbox; bottom_extra adds whitespace below if the subfloat needs
    # to bottom-align with the learning-curve panel in the LaTeX figure.
    pad, bottom_extra = 0.05, 0.0
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    bbox = Bbox.from_extents(tight.x0 - pad, tight.y0 - pad - bottom_extra,
                             tight.x1 + pad, tight.y1 + pad)
    # dpi applies to the rasterized colorbar gradient inside the PDF.
    fig.savefig(out_stem.with_suffix(".pdf"), dpi=300, bbox_inches=bbox)
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches=bbox)
    plt.close(fig)
    print(f"saved -> {out_stem.with_suffix('.pdf').name}  +  .png")


if __name__ == "__main__":
    make_learning_curve()
    make_trajectory_3d()
