"""Box-plot of AD vs FD cosine similarity across solver tolerances.

Reads the .npz files produced by benchmark_turbompc_gradient_accuracy.py and
plots three panels:
    - cosine similarity for the concatenated Q+R gradient
    - cosine similarity for Q alone
    - cosine similarity for R alone

Usage (from the benchmarking/linear-system directory):
    python plot_results/plot_gradient_accuracy.py \
        --results_dir timing_results/gradient_accuracy/fwd=..._bwd=.../...
        [--outfile figs/gradient_accuracy.pdf]
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import apply_ieee_style


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _tol_from_filename(fname: str) -> float:
    """Parse tolerance from filename like 'tol_1en3.npz' -> 1e-3."""
    stem = os.path.basename(fname).removeprefix("tol_").removesuffix(".npz")
    # stored as e.g. '1en3' (minus sign replaced with 'n')
    stem = stem.replace("n", "-")
    return float(stem)


def load_accuracy_results(results_dir: str) -> dict[float, dict[str, np.ndarray]]:
    """Return {tol: {metric: array[n_seeds]}} sorted by ascending tolerance."""
    pattern = os.path.join(results_dir, "tol_*.npz")
    files = sorted(glob.glob(pattern), key=_tol_from_filename)
    if not files:
        raise FileNotFoundError(f"No tol_*.npz files found in {results_dir}")

    results: dict[float, dict[str, np.ndarray]] = {}
    for f in files:
        tol = _tol_from_filename(f)
        d = np.load(f)
        results[tol] = {k: d[k] for k in d.files}
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _format_tol(tol: float) -> str:
    exp = int(round(np.log10(tol)))
    return rf"$10^{{{exp}}}$"


def plot_cosine_similarity(
    results: dict[float, dict[str, np.ndarray]],
    outfile: str | None = None,
    ieee_style: bool = True,
) -> None:
    if ieee_style:
        apply_ieee_style()

    tols = sorted(results.keys())
    x = np.arange(len(tols))
    labels = [_format_tol(t) for t in tols]

    metrics = [
        ("cos_all", "Q + R (combined)"),
        ("cos_Q",   "Q only"),
        ("cos_R",   "R only"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for ax, (metric, title) in zip(axes, metrics):
        data = [results[t][metric] for t in tols]
        bp = ax.boxplot(
            data,
            positions=x,
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.2),
            boxprops=dict(facecolor="#2166ac", alpha=0.55),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker=".", markersize=3, alpha=0.5),
        )
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Solver tolerance")
        ax.set_title(title)
        ax.set_ylim(None, 1.02)

    axes[0].set_ylabel("Cosine similarity (AD vs FD)")

    fig.suptitle(
        "TurboMPC gradient accuracy: AD vs finite-difference cosine similarity",
        fontsize=8 if ieee_style else 11,
        y=1.01,
    )
    fig.tight_layout()

    if outfile:
        os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
        fig.savefig(outfile, bbox_inches="tight")
        print(f"Saved {outfile}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Optional: relative L2 error subplot
# ---------------------------------------------------------------------------

def plot_rel_l2_error(
    results: dict[float, dict[str, np.ndarray]],
    outfile: str | None = None,
    ieee_style: bool = True,
) -> None:
    if ieee_style:
        apply_ieee_style()

    tols = sorted(results.keys())
    x = np.arange(len(tols))
    labels = [_format_tol(t) for t in tols]

    metrics = [
        ("rel_l2_all", "Q + R (combined)"),
        ("rel_l2_Q",   "Q only"),
        ("rel_l2_R",   "R only"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for ax, (metric, title) in zip(axes, metrics):
        data = [results[t][metric] for t in tols]
        ax.boxplot(
            data,
            positions=x,
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.2),
            boxprops=dict(facecolor="#d6604d", alpha=0.55),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker=".", markersize=3, alpha=0.5),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Solver tolerance")
        ax.set_title(title)
        ax.set_yscale("log")

    axes[0].set_ylabel(r"Relative $\ell_2$ error $\|g_\mathrm{AD}-g_\mathrm{FD}\|/\|g_\mathrm{FD}\|$")

    fig.suptitle(
        "TurboMPC gradient accuracy: relative L2 error",
        fontsize=8 if ieee_style else 11,
        y=1.01,
    )
    fig.tight_layout()

    if outfile:
        os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
        fig.savefig(outfile, bbox_inches="tight")
        print(f"Saved {outfile}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Multi-SCP grouped box / violin plot (cos_all only)
# ---------------------------------------------------------------------------

# Colors per SCP series (blue family → darker = more iterations)
_SCP_COLORS = {1: "#053061", 10: "#2166ac", 50: "#053061"}


def _median_mad(arr: np.ndarray) -> tuple[float, float]:
    """Return (median, median absolute deviation)."""
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad


def plot_cosine_by_scp(
    scp_results: dict[int, dict[float, dict[str, np.ndarray]]],
    outfile: str | None = None,
    ieee_style: bool = True,
    violin: bool = False,
) -> None:
    """Grouped box or violin plot: one group per tolerance, one glyph per SCP value.

    The median is shown as a green line (box plot) or green dot (violin).
    A median ± MAD error bar is overlaid on the violin.

    Args:
        scp_results: {scp_value: {tol: {metric: array[n_seeds]}}}
        violin: if True draw violins instead of boxes.
    """
    if ieee_style:
        apply_ieee_style()

    scp_values = sorted(scp_results.keys())
    tols = sorted(next(iter(scp_results.values())).keys())
    n_tols = len(tols)
    n_scp = len(scp_values)

    group_width = 0.8
    glyph_width = group_width / n_scp * 0.85
    offsets = np.linspace(-group_width / 2, group_width / 2, n_scp, endpoint=False)
    offsets += group_width / (2 * n_scp)

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    # Alternating gray/white column bands (one band per tolerance group)
    for i in range(n_tols):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="#f0f0f0", zorder=0)

    for scp, offset in zip(scp_values, offsets):
        color = _SCP_COLORS.get(scp, "#333333")
        positions = np.arange(n_tols) + offset
        data = [scp_results[scp][t]["cos_all"] for t in tols]

        if violin:
            parts = ax.violinplot(
                data,
                positions=positions,
                widths=glyph_width,
                showmedians=False,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.75)
                pc.set_edgecolor(color)
                pc.set_linewidth(0.5)
            # Overlay median bar ± MAD
            for pos, arr in zip(positions, data):
                med, mad = _median_mad(np.asarray(arr))
                ax.plot([pos - glyph_width * 0.4, pos + glyph_width * 0.4],
                        [med, med], color="#2ca02c", linewidth=1.2, zorder=4)
                ax.errorbar(pos, med, yerr=mad, fmt="none",
                            ecolor="#2ca02c", elinewidth=0.8,
                            capsize=1.5, capthick=0.8, zorder=4)
        else:
            ax.boxplot(
                data,
                positions=positions,
                widths=glyph_width * 0.5,
                patch_artist=True,
                medianprops=dict(color=color, linewidth=1.5),
                boxprops=dict(facecolor="white", edgecolor=color, linewidth=0.9),
                whiskerprops=dict(color=color, linewidth=0.7),
                capprops=dict(color=color, linewidth=0.7),
                flierprops=dict(marker=".", markersize=2, color=color, alpha=0.5),
            )


    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.6, zorder=1)
    ax.set_xticks(np.arange(n_tols))
    ax.set_xticklabels([_format_tol(t) for t in tols], fontsize=6)
    ax.set_xlabel("Solver tolerance", fontsize=7)
    ax.set_ylabel("Cosine similarity (AD vs FD)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlim(-0.5, n_tols - 0.5)
    ax.set_ylim(None, 1.02)
    ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.5, zorder=1)
    ax.set_axisbelow(False)  # grid drawn above bands but below violins

    fig.tight_layout()

    if outfile:
        os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
        fig.savefig(outfile, bbox_inches="tight")
        print(f"Saved {outfile}")
        base, _ = os.path.splitext(outfile)
        fig.savefig(base + ".png", bbox_inches="tight", dpi=200)
        print(f"Saved {base}.png")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot AD vs FD cosine similarity box plots from gradient accuracy sweep"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory containing tol_*.npz files",
    )
    parser.add_argument(
        "--scp_dirs",
        type=str,
        nargs="+",
        metavar="SCP=SUBDIR",
        default=None,
        help=("Triggers the grouped box plot instead of the per-key plots."
        ),
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Output file path (e.g. figs/gradient_accuracy.pdf). "
             "Omit to show interactively.",
    )
    parser.add_argument(
        "--rel_l2",
        action="store_true",
        help="Also produce a relative L2 error plot (saved with _rel_l2 suffix)",
    )
    parser.add_argument(
        "--no_ieee",
        action="store_true",
        help="Disable IEEE-style rcParams",
    )
    parser.add_argument(
        "--violin",
        action="store_true",
        help="Draw violin plots instead of box plots (grouped SCP mode only)",
    )
    args = parser.parse_args()

    # ---- Grouped SCP mode ----
    if args.scp_dirs:
        scp_results: dict[int, dict[float, dict[str, np.ndarray]]] = {}
        for entry in args.scp_dirs:
            scp_str, subdir = entry.split("=", 1)
            scp_val = int(scp_str)
            full_dir = os.path.join(args.results_dir, subdir)
            scp_results[scp_val] = load_accuracy_results(full_dir)
            print(
                f"  SCP={scp_val}: loaded {len(scp_results[scp_val])} tolerance levels "
                f"from {full_dir}"
            )
        plot_cosine_by_scp(
            scp_results,
            outfile=args.outfile,
            ieee_style=not args.no_ieee,
            violin=args.violin,
        )
        return

    # ---- Single-directory mode ----
    results = load_accuracy_results(args.results_dir)
    print(f"Loaded {len(results)} tolerance levels: {sorted(results.keys())}")

    plot_cosine_similarity(
        results,
        outfile=args.outfile,
        ieee_style=not args.no_ieee,
    )

    if args.rel_l2:
        if args.outfile:
            base, ext = os.path.splitext(args.outfile)
            rel_outfile = f"{base}_rel_l2{ext}"
        else:
            rel_outfile = None
        plot_rel_l2_error(
            results,
            outfile=rel_outfile,
            ieee_style=not args.no_ieee,
        )


if __name__ == "__main__":
    main()
