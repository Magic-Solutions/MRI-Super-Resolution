#!/usr/bin/env python3
"""Learning-curve (fig:curves) and method-overview (Fig. 1) figures for the resubmission.

Learning curves: validation PSNR of the six main runs from the val_metrics.csv files in
results/training_curves/ and results/cont_*/ (the Swin2SR continuation exported from W&B), with
continuations dashed and the reported checkpoint marked. The x axis is passes over the training
data, so runs that count epochs and steps differently share one axis.

Method overview: a schematic of the 2.5D and 3D pipelines.

Usage:
    python scripts/make_curves_and_overview.py [--out-dir publication/manuscript_ieee_access/figs]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT.parent / "results"

# One pass over the 88,000 training slices: 2.5D epochs are 400 steps x 32 segments x 2 targets,
# Res-SRDiff steps are batches of 16 slices; the others train one pass per epoch.
SLICES = 88_000
PASSES_PER_25D_EPOCH = 400 * 32 * 2 / SLICES
SLICES_PER_RSD_STEP = 16


def passes(run: str, df: pd.DataFrame) -> pd.Series:
    if run.startswith("25d"):
        return df["epoch"] * PASSES_PER_25D_EPOCH
    if run == "ressrdiff":
        return df["step"] * SLICES_PER_RSD_STEP / SLICES
    return df["epoch"].astype(float)


def load_runs() -> dict[str, tuple[pd.DataFrame, pd.DataFrame | None, int | None]]:
    """run -> (initial curve, continuation curve or None, reported epoch or None)."""
    tc = RES / "training_curves"
    runs = {}
    runs["25d_prev"] = (pd.read_csv(tc / "25d.csv"), pd.read_csv(RES / "cont_25d" / "val_metrics.csv"), 30)
    # Re-validated with the edge-padding fix by scripts/recompute_val_psnr.py.
    both = pd.read_csv(tc / "25d_both_val_psnr_fixed.csv")
    both_cont = pd.read_csv(RES / "cont_25d_both" / "val_metrics.csv")
    runs["25d"] = (both, both_cont, 27)
    runs["3d"] = (pd.read_csv(tc / "3d.csv"), pd.read_csv(RES / "cont_3d" / "val_metrics.csv"), 47)
    runs["edsr"] = (pd.read_csv(tc / "edsr.csv"), None, 16)
    sw_cont = tc / "swin2sr_cont_wandb.csv"
    runs["swin2sr"] = (pd.read_csv(tc / "swin2sr.csv"), pd.read_csv(sw_cont) if sw_cont.exists() else None, None)
    runs["ressrdiff"] = (pd.read_csv(tc / "ressrdiff.csv"), None, None)
    return runs


LABELS = {"25d": "2.5D", "25d_prev": "2.5D-prev", "3d": "3D", "edsr": "EDSR", "swin2sr": "Swin2SR$^{p}$", "ressrdiff": "Res-SRDiff"}
COLORS = {"25d": "#d62728", "25d_prev": "#ff9f1c", "3d": "#5b5bd6", "edsr": "#1f77b4", "swin2sr": "#e8743b", "ressrdiff": "#2ca58d"}


def fig_curves(out_dir: Path, png_dir: Path) -> None:
    runs = load_runs()
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    for run, (init, cont, best) in runs.items():
        c = COLORS[run]
        x0, y0 = passes(run, init), init["val_psnr"]
        ax.plot(x0, y0, color=c, lw=1.2, label=LABELS[run])
        if cont is not None:
            # The continuation's first row is the starting weights (the initial run's last epoch).
            cc = cont[cont["epoch"] > init["epoch"].max()] if "epoch" in cont else cont
            if len(cc):
                ax.plot(passes(run, cc), cc["val_psnr"], color=c, lw=1.2, ls="--")
        if best is not None:
            src = pd.concat([init, cont]) if cont is not None else init
            row = src[src["epoch"] == best].iloc[-1]
            ax.plot(passes(run, pd.DataFrame([row])).iloc[0], row["val_psnr"], "o", color=c, ms=5, mec="black", mew=0.6)
    ax.set_xlabel("Passes over the training data")
    ax.set_ylabel("Validation PSNR (dB)")
    ax.set_ylim(33, 39)
    ax.grid(alpha=0.3)
    ax.plot([], [], color="gray", lw=1.2, ls="--", label="continuation")
    ax.plot([], [], "o", color="white", mec="black", mew=0.6, label="reported checkpoint")
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_training_curves.pdf")
    fig.savefig(png_dir / "fig_training_curves.png", dpi=150)
    plt.close(fig)
    print("wrote", out_dir / "fig_training_curves.pdf")


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec="black", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "black", "shrinkA": 0, "shrinkB": 0})


def fig_overview(out_dir: Path, png_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 2.6))
    rows = [
        ("(a) 2.5D slice-conditioned pipeline", [
            ("LR volume\n$128\\times128\\times D$", "#e6e6e6"),
            ("Target slice +\nboth neighbours\n(bicubic $\\uparrow 2$)", "#fff3c4"),
            ("2D EDM U-Net\n64/64/128/256\nattention at level 4", "#d9f0d3"),
            ("Shared sampler\n20 Euler steps", "#d4e6f6"),
            ("HR slice\n$256\\times256$", "#f8d7da"),
        ]),
        ("(b) 3D volumetric pipeline", [
            ("LR volume\n$128\\times128\\times D$", "#e6e6e6"),
            ("$64^3$ LR patch\n(trilinear $\\uparrow 2$)", "#fff3c4"),
            ("3D EDM U-Net\n32/64/128/256\nattention at level 4", "#d9f0d3"),
            ("Shared sampler\n20 Euler steps", "#d4e6f6"),
            ("HR patch, blended\n$64\\times128\\times128$", "#f8d7da"),
        ]),
    ]
    for ax, (title, boxes) in zip(axes, rows):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.0, 0.93, title, fontsize=8.5, fontweight="bold", va="center")
        n = len(boxes)
        w, gap = 0.16, (1 - 0.16 * n) / (n - 1)
        for i, (text, fc) in enumerate(boxes):
            x = i * (w + gap)
            box(ax, x, 0.08, w, 0.68, text, fc)
            if i < n - 1:
                arrow(ax, x + w + 0.005, x + w + gap - 0.005, 0.42)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(out_dir / "fig_method_overview.pdf")
    fig.savefig(png_dir / "fig_method_overview.png", dpi=150)
    plt.close(fig)
    print("wrote", out_dir / "fig_method_overview.pdf")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=ROOT / "publication" / "manuscript_ieee_access" / "figs")
    p.add_argument("--png-dir", type=Path, default=RES / "figures" / "png")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.png_dir.mkdir(parents=True, exist_ok=True)
    fig_curves(args.out_dir, args.png_dir)
    fig_overview(args.out_dir, args.png_dir)


if __name__ == "__main__":
    main()
