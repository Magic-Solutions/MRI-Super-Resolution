#!/usr/bin/env python3
"""Per-sigma denoising error figure (ablation B1) from the CSVs written by scripts/per_sigma_error.py.

Left: the EDM-weighted denoising loss of each 2.5D model as a function of the noise level, with
the training distribution of ln(sigma) (shaded) and the 20 sampler sigmas (ticks) on the same axis.
Right: PSNR of the single-step denoised estimate at each sigma. Together they show whether the
training distribution covers the noise levels the sampler visits and where the denoiser is weakest.

Usage: python scripts/make_per_sigma_figure.py [--in-dir ../results/ablation/per_sigma]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LABELS = {"25d_both": "2.5D", "25d": "2.5D-prev"}
COLORS = {"25d_both": "#d62728", "25d": "#ff9f1c"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=ROOT.parent / "results" / "ablation" / "per_sigma")
    p.add_argument("--out-dir", type=Path, default=ROOT / "publication" / "manuscript_ieee_access" / "figs")
    p.add_argument("--png-dir", type=Path, default=ROOT.parent / "results" / "figures" / "png")
    args = p.parse_args()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.4))
    sampler = [float(x) for x in (args.in_dir / "25d_both.sampler_sigmas.txt").read_text().split()]
    dens = None
    for key in ("25d_both", "25d"):
        d = pd.read_csv(args.in_dir / f"{key}.csv")
        ax1.plot(d.sigma, d.loss_denoising, "-o", ms=3, color=COLORS[key], label=LABELS[key])
        ax2.plot(d.sigma, d.psnr_denoised, "-o", ms=3, color=COLORS[key], label=LABELS[key])
        dens = d[["sigma", "train_density"]]
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel(r"noise level $\sigma$")
        ax.grid(alpha=0.3)
        for s in sampler:
            ax.axvline(s, color="gray", lw=0.5, alpha=0.5)
    # Training density of ln(sigma) as a shaded band on a secondary axis of the left panel.
    ax1b = ax1.twinx()
    ax1b.fill_between(dens.sigma, 0, dens.train_density, color="#d4e6f6", alpha=0.8, zorder=0)
    ax1b.set_ylim(0, dens.train_density.max() * 3)
    ax1b.set_yticks([])
    ax1b.set_ylabel(r"training density of $\ln\sigma$", color="#4a7fb5", fontsize=7)
    ax1.set_zorder(ax1b.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax1.set_ylabel("denoising loss (EDM-weighted)")
    ax2.set_ylabel("one-step denoised PSNR (dB)")
    ax1.plot([], [], color="gray", lw=0.5, label="sampler sigmas (20 steps)")
    ax1.legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "fig_per_sigma.pdf")
    fig.savefig(args.png_dir / "fig_per_sigma.png", dpi=150)
    print("wrote", args.out_dir / "fig_per_sigma.pdf")


if __name__ == "__main__":
    main()
