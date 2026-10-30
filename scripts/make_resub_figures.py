#!/usr/bin/env python3
"""Qualitative figures for the resubmission, from stored predictions and the per-slice CSVs.

Inputs:
- ``--pred-dir``: ``data/<volume>.hdf5`` (datasets ``hr``, ``lr``, uint8) and
  ``pred/<method>/<volume>.hdf5`` as written by ``evaluate_sr.py --save-predictions``.
- ``--csv-dir``: the final evaluation's block-averaging folder (per-slice CSVs of every method) and
  ``--csv-25d``: the per-slice CSV of the 2.5D (both-neighbours) model.

Outputs (PDF in ``--out-dir``, PNG previews in ``--png-dir``):
- fig_visual_all_methods: one mid-sagittal slice, all methods, full slice over a zoomed cortical crop.
- fig_pixel_error_heatmap: |prediction - GT| of the same slice, one colour scale for every method.
- fig_failure_case: the slice where the 2.5D model loses most foreground PSNR against EDSR.
- fig_per_subject_psnr: per-subject PSNR of every method over the 70 test volumes, sorted by
  bicubic PSNR, with a bootstrap 95% CI band (over slices within the subject) for the 2.5D model.

Usage:
    python scripts/make_resub_figures.py --per-subject-only      # needs only the CSVs
    python scripts/make_resub_figures.py --pred-dir ../results/figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Method key in the prediction folders / CSVs -> label in the figures. The paper's "2.5D" is the
# both-neighbours model; "2.5D-prev" conditions on the previous slice only.
METHODS = [("bicubic", "Bicubic"), ("edsr", "EDSR"), ("swin2sr", "Swin2SR"), ("ressrdiff", "Res-SRDiff"),
           ("25d_prev", "2.5D-prev"), ("25d", "2.5D"), ("3d", "3D")]
# Per-slice CSV of each method in the final evaluation folder (the 2.5D one is passed separately).
CSV_NAMES = {"bicubic": "bicubic.csv", "edsr": "edsr.csv", "swin2sr": "swin2sr.csv", "ressrdiff": "ressrdiff.csv",
             "25d_prev": "25d.csv", "3d": "3d.csv"}
# Colour-blind-safe palette, one fixed colour per method.
COLORS = {"bicubic": "#555555", "edsr": "#2a78d6", "swin2sr": "#eb6834", "ressrdiff": "#1baf7a",
          "25d_prev": "#eda100", "25d": "#e34948", "3d": "#6250d6"}
MARKERS = {"bicubic": "s", "edsr": "o", "swin2sr": "^", "ressrdiff": "v", "25d_prev": "D", "25d": "o", "3d": "x"}

plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "pdf.fonttype": 42})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", type=Path, default=ROOT.parent / "results" / "figures")
    p.add_argument("--csv-dir", type=Path, default=ROOT.parent / "results" / "final_eval_20261024-2204" / "blockavg")
    p.add_argument("--csv-25d", type=Path,
                   default=ROOT.parent / "results" / "converged_25d_both" / "blockavg" / "25d_both_conv.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "publication" / "manuscript_ieee_access" / "figs")
    p.add_argument("--png-dir", type=Path, default=ROOT.parent / "results" / "figures" / "png")
    p.add_argument("--visual-volume", default="sub_2794_ses_1_t1_sagittal", help="Volume of figures 1 and 2")
    p.add_argument("--visual-slice", type=int, default=88, help="Slice of figures 1 and 2 (88 = mid-sagittal of 176)")
    p.add_argument("--zoom", type=int, nargs=4, default=[36, 100, 80, 144], metavar=("R0", "R1", "C0", "C1"),
                   help="Rows and columns of the zoomed crop (HR pixels)")
    p.add_argument("--failure-cases", nargs="+", default=["sub_5736_ses_1_t1_sagittal:170", "sub_4897_ses_1_t1_sagittal:47"],
                   metavar="VOLUME:SLICE", help="One row per case: the global worst slice (2.5D vs EDSR, foreground "
                   "PSNR) and the worst central slice among the stored volumes")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--per-subject-only", action="store_true")
    return p.parse_args()


def psnr_uint8(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """PSNR of uint8 images (peak 255); with ``mask``, over the masked pixels only (foreground PSNR)."""
    a, b = pred.astype(np.float64), gt.astype(np.float64)
    d = (a - b)[mask] if mask is not None else a - b
    mse = np.mean(d ** 2)
    return 100.0 if mse < 1e-10 else 10 * math.log10(255.0 ** 2 / mse)


def load_slice(pred_dir: Path, volume: str, si: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(pred_dir / "data" / f"{volume}.hdf5", "r") as f:
        gt, lr = np.asarray(f["hr"][si]), np.asarray(f["lr"][si])
    preds = {}
    for key, _ in METHODS:
        with h5py.File(pred_dir / "pred" / key / f"{volume}.hdf5", "r") as f:
            preds[key] = np.asarray(f[f"frame_{si}_x"])
    return gt, lr, preds


def save(fig: plt.Figure, name: str, args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.png_dir / f"{name}.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {args.out_dir / f'{name}.pdf'}")


def fig_visual_all_methods(args: argparse.Namespace) -> dict:
    gt, _, preds = load_slice(args.pred_dir, args.visual_volume, args.visual_slice)
    r0, r1, c0, c1 = args.zoom
    panels = [("gt", "Ground truth", gt)] + [(k, label, preds[k]) for k, label in METHODS]
    fig, axes = plt.subplots(2, len(panels), figsize=(7.16, 2.05),  # 8 square panels across 7.16 in -> two rows of ~0.9 in plus titles
                             gridspec_kw={"wspace": 0.03, "hspace": 0.04})
    numbers = {}
    for j, (key, label, img) in enumerate(panels):
        top, bottom = axes[0, j], axes[1, j]
        top.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        top.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, edgecolor="red", linewidth=0.8))
        top.set_title(label, pad=2)
        crop = img[r0:r1, c0:c1]
        bottom.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        for spine in bottom.spines.values():
            spine.set_edgecolor("red"); spine.set_linewidth(0.8)
        bottom.set_xticks([]); bottom.set_yticks([])
        top.axis("off")
        if key != "gt":
            full, zoom = psnr_uint8(img, gt), psnr_uint8(crop, gt[r0:r1, c0:c1])
            numbers[key] = {"psnr_full": round(full, 2), "psnr_zoom": round(zoom, 2),
                            "psnr_fg": round(psnr_uint8(img, gt, gt > 0), 2)}
            kw = dict(color="white", fontsize=6, ha="right", va="bottom",
                      bbox=dict(facecolor="black", alpha=0.55, pad=1, edgecolor="none"))
            top.text(0.97, 0.03, f"{full:.2f} dB", transform=top.transAxes, **kw)
            bottom.text(0.97, 0.03, f"{zoom:.2f} dB", transform=bottom.transAxes, **kw)
    save(fig, "fig_visual_all_methods", args)
    return {"volume": args.visual_volume, "slice": args.visual_slice, "zoom_rows_cols": args.zoom, "psnr": numbers}


def fig_pixel_error_heatmap(args: argparse.Namespace) -> dict:
    gt, _, preds = load_slice(args.pred_dir, args.visual_volume, args.visual_slice)
    errors = {k: np.abs(preds[k].astype(np.float64) - gt.astype(np.float64)) for k, _ in METHODS}
    # One shared scale for every method: the 99.5th percentile of the bicubic error.
    vmax = float(np.percentile(errors["bicubic"], 99.5))
    panels = [("gt", "Ground truth")] + METHODS
    fig, axes = plt.subplots(2, 4, figsize=(3.5, 1.95), gridspec_kw={"wspace": 0.03, "hspace": 0.12})
    for ax, (key, label) in zip(axes.ravel(), panels):
        if key == "gt":
            ax.imshow(gt, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        else:
            im = ax.imshow(errors[key], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            ax.text(0.97, 0.03, f"MAE {errors[key][gt > 0].mean():.2f}", transform=ax.transAxes, color="white",
                    fontsize=5, ha="right", va="bottom")
        ax.set_title(label, pad=1.5, fontsize=6)
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("|prediction − ground truth| (0–255)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    save(fig, "fig_pixel_error_heatmap", args)
    return {"volume": args.visual_volume, "slice": args.visual_slice, "vmax": round(vmax, 1),
            "mae_foreground": {k: round(float(errors[k][gt > 0].mean()), 3) for k, _ in METHODS}}


def fig_failure_case(args: argparse.Namespace) -> dict:
    cases = [(c.split(":")[0], int(c.split(":")[1])) for c in args.failure_cases]
    loaded = [load_slice(args.pred_dir, v, si) for v, si in cases]
    errors = [{k: np.abs(p[k].astype(np.float64) - gt.astype(np.float64)) for k in ("25d", "edsr")} for gt, _, p in loaded]
    # One error scale for every map in the figure.
    vmax = float(max(np.percentile(e[k], 99.5) for e in errors for k in e))
    fig, axes = plt.subplots(len(cases), 5, figsize=(7.16, 1.55 * len(cases)), squeeze=False,
                             gridspec_kw={"wspace": 0.04, "hspace": 0.08})
    report = {"vmax": round(vmax, 1), "cases": []}
    for row, ((volume, si), (gt, _, preds), err) in enumerate(zip(cases, loaded, errors)):
        fg = gt > 0
        # Display window per row: lateral slices are dim, so stretch them to their own 99.9th percentile.
        window = float(max(np.percentile(gt, 99.9), 1))
        stats = {"volume": volume, "slice": si, "foreground_fraction": round(float(fg.mean()), 3), "display_vmax": round(window)}
        panels = [("Ground truth", gt, None), ("2.5D", preds["25d"], None), ("EDSR", preds["edsr"], None),
                  ("|2.5D − GT|", None, err["25d"]), ("|EDSR − GT|", None, err["edsr"])]
        for ax, (title, img, e) in zip(axes[row], panels):
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=window, interpolation="nearest")
                if title != "Ground truth":
                    key = "25d" if title == "2.5D" else "edsr"
                    stats[key] = {"psnr": round(psnr_uint8(img, gt), 2), "psnr_fg": round(psnr_uint8(img, gt, fg), 2)}
                    ax.text(0.97, 0.03, f"{stats[key]['psnr']:.2f} dB\nfg {stats[key]['psnr_fg']:.2f} dB",
                            transform=ax.transAxes, color="white", fontsize=6, ha="right", va="bottom",
                            bbox=dict(facecolor="black", alpha=0.55, pad=1, edgecolor="none"))
            else:
                im = ax.imshow(e, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            if row == 0:
                ax.set_title(title, pad=2)
            ax.axis("off")
        axes[row, 0].text(0.03, 0.97, f"{volume.split('_ses')[0]}, slice {si}", transform=axes[row, 0].transAxes,
                          color="white", fontsize=6, ha="left", va="top")
        report["cases"].append(stats)
    cbar = fig.colorbar(im, ax=axes[:, 3:].ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("|error| (0–255)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    save(fig, "fig_failure_case", args)
    return report


def fig_per_subject_psnr(args: argparse.Namespace) -> dict:
    per_slice = {k: pd.read_csv(args.csv_dir / name) for k, name in CSV_NAMES.items()}
    per_slice["25d"] = pd.read_csv(args.csv_25d)
    per_subject = pd.DataFrame({k: df.groupby("subject").psnr.mean() for k, df in per_slice.items()})
    assert per_subject.notna().all().all() and len(per_subject) == 70, per_subject.shape
    order = per_subject.sort_values("bicubic").index
    per_subject = per_subject.loc[order]
    # Bootstrap CI for the 2.5D model: resample each subject's slices with replacement.
    rng = np.random.default_rng(0)
    lo, hi = [], []
    for subject in order:
        v = per_slice["25d"].loc[per_slice["25d"].subject == subject, "psnr"].to_numpy()
        means = v[rng.integers(0, len(v), size=(args.bootstrap, len(v)))].mean(axis=1)
        lo.append(np.percentile(means, 2.5)); hi.append(np.percentile(means, 97.5))
    x = np.arange(1, len(order) + 1)
    fig, ax = plt.subplots(figsize=(7.16, 2.6))
    ax.fill_between(x, lo, hi, color=COLORS["25d"], alpha=0.25, linewidth=0, label="2.5D bootstrap 95% CI")
    for key, label in METHODS:
        ax.plot(x, per_subject[key], color=COLORS[key], marker=MARKERS[key], markersize=2.6, linewidth=0.7,
                markerfacecolor="none" if key not in ("25d", "bicubic") else COLORS[key], markeredgewidth=0.6,
                label=label, alpha=0.95 if key in ("25d", "bicubic") else 0.8)
    ax.set_xlabel("Test subject, sorted by bicubic PSNR")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xlim(0.5, len(order) + 0.5)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70])
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(ncol=4, frameon=False, loc="upper left", handlelength=2.2, columnspacing=1.2)
    save(fig, "fig_per_subject_psnr", args)
    width = np.array(hi) - np.array(lo)
    return {"n_subjects": len(order), "bootstrap": args.bootstrap, "ci_width_db": {"min": round(width.min(), 3),
            "median": round(float(np.median(width)), 3), "max": round(width.max(), 3)},
            "subject_order": list(order), "per_subject_mean": per_subject.round(3).to_dict("index")}


def main() -> None:
    args = parse_args()
    report = {"per_subject": fig_per_subject_psnr(args)}
    if not args.per_subject_only:
        report["visual"] = fig_visual_all_methods(args)
        report["error_heatmap"] = fig_pixel_error_heatmap(args)
        report["failure"] = fig_failure_case(args)
    args.png_dir.parent.mkdir(parents=True, exist_ok=True)
    (args.png_dir.parent / "figure_numbers.json").write_text(json.dumps(report, indent=1) + "\n")
    print({k: {kk: vv for kk, vv in v.items() if kk not in ("subject_order", "per_subject_mean")} for k, v in report.items()})


if __name__ == "__main__":
    main()
