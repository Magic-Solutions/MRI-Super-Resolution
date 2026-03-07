#!/usr/bin/env python3
"""Generate publication-quality figures for the MRI SR paper.

Reads evaluation results and inference outputs, then produces PDF/PNG
figures in paper_figures/.

Usage:
    python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "paper_figures"

EVAL_25D_CSV = REPO / "runs" / "run-20260306-035738" / "checkpoints" / "eval_results.csv"
INFERENCE_3D_DIR = REPO / "runs" / "run-3d-20260306-052515" / "inference_3d"
INFERENCE_25D_DIR = REPO / "runs" / "run-20260306-035738" / "inference_output"
CKPT_3D = REPO / "runs" / "run-3d-20260306-052515" / "checkpoints" / "model_final.pt"
DATA_DIR = REPO / "src" / "processed_data_mri"

RESULTS_25D = {"psnr": 35.82, "ssim": 0.9706, "lpips": 0.0396}
RESULTS_3D = {"psnr": 37.75, "ssim": 0.9968, "lpips": 0.0202}
BASELINE_BICUBIC = {"psnr": 33.89, "ssim": 0.9572, "lpips": 0.0912}
BASELINE_EDSR = {"psnr": 35.57, "ssim": 0.9774, "lpips": 0.0236}
BASELINE_SWIN2SR = {"psnr": 35.50, "ssim": 0.9776, "lpips": 0.0238}

IEEE_STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}
plt.rcParams.update(IEEE_STYLE)

C_3D = "#1f77b4"
C_25D = "#ff7f0e"
C_BIC = "#2ca02c"
C_EDSR = "#d62728"
C_SWIN = "#9467bd"


def fig_psnr_ssim_bars() -> None:
    methods = ["Bicubic", "EDSR", "Swin2SR", "2.5D EDM\n(ours)", "3D EDM\n(ours)"]
    psnr_vals = [BASELINE_BICUBIC["psnr"], BASELINE_EDSR["psnr"],
                 BASELINE_SWIN2SR["psnr"], RESULTS_25D["psnr"], RESULTS_3D["psnr"]]
    ssim_vals = [BASELINE_BICUBIC["ssim"], BASELINE_EDSR["ssim"],
                 BASELINE_SWIN2SR["ssim"], RESULTS_25D["ssim"], RESULTS_3D["ssim"]]
    lpips_vals = [BASELINE_BICUBIC["lpips"], BASELINE_EDSR["lpips"],
                  BASELINE_SWIN2SR["lpips"], RESULTS_25D["lpips"], RESULTS_3D["lpips"]]
    colors = [C_BIC, C_EDSR, C_SWIN, C_25D, C_3D]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.0, 3.2))

    x = np.arange(len(methods))
    w = 0.55

    bars1 = ax1.bar(x, psnr_vals, color=colors, width=w, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_ylim(31, 40)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=7, rotation=25, ha="right")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, psnr_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    bars2 = ax2.bar(x, ssim_vals, color=colors, width=w, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("SSIM")
    ax2.set_ylim(0.94, 1.005)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=7, rotation=25, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, ssim_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    bars3 = ax3.bar(x, lpips_vals, color=colors, width=w, edgecolor="black", linewidth=0.5)
    ax3.set_ylabel(r"LPIPS $\downarrow$")
    ax3.set_ylim(0, 0.12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, fontsize=7, rotation=25, ha="right")
    ax3.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars3, lpips_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_psnr_ssim_bars.pdf")
    fig.savefig(OUT_DIR / "fig_psnr_ssim_bars.png")
    plt.close(fig)
    print("  -> fig_psnr_ssim_bars.pdf")


def _load_test_volume_pair(vol_idx: int = 0):
    """Load one LR/HR test volume pair and compute trilinear + 3D SR."""
    import torch
    import torch.nn.functional as F
    import h5py

    from data.dataset import Dataset, MRIHdf5Dataset
    from data.dataset_3d import MRIVolumeDataset3d

    lr_dir = DATA_DIR / "low_res" / "test"
    hr_dir = DATA_DIR / "full_res" / "test"

    hr_dataset = MRIHdf5Dataset(hr_dir)
    lr_dataset = Dataset(lr_dir, dataset_full_res=hr_dataset, name="test")
    lr_dataset.load_from_default_path()

    ep = lr_dataset.load_episode(vol_idx)
    obs_lr = ep.obs.float().div(255).mul(2).sub(1) if ep.obs.dtype == torch.uint8 else ep.obs.float()

    file_id = ep.info.get("original_file_id", "").replace("test/", "")
    hr_path = hr_dir / file_id
    gt_slices = None
    if hr_path.exists():
        with h5py.File(hr_path, "r") as f:
            keys = sorted([k for k in f.keys() if k.endswith("_x")], key=lambda k: int(k.split("_")[1]))
            gt_slices = np.stack([np.asarray(f[k]) for k in keys])

    lr_np = obs_lr.squeeze(1).numpy()
    lr_up = F.interpolate(
        obs_lr, scale_factor=2, mode="bilinear"
    ).squeeze(1).numpy()
    lr_up_uint8 = np.clip((lr_up + 1) / 2 * 255, 0, 255).astype(np.uint8)

    trilinear_vol = None
    vol_3d = MRIVolumeDataset3d(lr_dir, hr_dir, 2)
    if len(vol_3d) > vol_idx:
        sample = vol_3d[vol_idx]
        lr_t = sample["lr"].unsqueeze(0)
        hr_t = sample["hr"]
        _, _, d, h, w = hr_t.unsqueeze(0).shape
        tri = F.interpolate(lr_t, size=(d, h, w), mode="trilinear", align_corners=False)
        trilinear_vol = np.clip((tri.squeeze(0).squeeze(0).numpy() + 1) / 2 * 255, 0, 255).astype(np.uint8)

    return gt_slices, lr_up_uint8, trilinear_vol, file_id


def _compute_3d_sr_volume(vol_idx: int = 0):
    """Run 3D model on one test volume and return the SR result as uint8."""
    import torch
    import torch.nn.functional as F

    if not CKPT_3D.exists():
        return None

    from models.diffusion.denoiser_3d import Denoiser3d, Denoiser3dConfig
    from models.diffusion.inner_model_3d import InnerModel3dConfig
    from models.diffusion.denoiser import SigmaDistributionConfig
    from data.dataset_3d import MRIVolumeDataset3d

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    inner_cfg = InnerModel3dConfig(img_channels=1, cond_channels=1024, depths=[2,2,2,2], channels=[32,64,128,256], attn_depths=[0,0,0,1])
    denoiser_cfg = Denoiser3dConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.1, upsampling_factor=2)
    model = Denoiser3d(denoiser_cfg)
    state = torch.load(CKPT_3D, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    sigma_cfg = SigmaDistributionConfig(loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=20)
    model.setup_training(sigma_cfg)

    lr_dir = DATA_DIR / "low_res" / "test"
    hr_dir = DATA_DIR / "full_res" / "test"
    vol_ds = MRIVolumeDataset3d(lr_dir, hr_dir, 2)
    if vol_idx >= len(vol_ds):
        return None

    sample = vol_ds[vol_idx]
    lr_vol = sample["lr"].unsqueeze(0)

    patch_size, overlap, uf = 32, 8, 2
    stride = patch_size - overlap
    _, _, d, h, w = lr_vol.shape
    hr_d, hr_h, hr_w = d, h * uf, w * uf
    output = torch.zeros(1, 1, hr_d, hr_h, hr_w)
    counts = torch.zeros(1, 1, hr_d, hr_h, hr_w)

    d_starts = list(range(0, max(d - patch_size, 0) + 1, stride))
    if d_starts[-1] + patch_size < d:
        d_starts.append(d - patch_size)
    h_starts = list(range(0, max(h - patch_size, 0) + 1, stride))
    if h_starts[-1] + patch_size < h:
        h_starts.append(h - patch_size)
    w_starts = list(range(0, max(w - patch_size, 0) + 1, stride))
    if w_starts[-1] + patch_size < w:
        w_starts.append(w - patch_size)

    total = len(d_starts) * len(h_starts) * len(w_starts)
    pi = 0
    for ds in d_starts:
        for hs in h_starts:
            for ws in w_starts:
                de, he, we = min(ds+patch_size, d), min(hs+patch_size, h), min(ws+patch_size, w)
                lr_patch = lr_vol[:, :, ds:de, hs:he, ws:we].to(device)
                with torch.no_grad():
                    pred = model.sample(lr_patch, num_steps=20).cpu()
                output[:, :, ds:de, hs*uf:he*uf, ws*uf:we*uf] += pred
                counts[:, :, ds:de, hs*uf:he*uf, ws*uf:we*uf] += 1
                pi += 1
                if pi % 20 == 0:
                    print(f"    3D patch {pi}/{total}")

    counts = counts.clamp(min=1)
    sr = (output / counts).squeeze(0).squeeze(0).numpy()
    return np.clip((sr + 1) / 2 * 255, 0, 255).astype(np.uint8)


CACHE_3D = OUT_DIR / "cache_3d_sr_vol0.npy"


def _psnr_uint8(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_f = gt.astype(np.float64) / 255.0 * 2.0 - 1.0
    pred_f = pred.astype(np.float64) / 255.0 * 2.0 - 1.0
    mse = np.mean((gt_f - pred_f) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(4.0 / mse))


def _load_3d_cache() -> np.ndarray | None:
    if CACHE_3D.exists():
        return np.load(CACHE_3D)
    return None


def fig_per_slice_psnr() -> None:
    if not EVAL_25D_CSV.exists():
        print("  SKIP fig_per_slice_psnr (no CSV)")
        return

    subject_data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with open(EVAL_25D_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row["subject"]
            subject_data[subj]["slice"].append(int(row["slice"]))
            subject_data[subj]["model_psnr"].append(float(row["model_psnr"]))

    first_subj = sorted(subject_data.keys())[0]
    d = subject_data[first_subj]
    slices = np.array(d["slice"])
    model_25d_psnr = np.array(d["model_psnr"])

    print("    Loading trilinear + 3D per-slice data...")
    gt_slices, lr_up, tri_vol, _ = _load_test_volume_pair(0)
    sr_3d = _load_3d_cache()

    trilinear_psnr, psnr_3d = [], []
    for si in slices:
        if gt_slices is not None and tri_vol is not None and si < len(gt_slices) and si < len(tri_vol):
            trilinear_psnr.append(_psnr_uint8(gt_slices[si], tri_vol[si]))
        else:
            trilinear_psnr.append(float("nan"))
        if gt_slices is not None and sr_3d is not None and si < len(gt_slices) and si < len(sr_3d):
            psnr_3d.append(_psnr_uint8(gt_slices[si], sr_3d[si]))
        else:
            psnr_3d.append(float("nan"))
    trilinear_psnr = np.array(trilinear_psnr)
    psnr_3d = np.array(psnr_3d)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    if not np.all(np.isnan(psnr_3d)):
        ax.plot(slices, psnr_3d, color=C_3D, linewidth=1.4, label="3D EDM (ours)", alpha=0.9)
    ax.plot(slices, model_25d_psnr, color=C_25D, linewidth=1.2, label="2.5D EDM (ours)", alpha=0.85)
    ax.plot(slices, trilinear_psnr, color="#8c564b", linewidth=1.2, label="Trilinear", alpha=0.85, linestyle="--")
    ax.set_xlabel("Sagittal slice index")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(loc="lower center", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    ax.set_title(f"Per-slice PSNR: {first_subj}", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_per_slice_psnr.pdf")
    fig.savefig(OUT_DIR / "fig_per_slice_psnr.png")
    plt.close(fig)
    print("  -> fig_per_slice_psnr.pdf")


def fig_visual_comparison_3d() -> None:
    if not INFERENCE_3D_DIR.exists():
        print("  SKIP fig_visual_comparison_3d (no inference dir)")
        return

    views = ["sagittal_mid", "axial_mid", "coronal_mid"]
    view_labels = ["Sagittal", "Axial", "Coronal"]
    n_vols = min(2, len(list(INFERENCE_3D_DIR.glob("vol*_sagittal_mid.png"))))

    if n_vols == 0:
        print("  SKIP fig_visual_comparison_3d (no images)")
        return

    total_rows = n_vols * 3
    fig, axes = plt.subplots(total_rows, 1, figsize=(6.5, 2.5 * total_rows))

    row = 0
    for vi in range(n_vols):
        for ci, (view, vlabel) in enumerate(zip(views, view_labels)):
            img_path = INFERENCE_3D_DIR / f"vol{vi}_{view}.png"
            if img_path.exists():
                img = np.array(Image.open(img_path).convert("L"))
                axes[row].imshow(img, cmap="gray", aspect="auto")
            axes[row].axis("off")
            axes[row].set_title(f"Vol {vi} - {vlabel}  (GT HR | Model SR | Trilinear)", fontsize=9)
            row += 1

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_visual_comparison_3d.pdf")
    fig.savefig(OUT_DIR / "fig_visual_comparison_3d.png")
    plt.close(fig)
    print("  -> fig_visual_comparison_3d.pdf")


def fig_visual_comparison_25d() -> None:
    if not INFERENCE_25D_DIR.exists():
        print("  SKIP fig_visual_comparison_25d (no inference dir)")
        return

    samples = sorted(INFERENCE_25D_DIR.glob("sample_*.png"))[:3]
    if not samples:
        print("  SKIP fig_visual_comparison_25d (no images)")
        return

    fig, axes = plt.subplots(1, len(samples), figsize=(6.5, 2.5))
    if len(samples) == 1:
        axes = [axes]

    for i, sp in enumerate(samples):
        img = np.array(Image.open(sp).convert("L"))
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        name = sp.stem.replace("sample_", "S")
        axes[i].set_title(name, fontsize=8)

    fig.suptitle("2.5D Model: GT HR | Generated HR | Bicubic LR  (per panel)", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_visual_comparison_25d.pdf")
    fig.savefig(OUT_DIR / "fig_visual_comparison_25d.png")
    plt.close(fig)
    print("  -> fig_visual_comparison_25d.pdf")


def fig_comparison_table_literature() -> None:
    """Generate a visual comparison table as a figure with actual measured results."""
    data = [
        ("Bicubic interp.", "--", "33.89", "0.957", "0.091", "--"),
        ("EDSR (DIV2K)", "2D CNN", "35.57", "0.977", "0.024", "1.4M"),
        ("Swin2SR (DIV2K)", "Transf.", "35.50", "0.978", "0.024", "1.0M"),
        ("2.5D EDM (ours)", "2.5D Diff.", "35.82", "0.971", "0.040", "51.1M"),
        ("3D EDM (ours)", "3D Diff.", "37.77", "0.996", "0.029", "50.7M"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.axis("off")
    col_labels = ["Method", "Type", "PSNR (dB)", "SSIM", "LPIPS", "Params"]
    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row > 0 and "ours" in str(data[row - 1][0]):
            cell.set_facecolor("#D6E4F0")
        cell.set_edgecolor("#CCCCCC")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_comparison_table.pdf")
    fig.savefig(OUT_DIR / "fig_comparison_table.png")
    plt.close(fig)
    print("  -> fig_comparison_table.pdf")


def fig_error_heatmap() -> None:
    """Generate error heatmap comparing methods on a sample slice."""
    if not EVAL_25D_CSV.exists():
        print("  SKIP fig_error_heatmap (no CSV data)")
        return

    subject_data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with open(EVAL_25D_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row["subject"]
            subject_data[subj]["model_psnr"].append(float(row["model_psnr"]))
            subject_data[subj]["bicubic_psnr"].append(float(row["bicubic_psnr"]))

    subjects = sorted(subject_data.keys())
    subj_means_model = [np.mean(subject_data[s]["model_psnr"]) for s in subjects]
    subj_means_bic = [np.mean(subject_data[s]["bicubic_psnr"]) for s in subjects]
    improvement = [m - b for m, b in zip(subj_means_model, subj_means_bic)]

    short_names = [s.replace("_t1_sagittal", "").replace("sub_", "S") for s in subjects]

    fig, ax = plt.subplots(figsize=(5, 2.5))
    x = np.arange(len(subjects))
    width = 0.35
    ax.bar(x - width / 2, subj_means_bic, width, label="Bicubic", color=C_BIC, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, subj_means_model, width, label="2.5D EDM (ours)", color=C_25D, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Per-subject PSNR: 2.5D Model vs Bicubic", fontsize=9)

    for i, imp in enumerate(improvement):
        ax.annotate(f"+{imp:.1f}", (x[i] + width / 2, subj_means_model[i] + 0.15),
                    ha="center", fontsize=6, color="darkred", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_per_subject_psnr.pdf")
    fig.savefig(OUT_DIR / "fig_per_subject_psnr.png")
    plt.close(fig)
    print("  -> fig_per_subject_psnr.pdf")


def fig_method_overview() -> None:
    """Generate a simple method overview diagram showing 2.5D vs 3D pipeline."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.0))

    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2)
        ax.axis("off")

    # 2.5D pipeline
    ax1.set_title("(a) 2.5D Slice-Conditioned EDM Pipeline", fontsize=10, fontweight="bold", loc="left")
    boxes_25d = [
        (0.2, 0.5, 1.5, 1.0, "LR Volume\n128x128xD", "#E8E8E8"),
        (2.2, 0.5, 1.8, 1.0, "Extract Slice\n+ Neighbor", "#FFF3CD"),
        (4.5, 0.5, 2.0, 1.0, "2D UNet\n(EDM Denoiser)\n64/64/128/256", "#D4EDDA"),
        (7.0, 0.5, 1.5, 1.0, "Heun Step\n(1-step)", "#D1ECF1"),
        (8.8, 0.5, 1.0, 1.0, "HR Slice\n256x256", "#F8D7DA"),
    ]
    for x, y, w, h, text, color in boxes_25d:
        ax1.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=0.8, zorder=2))
        ax1.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, zorder=3)
    for i in range(len(boxes_25d) - 1):
        x1 = boxes_25d[i][0] + boxes_25d[i][2]
        x2 = boxes_25d[i + 1][0]
        y_mid = 1.0
        ax1.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # 3D pipeline
    ax2.set_title("(b) 3D Volumetric EDM Pipeline", fontsize=10, fontweight="bold", loc="left")
    boxes_3d = [
        (0.2, 0.5, 1.5, 1.0, "LR Volume\n128x128xD", "#E8E8E8"),
        (2.2, 0.5, 1.8, 1.0, "Extract 3D\nPatch 32^3", "#FFF3CD"),
        (4.5, 0.5, 2.0, 1.0, "3D UNet\n(EDM Denoiser)\n32/64/128/256", "#D4EDDA"),
        (7.0, 0.5, 1.5, 1.0, "Euler Steps\n(20-step)", "#D1ECF1"),
        (8.8, 0.5, 1.0, 1.0, "HR Patch\n64x64xD", "#F8D7DA"),
    ]
    for x, y, w, h, text, color in boxes_3d:
        ax2.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=0.8, zorder=2))
        ax2.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, zorder=3)
    for i in range(len(boxes_3d) - 1):
        x1 = boxes_3d[i][0] + boxes_3d[i][2]
        x2 = boxes_3d[i + 1][0]
        y_mid = 1.0
        ax2.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_method_overview.pdf")
    fig.savefig(OUT_DIR / "fig_method_overview.png")
    plt.close(fig)
    print("  -> fig_method_overview.pdf")


def fig_pixel_error_heatmap() -> None:
    """Per-pixel absolute error heatmap comparing all methods on a mid-sagittal slice."""
    print("    Loading test volume for error maps...")
    gt_slices, lr_up, tri_vol, file_id = _load_test_volume_pair(0)
    sr_3d = _load_3d_cache()

    if gt_slices is None:
        print("  SKIP fig_pixel_error_heatmap (no GT)")
        return

    mid = len(gt_slices) // 2
    gt = gt_slices[mid].astype(np.float64)

    bicubic = lr_up[mid].astype(np.float64)
    err_bicubic = np.abs(gt - bicubic)

    err_trilinear = None
    if tri_vol is not None and mid < len(tri_vol):
        err_trilinear = np.abs(gt - tri_vol[mid].astype(np.float64))

    err_3d = None
    if sr_3d is not None and mid < len(sr_3d):
        err_3d = np.abs(gt - sr_3d[mid].astype(np.float64))

    n_panels = 4 + (1 if err_3d is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.0 * n_panels, 2.5))

    axes[0].imshow(gt.astype(np.uint8), cmap="gray")
    axes[0].set_title("GT HR", fontsize=8)
    axes[0].axis("off")

    vmax = 60
    axes[1].imshow(err_bicubic, cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title("Bicubic err", fontsize=8)
    axes[1].axis("off")

    if err_trilinear is not None:
        axes[2].imshow(err_trilinear, cmap="hot", vmin=0, vmax=vmax)
        axes[2].set_title("Trilinear err", fontsize=8)
    axes[2].axis("off")

    import torch, torch.nn.functional as F
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from agent import Agent
    from data.dataset import Dataset, MRIHdf5Dataset
    from envs.world_model_env import DiffusionSampler

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config_dir = str(REPO / "config")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="trainer_mri")
    agent = Agent(instantiate(cfg.agent, num_actions=0)).to(device)
    ckpt_25d = REPO / "runs" / "run-20260306-035738" / "checkpoints" / "agent_epoch_00010.pt"
    if ckpt_25d.exists():
        ckpt = torch.load(ckpt_25d, map_location=device, weights_only=False)
        agent_state = {k.replace("agent.", ""): v for k, v in ckpt.items() if k.startswith("agent.")}
        if not agent_state:
            agent_state = ckpt
        agent.load_state_dict(agent_state, strict=False)
    agent.eval()

    upsampler = agent.upsampler
    up_factor = upsampler.cfg.upsampling_factor
    n_cond = cfg.agent.upsampler.inner_model.num_steps_conditioning
    sampler_cfg = instantiate(cfg.world_model_env.diffusion_sampler_upsampling)
    sampler_cfg.num_steps_denoising = 1
    sampler = DiffusionSampler(upsampler, sampler_cfg)

    lr_dir = DATA_DIR / "low_res" / "test"
    hr_dir = DATA_DIR / "full_res" / "test"
    hr_dataset = MRIHdf5Dataset(hr_dir)
    lr_dataset = Dataset(lr_dir, dataset_full_res=hr_dataset, name="test")
    lr_dataset.load_from_default_path()
    ep = lr_dataset.load_episode(0)
    obs_lr = ep.obs.float().div(255).mul(2).sub(1) if ep.obs.dtype == torch.uint8 else ep.obs.float()

    ctx_start = max(0, mid - n_cond)
    ctx = obs_lr[ctx_start:mid].unsqueeze(0).to(device)
    if ctx.size(1) < n_cond:
        ctx = F.pad(ctx, [0,0,0,0,0,0, n_cond - ctx.size(1), 0])
    tgt = obs_lr[mid].unsqueeze(0).to(device)
    _, c, h, w = tgt.shape
    tgt_up = F.interpolate(tgt, scale_factor=up_factor, mode="bilinear").unsqueeze(1)
    ctx_up = F.interpolate(ctx.reshape(-1,c,h,w), scale_factor=up_factor, mode="bilinear").reshape(1,n_cond,c,up_factor*h,up_factor*w)
    prev = torch.cat((ctx_up, tgt_up), dim=1)
    with torch.no_grad():
        sr_frame, _ = sampler.sample(prev, None)
    sr_np = np.clip((sr_frame[0].cpu().squeeze(0).numpy() + 1) / 2 * 255, 0, 255)
    err_25d = np.abs(gt - sr_np)

    axes[3].imshow(err_25d, cmap="hot", vmin=0, vmax=vmax)
    axes[3].set_title("2.5D EDM err", fontsize=8)
    axes[3].axis("off")

    if err_3d is not None and n_panels == 5:
        im = axes[4].imshow(err_3d, cmap="hot", vmin=0, vmax=vmax)
        axes[4].set_title("3D EDM err", fontsize=8)
        axes[4].axis("off")
    else:
        im = axes[3].images[0]

    fig.suptitle(f"Per-pixel absolute error (slice {mid})", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_pixel_error_heatmap.pdf")
    fig.savefig(OUT_DIR / "fig_pixel_error_heatmap.png")
    plt.close(fig)
    print("  -> fig_pixel_error_heatmap.pdf")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures in {OUT_DIR}/\n")

    print("[1/7] PSNR/SSIM bar charts...")
    fig_psnr_ssim_bars()

    print("[2/7] Per-slice PSNR curve (with trilinear)...")
    fig_per_slice_psnr()

    print("[3/7] 3D visual comparison...")
    fig_visual_comparison_3d()

    print("[4/7] 2.5D visual comparison...")
    fig_visual_comparison_25d()

    print("[5/7] Per-subject PSNR comparison...")
    fig_error_heatmap()

    print("[6/7] Method overview diagram...")
    fig_method_overview()

    print("[7/7] Per-pixel error heatmap...")
    fig_pixel_error_heatmap()

    print(f"\nDone! All figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
