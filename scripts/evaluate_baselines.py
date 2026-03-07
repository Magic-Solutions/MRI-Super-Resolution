#!/usr/bin/env python3
"""Evaluate pretrained baseline SR models on the MRI test set.

Runs EDSR (DIV2K) and Swin2SR (DIV2K) on every test slice alongside
bicubic and trilinear baselines.  Reports PSNR, SSIM, and LPIPS.

Usage:
    python scripts/evaluate_baselines.py
    python scripts/evaluate_baselines.py --data-dir src/processed_data_mri --csv-out baselines.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import psnr as compute_psnr, ssim as compute_ssim, lpips_distance as compute_lpips


def load_edsr(device: torch.device):
    from super_image import EdsrModel
    model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2).to(device).eval()
    return model


def load_swin2sr(device: torch.device):
    from transformers import Swin2SRForImageSuperResolution
    model = Swin2SRForImageSuperResolution.from_pretrained(
        "caidas/swin2SR-lightweight-x2-64"
    ).to(device).eval()
    return model


def run_edsr(model, lr_slice: torch.Tensor) -> torch.Tensor:
    """Run EDSR on a single LR slice. Input/output in [-1, 1], shape (1,1,H,W)."""
    img_01 = lr_slice.add(1).div(2)
    img_3ch = img_01.repeat(1, 3, 1, 1)
    with torch.no_grad():
        out = model(img_3ch)
    out_gray = out.mean(dim=1, keepdim=True)
    return out_gray.mul(2).sub(1).clamp(-1, 1)


def run_swin2sr(model, lr_slice: torch.Tensor) -> torch.Tensor:
    """Run Swin2SR on a single LR slice. Input/output in [-1, 1], shape (1,1,H,W)."""
    img_01 = lr_slice.add(1).div(2)
    img_3ch = img_01.repeat(1, 3, 1, 1)
    with torch.no_grad():
        out = model(img_3ch).reconstruction
    out_gray = out.mean(dim=1, keepdim=True)
    return out_gray.mul(2).sub(1).clamp(-1, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("src/processed_data_mri"))
    p.add_argument("--csv-out", type=Path, default=Path("baseline_results.csv"))
    p.add_argument("--skip-slices", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    print("Loading EDSR …")
    edsr = load_edsr(device)
    print("Loading Swin2SR …")
    swin2sr = load_swin2sr(device)

    hr_dir = args.data_dir / "full_res" / "test"
    lr_dir = args.data_dir / "low_res" / "test"

    hdf5_files = sorted(hr_dir.glob("*.hdf5"))
    if not hdf5_files:
        print(f"No HDF5 files in {hr_dir}")
        return
    print(f"Found {len(hdf5_files)} test volumes\n")

    from data.dataset import Dataset, MRIHdf5Dataset
    hr_dataset = MRIHdf5Dataset(hr_dir)
    lr_dataset = Dataset(lr_dir, dataset_full_res=hr_dataset, name="test")
    lr_dataset.load_from_default_path()

    methods = ["bicubic", "trilinear", "edsr", "swin2sr"]
    results: list[dict] = []
    subject_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    up_mode = "bilinear" if device.type == "mps" else "bicubic"
    t0 = time.time()

    for ep_id in range(lr_dataset.num_episodes):
        ep = lr_dataset.load_episode(ep_id)
        obs_lr = ep.obs.float().div(255).mul(2).sub(1) if ep.obs.dtype == torch.uint8 else ep.obs.float()
        t_total = obs_lr.size(0)

        file_id = ep.info.get("original_file_id", f"episode_{ep_id}")
        subject_name = file_id.split("/")[-1].replace(".hdf5", "")

        hr_file_id = file_id
        if hr_file_id.startswith("test/"):
            hr_file_id = hr_file_id[len("test/"):]
        hr_path = hr_dir / hr_file_id
        if not hr_path.exists():
            print(f"  Skipping {subject_name}: HR file not found at {hr_path}")
            continue

        with h5py.File(hr_path, "r") as f:
            keys = sorted([k for k in f.keys() if k.endswith("_x")], key=lambda k: int(k.split("_")[1]))
            gt_slices = [np.asarray(f[k]) for k in keys]

        skip = args.skip_slices
        slice_start = skip
        slice_end = t_total - skip
        n_eval = slice_end - slice_start
        print(f"[{ep_id + 1}/{lr_dataset.num_episodes}] {subject_name}: {n_eval} slices")

        for si in range(slice_start, slice_end):
            target_lr = obs_lr[si].unsqueeze(0).to(device)

            if si >= len(gt_slices):
                continue
            gt_np = gt_slices[si]
            gt_tensor = torch.from_numpy(gt_np).float().div(255).mul(2).sub(1).unsqueeze(0).unsqueeze(0).to(device)

            bicubic_sr = F.interpolate(target_lr, scale_factor=2, mode=up_mode)
            trilinear_lr_3d = target_lr.unsqueeze(2)
            _, _, _, gh, gw = gt_tensor.unsqueeze(2).shape
            trilinear_sr = F.interpolate(
                trilinear_lr_3d, size=(1, gh, gw), mode="trilinear", align_corners=False
            ).squeeze(2)

            edsr_sr = run_edsr(edsr, target_lr)
            swin2sr_sr = run_swin2sr(swin2sr, target_lr)

            preds = {
                "bicubic": bicubic_sr,
                "trilinear": trilinear_sr,
                "edsr": edsr_sr,
                "swin2sr": swin2sr_sr,
            }

            row = {"subject": subject_name, "slice": si}
            for method, pred in preds.items():
                pred = pred[:, :, :gt_tensor.size(2), :gt_tensor.size(3)]
                p = compute_psnr(pred, gt_tensor)
                s = compute_ssim(pred, gt_tensor)
                lp = compute_lpips(pred, gt_tensor)
                row[f"{method}_psnr"] = p
                row[f"{method}_ssim"] = s
                row[f"{method}_lpips"] = lp
                subject_metrics[subject_name][f"{method}_psnr"].append(p)
                subject_metrics[subject_name][f"{method}_ssim"].append(s)
                subject_metrics[subject_name][f"{method}_lpips"].append(lp)

            results.append(row)

    elapsed = time.time() - t0

    header = f"{'Subject':<35}"
    for m in methods:
        header += f" {m:>10} PSNR {m:>10} SSIM {m:>10} LPIPS"
    print(f"\n{'=' * 180}")
    print(f"{'Subject':<35}", end="")
    for m in methods:
        print(f"  |  {m.upper():>8} PSNR  SSIM   LPIPS", end="")
    print()
    print("-" * 180)

    global_metrics: dict[str, list[float]] = defaultdict(list)
    for subj in sorted(subject_metrics.keys()):
        sm = subject_metrics[subj]
        print(f"{subj:<35}", end="")
        for m in methods:
            p = np.nanmean(sm[f"{m}_psnr"])
            s = np.nanmean(sm[f"{m}_ssim"])
            lp = np.nanmean(sm[f"{m}_lpips"])
            print(f"  |  {p:>8.2f} {s:.4f} {lp:.4f}", end="")
            global_metrics[f"{m}_psnr"].extend(sm[f"{m}_psnr"])
            global_metrics[f"{m}_ssim"].extend(sm[f"{m}_ssim"])
            global_metrics[f"{m}_lpips"].extend(sm[f"{m}_lpips"])
        print()

    print("-" * 180)
    print(f"{'AVERAGE':<35}", end="")
    for m in methods:
        p = np.nanmean(global_metrics[f"{m}_psnr"])
        s = np.nanmean(global_metrics[f"{m}_ssim"])
        lp = np.nanmean(global_metrics[f"{m}_lpips"])
        print(f"  |  {p:>8.2f} {s:.4f} {lp:.4f}", end="")
    print(f"\n{'=' * 180}")
    print(f"\nEvaluated {len(results)} slices in {elapsed:.1f}s")

    print("\n\n=== PAPER-READY TABLE ===")
    print(f"{'Method':<25} {'PSNR (dB)':>10} {'SSIM':>8} {'LPIPS':>8}")
    print("-" * 55)
    for m in methods:
        p = np.nanmean(global_metrics[f"{m}_psnr"])
        s = np.nanmean(global_metrics[f"{m}_ssim"])
        lp = np.nanmean(global_metrics[f"{m}_lpips"])
        label = {
            "bicubic": "Bicubic interp.",
            "trilinear": "Trilinear interp.",
            "edsr": "EDSR (DIV2K pretrained)",
            "swin2sr": "Swin2SR (DIV2K pretrained)",
        }[m]
        print(f"{label:<25} {p:>10.2f} {s:>8.4f} {lp:>8.4f}")

    csv_path = args.csv_out
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject", "slice"]
    for m in methods:
        fieldnames += [f"{m}_psnr", f"{m}_ssim", f"{m}_lpips"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
