#!/usr/bin/env python3
"""Run 3D volumetric upsampler inference and save comparison slices + metrics.

Usage:
    python scripts/run_inference_3d.py \
        --checkpoint runs/run-3d-20260306-052515/checkpoints/model_final.pt \
        --num-samples 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.diffusion.denoiser_3d import Denoiser3d, Denoiser3dConfig
from models.diffusion.inner_model_3d import InnerModel3dConfig
from models.diffusion.denoiser import SigmaDistributionConfig
from data.dataset_3d import MRIVolumeDataset3d
from metrics import psnr as compute_psnr, ssim as compute_ssim, lpips_distance as compute_lpips


def build_model(
    channels: list[int],
    depths: list[int],
    attn_depths: list[int],
    cond_channels: int,
    upsampling_factor: int,
) -> Denoiser3d:
    inner_cfg = InnerModel3dConfig(
        img_channels=1,
        cond_channels=cond_channels,
        depths=depths,
        channels=channels,
        attn_depths=attn_depths,
    )
    denoiser_cfg = Denoiser3dConfig(
        inner_model=inner_cfg,
        sigma_data=0.5,
        sigma_offset_noise=0.1,
        upsampling_factor=upsampling_factor,
    )
    return Denoiser3d(denoiser_cfg)


def save_slice_comparison(gt: np.ndarray, pred: np.ndarray, bicubic: np.ndarray, path: Path) -> None:
    from PIL import Image, ImageDraw

    def to_pil(a: np.ndarray) -> Image.Image:
        a = np.clip(a, 0, 255).astype(np.uint8)
        return Image.fromarray(a, mode="L")

    imgs = [to_pil(gt), to_pil(pred), to_pil(bicubic)]
    gap = 4
    w_total = sum(im.width for im in imgs) + gap * 2
    h_total = max(im.height for im in imgs)
    canvas = Image.new("L", (w_total, h_total + 20), 200)
    draw = ImageDraw.Draw(canvas)
    x = 0
    for im, label in zip(imgs, ["GT HR", "Model SR", "Trilinear"]):
        canvas.paste(im, (x, 20))
        draw.text((x + 2, 2), label, fill=0)
        x += im.width + gap
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--lr-dir", type=Path, default=Path("src/processed_data_mri/low_res/test"))
    p.add_argument("--hr-dir", type=Path, default=Path("src/processed_data_mri/full_res/test"))
    p.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--depths", type=int, nargs="+", default=[2, 2, 2, 2])
    p.add_argument("--attn-depths", type=int, nargs="+", default=[0, 0, 0, 1])
    p.add_argument("--cond-channels", type=int, default=1024)
    p.add_argument("--upsampling-factor", type=int, default=2)
    p.add_argument("--num-samples", type=int, default=3, help="Number of test volumes to evaluate")
    p.add_argument("--num-steps", type=int, default=20, help="Denoising steps for sampling")
    p.add_argument("--patch-size", type=int, default=32, help="Patch size for sliding-window inference")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: next to checkpoint)")
    return p.parse_args()


def patchwise_sample(
    model: Denoiser3d,
    lr_vol: torch.Tensor,
    device: torch.device,
    patch_size: int,
    overlap: int,
    uf: int,
    num_steps: int,
) -> torch.Tensor:
    """Run patch-based 3D inference with overlap blending."""
    _, _, d, h, w = lr_vol.shape
    stride = patch_size - overlap
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

    total_patches = len(d_starts) * len(h_starts) * len(w_starts)
    patch_idx = 0

    for ds in d_starts:
        for hs in h_starts:
            for ws in w_starts:
                de = min(ds + patch_size, d)
                he = min(hs + patch_size, h)
                we = min(ws + patch_size, w)

                lr_patch = lr_vol[:, :, ds:de, hs:he, ws:we].to(device)
                pred_patch = model.sample(lr_patch, num_steps=num_steps).cpu()

                output[:, :, ds:de, hs * uf:he * uf, ws * uf:we * uf] += pred_patch
                counts[:, :, ds:de, hs * uf:he * uf, ws * uf:we * uf] += 1

                patch_idx += 1
                if patch_idx % 10 == 0 or patch_idx == total_patches:
                    print(f"    patch {patch_idx}/{total_patches}")

    counts = counts.clamp(min=1)
    return output / counts


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    model = build_model(args.channels, args.depths, args.attn_depths, args.cond_channels, args.upsampling_factor)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    sigma_cfg = SigmaDistributionConfig(loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=20)
    model.setup_training(sigma_cfg)

    test_ds = MRIVolumeDataset3d(args.lr_dir, args.hr_dir, args.upsampling_factor)
    print(f"Test volumes: {len(test_ds)}")

    out_dir = args.out_dir or (args.checkpoint.parent.parent / "inference_3d")
    out_dir.mkdir(parents=True, exist_ok=True)

    uf = args.upsampling_factor
    patch_size = args.patch_size
    overlap = patch_size // 4
    all_psnr, all_ssim, all_lpips = [], [], []
    all_bic_psnr, all_bic_ssim, all_bic_lpips = [], [], []

    for vol_idx in range(min(args.num_samples, len(test_ds))):
        sample = test_ds[vol_idx]
        lr_vol = sample["lr"].unsqueeze(0)
        hr_vol = sample["hr"].unsqueeze(0)

        print(f"\n[{vol_idx + 1}/{min(args.num_samples, len(test_ds))}] LR={tuple(lr_vol.shape)} HR={tuple(hr_vol.shape)}")
        print(f"  Patch-based inference (patch={patch_size}, overlap={overlap}) ...")

        with torch.no_grad():
            pred_hr = patchwise_sample(model, lr_vol, device, patch_size, overlap, uf, args.num_steps)

        _, _, d, h, w = hr_vol.shape
        trilinear = F.interpolate(lr_vol, size=(d, h, w), mode="trilinear", align_corners=False)

        pred_hr_cpu = pred_hr.cpu()
        hr_vol_cpu = hr_vol.cpu()
        trilinear_cpu = trilinear.cpu()

        slice_psnrs, slice_ssims, slice_lpips = [], [], []
        slice_bpsnrs, slice_bssims, slice_blpips = [], [], []
        for si in range(d):
            p_s = pred_hr_cpu[:, :, si:si+1]
            h_s = hr_vol_cpu[:, :, si:si+1]
            t_s = trilinear_cpu[:, :, si:si+1]
            slice_psnrs.append(compute_psnr(p_s, h_s))
            slice_ssims.append(compute_ssim(p_s, h_s))
            slice_lpips.append(compute_lpips(p_s, h_s))
            slice_bpsnrs.append(compute_psnr(t_s, h_s))
            slice_bssims.append(compute_ssim(t_s, h_s))
            slice_blpips.append(compute_lpips(t_s, h_s))
        p = float(np.mean(slice_psnrs))
        s = float(np.mean(slice_ssims))
        lp = float(np.mean(slice_lpips))
        bp = float(np.mean(slice_bpsnrs))
        bs = float(np.mean(slice_bssims))
        blp = float(np.mean(slice_blpips))
        all_psnr.append(p)
        all_ssim.append(s)
        all_lpips.append(lp)
        all_bic_psnr.append(bp)
        all_bic_ssim.append(bs)
        all_bic_lpips.append(blp)
        print(f"  Model:     PSNR={p:.2f} dB  SSIM={s:.4f}  LPIPS={lp:.4f}")
        print(f"  Trilinear: PSNR={bp:.2f} dB  SSIM={bs:.4f}  LPIPS={blp:.4f}")

        def vol_to_uint8(t):
            return t.clamp(-1, 1).add(1).div(2).mul(255).byte().cpu().squeeze(0).squeeze(0).numpy()

        gt_np = vol_to_uint8(hr_vol)
        pred_np = vol_to_uint8(pred_hr)
        tri_np = vol_to_uint8(trilinear)

        mid_d = d // 2
        mid_h = h // 2
        mid_w = w // 2

        save_slice_comparison(gt_np[mid_d], pred_np[mid_d], tri_np[mid_d], out_dir / f"vol{vol_idx}_axial_mid.png")
        save_slice_comparison(gt_np[:, mid_h], pred_np[:, mid_h], tri_np[:, mid_h], out_dir / f"vol{vol_idx}_coronal_mid.png")
        save_slice_comparison(gt_np[:, :, mid_w], pred_np[:, :, mid_w], tri_np[:, :, mid_w], out_dir / f"vol{vol_idx}_sagittal_mid.png")
        print(f"  Saved slices to {out_dir}/vol{vol_idx}_*.png")

    print(f"\n{'=' * 70}")
    print(f"AVERAGE ({len(all_psnr)} volumes):")
    print(f"  Model:     PSNR={np.mean(all_psnr):.2f} dB  SSIM={np.mean(all_ssim):.4f}  LPIPS={np.mean(all_lpips):.4f}")
    print(f"  Trilinear: PSNR={np.mean(all_bic_psnr):.2f} dB  SSIM={np.mean(all_bic_ssim):.4f}  LPIPS={np.mean(all_bic_lpips):.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
