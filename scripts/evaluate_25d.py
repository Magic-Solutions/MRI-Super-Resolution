#!/usr/bin/env python3
"""Evaluate the 2.5D upsampler on the test set and report PSNR + SSIM.

Runs the upsampler on every test slice, computes per-subject and averaged
metrics, and optionally saves a CSV.

Usage:
    python scripts/evaluate_25d.py \
        --checkpoint outputs/.../checkpoints/agent_versions/agent_epoch_00001.pt
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

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from agent import Agent
from data.dataset import Dataset, MRIHdf5Dataset
from envs.world_model_env import DiffusionSampler
from metrics import psnr as compute_psnr, ssim as compute_ssim, lpips_distance as compute_lpips


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("src/processed_data_mri"))
    p.add_argument("--config-name", default="trainer_mri")
    p.add_argument("--num-steps", type=int, default=20, help="Denoising steps for sampling")
    p.add_argument("--csv-out", type=Path, default=None, help="Save results CSV (default: eval_results.csv next to checkpoint)")
    p.add_argument("--skip-slices", type=int, default=0, help="Skip first/last N slices per volume (often blank)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    config_dir = str(Path(__file__).resolve().parent.parent / "config")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=args.config_name)

    agent_cfg = instantiate(cfg.agent, num_actions=0)
    agent = Agent(agent_cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent_state = {k.replace("agent.", ""): v for k, v in ckpt.items() if k.startswith("agent.")}
    if not agent_state:
        agent_state = ckpt
    agent.load_state_dict(agent_state, strict=False)
    agent.eval()

    upsampler = agent.upsampler
    if upsampler is None:
        raise RuntimeError("No upsampler found in agent")

    up_factor = upsampler.cfg.upsampling_factor
    n_cond = cfg.agent.upsampler.inner_model.num_steps_conditioning
    up_mode = "bilinear" if device.type == "mps" else "bicubic"

    sampler_cfg = instantiate(cfg.world_model_env.diffusion_sampler_upsampling)
    sampler_cfg.num_steps_denoising = args.num_steps
    sampler = DiffusionSampler(upsampler, sampler_cfg)

    lr_dir = args.data_dir / "low_res" / "test"
    hr_dir = args.data_dir / "full_res" / "test"
    hr_dataset = MRIHdf5Dataset(hr_dir)
    lr_dataset = Dataset(lr_dir, dataset_full_res=hr_dataset, name="test")
    lr_dataset.load_from_default_path()
    print(f"Test set: {lr_dataset.num_episodes} episodes, {lr_dataset.num_steps} slices\n")

    if lr_dataset.num_episodes == 0:
        print("No test episodes found. Run process_test_subjects.py first.")
        return

    skip = args.skip_slices
    results: list[dict] = []
    subject_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

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

        gt_slices = None
        if hr_path.exists():
            with h5py.File(hr_path, "r") as f:
                keys = sorted([k for k in f.keys() if k.endswith("_x")], key=lambda k: int(k.split("_")[1]))
                gt_slices = [np.asarray(f[k]) for k in keys]

        slice_start = skip
        slice_end = t_total - skip
        n_eval = slice_end - slice_start

        print(f"[{ep_id + 1}/{lr_dataset.num_episodes}] {subject_name}: {n_eval} slices")

        for si in range(slice_start, slice_end):
            context_start = max(0, si - n_cond)
            context = obs_lr[context_start:si].unsqueeze(0).to(device)
            if context.size(1) < n_cond:
                pad = n_cond - context.size(1)
                context = F.pad(context, [0, 0, 0, 0, 0, 0, pad, 0])

            target_lr = obs_lr[si].unsqueeze(0).to(device)
            _, c, h, w = target_lr.shape

            target_lr_up = F.interpolate(target_lr, scale_factor=up_factor, mode=up_mode).unsqueeze(1)
            context_up = F.interpolate(
                context.reshape(-1, c, h, w), scale_factor=up_factor, mode=up_mode,
            ).reshape(1, n_cond, c, up_factor * h, up_factor * w)
            prev = torch.cat((context_up, target_lr_up), dim=1)

            with torch.no_grad():
                sr_frame, _ = sampler.sample(prev, None)

            bicubic = F.interpolate(target_lr, scale_factor=up_factor, mode=up_mode)

            if gt_slices is not None and si < len(gt_slices):
                gt_np = gt_slices[si]
                gt_tensor = torch.from_numpy(gt_np).float().div(255).mul(2).sub(1).unsqueeze(0).unsqueeze(0).to(device)

                model_psnr = compute_psnr(sr_frame, gt_tensor)
                model_ssim = compute_ssim(sr_frame, gt_tensor)
                model_lpips = compute_lpips(sr_frame, gt_tensor)
                bicubic_psnr = compute_psnr(bicubic, gt_tensor)
                bicubic_ssim = compute_ssim(bicubic, gt_tensor)
                bicubic_lpips = compute_lpips(bicubic, gt_tensor)
            else:
                model_psnr = model_ssim = model_lpips = float("nan")
                bicubic_psnr = bicubic_ssim = bicubic_lpips = float("nan")

            results.append({
                "subject": subject_name,
                "slice": si,
                "model_psnr": model_psnr,
                "model_ssim": model_ssim,
                "model_lpips": model_lpips,
                "bicubic_psnr": bicubic_psnr,
                "bicubic_ssim": bicubic_ssim,
                "bicubic_lpips": bicubic_lpips,
            })
            subject_metrics[subject_name]["model_psnr"].append(model_psnr)
            subject_metrics[subject_name]["model_ssim"].append(model_ssim)
            subject_metrics[subject_name]["model_lpips"].append(model_lpips)
            subject_metrics[subject_name]["bicubic_psnr"].append(bicubic_psnr)
            subject_metrics[subject_name]["bicubic_ssim"].append(bicubic_ssim)
            subject_metrics[subject_name]["bicubic_lpips"].append(bicubic_lpips)

    elapsed = time.time() - t0

    print(f"\n{'=' * 100}")
    print(f"{'Subject':<40} {'Mdl PSNR':>9} {'Mdl SSIM':>9} {'Mdl LPIPS':>10} {'Bic PSNR':>9} {'Bic SSIM':>9} {'Bic LPIPS':>10}")
    print(f"{'-' * 100}")

    all_model_psnr, all_model_ssim, all_model_lpips = [], [], []
    all_bic_psnr, all_bic_ssim, all_bic_lpips = [], [], []

    for subj in sorted(subject_metrics.keys()):
        m = subject_metrics[subj]
        mp = np.nanmean(m["model_psnr"])
        ms = np.nanmean(m["model_ssim"])
        ml = np.nanmean(m["model_lpips"])
        bp = np.nanmean(m["bicubic_psnr"])
        bs = np.nanmean(m["bicubic_ssim"])
        bl = np.nanmean(m["bicubic_lpips"])
        print(f"{subj:<40} {mp:>9.2f} {ms:>9.4f} {ml:>10.4f} {bp:>9.2f} {bs:>9.4f} {bl:>10.4f}")
        all_model_psnr.extend(m["model_psnr"])
        all_model_ssim.extend(m["model_ssim"])
        all_model_lpips.extend(m["model_lpips"])
        all_bic_psnr.extend(m["bicubic_psnr"])
        all_bic_ssim.extend(m["bicubic_ssim"])
        all_bic_lpips.extend(m["bicubic_lpips"])

    print(f"{'-' * 100}")
    print(
        f"{'AVERAGE':<40} "
        f"{np.nanmean(all_model_psnr):>9.2f} "
        f"{np.nanmean(all_model_ssim):>9.4f} "
        f"{np.nanmean(all_model_lpips):>10.4f} "
        f"{np.nanmean(all_bic_psnr):>9.2f} "
        f"{np.nanmean(all_bic_ssim):>9.4f} "
        f"{np.nanmean(all_bic_lpips):>10.4f}"
    )
    print(f"{'=' * 100}")
    print(f"\nEvaluated {len(results)} slices in {elapsed:.1f}s ({elapsed / max(len(results), 1):.2f}s/slice)")

    csv_path = args.csv_out or (args.checkpoint.parent / "eval_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject", "slice",
            "model_psnr", "model_ssim", "model_lpips",
            "bicubic_psnr", "bicubic_ssim", "bicubic_lpips",
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
