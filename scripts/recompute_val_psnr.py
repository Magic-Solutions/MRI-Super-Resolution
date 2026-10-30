"""Re-validate saved 2.5D epoch checkpoints with the trainer's sampled validation and the current code.

Replays Trainer.validate_upsampler_sr (same slice selection, shared 20-step Euler sampler, fixed
seed) for every agent_epoch_*.pt in a folder and writes one row per epoch. Runs validated before
the slice-conditioning edge-padding fix logged a lower val PSNR.

Usage:
    python scripts/recompute_val_psnr.py --val <dir with low_res/test and full_res/test> \
        --ckpt-dir <agent_versions dir> --out val_psnr_fixed.csv [--override k=v ...]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import Agent  # noqa: E402
from data import Dataset, MRIHdf5Dataset, SegmentId  # noqa: E402
from metrics import psnr as compute_psnr, ssim as compute_ssim  # noqa: E402
from models.diffusion import EDMSamplerConfig, edm_sample, upsampled_condition  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val", type=Path, required=True, help="Directory holding low_res/test and full_res/test")
    p.add_argument("--ckpt-dir", type=Path, required=True, help="Directory of agent_epoch_*.pt files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config-name", default="trainer_mri")
    p.add_argument("--override", nargs="*", default=[], help="Hydra overrides, e.g. the both-neighbours option")
    p.add_argument("--slices-per-volume", type=int, default=16, help="As evaluation.val_sr_slices_per_volume")
    p.add_argument("--num-steps", type=int, default=20, help="As evaluation.val_sr_num_steps")
    p.add_argument("--seed", type=int, default=0, help="As evaluation.val_seed")
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: the first available)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    up_mode = "bilinear" if device.type == "mps" else "bicubic"  # same rule as the trainer

    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    full_res = MRIHdf5Dataset(args.val / "full_res")
    val = Dataset(args.val / "low_res" / "test", full_res, "val", cache_in_ram=True)
    val.load_from_default_path()

    agent = Agent(instantiate(cfg.agent, num_actions=0))
    upsampler = agent.upsampler
    n_cond = upsampler.cfg.inner_model.num_steps_conditioning
    n_next = upsampler.cfg.inner_model.num_next_slices_conditioning
    up_factor = upsampler.cfg.upsampling_factor
    sampler_cfg = EDMSamplerConfig(num_steps=args.num_steps)

    ckpts = sorted(args.ckpt_dir.glob("agent_epoch_*.pt"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_psnr", "val_ssim", "num_slices"])
        writer.writeheader()
        for ck in ckpts:
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            sd = {k.replace("agent.", ""): v for k, v in sd.items() if k.startswith("agent.")} or sd
            agent.load_state_dict(sd, strict=True)
            upsampler.eval().to(device)
            # The trainer seeds one generator per validation call, so every epoch sees the same noise.
            generator = torch.Generator(device=device).manual_seed(args.seed)
            psnrs, ssims = [], []
            with torch.no_grad():
                for ep_id in range(val.num_episodes):
                    episode = val.load_episode(ep_id)
                    obs_lr = episode.obs.float().div(255).mul(2).sub(1) if episode.obs.dtype == torch.uint8 else episode.obs.float()
                    length = obs_lr.size(0)
                    slice_ids = torch.linspace(0, length - 1, min(args.slices_per_volume, length)).round().long().unique().tolist()
                    for si in slice_ids:
                        cond_frames = upsampled_condition(obs_lr.to(device), si, n_cond, n_next, up_factor, up_mode)
                        _, t, c, h_up, w_up = cond_frames.shape
                        cond = cond_frames.reshape(1, t * c, h_up, w_up)
                        sr = edm_sample(lambda x, sigma: upsampler.denoise(x, sigma, None, cond, None),
                                        torch.Size((1, c, h_up, w_up)), sampler_cfg, device, generator)
                        hr = val[SegmentId(ep_id, si, si + 1)].info["full_res"][:1].to(device)
                        psnrs.append(compute_psnr(sr, hr))
                        ssims.append(compute_ssim(sr, hr))
            upsampler.to("cpu")
            epoch = int(re.search(r"(\d+)", ck.stem).group(1))
            row = {"epoch": epoch, "val_psnr": float(np.mean(psnrs)), "val_ssim": float(np.mean(ssims)), "num_slices": len(psnrs)}
            writer.writerow(row)
            f.flush()
            print(f"epoch {epoch}: val PSNR {row['val_psnr']:.2f} dB, SSIM {row['val_ssim']:.4f} ({len(psnrs)} slices)", flush=True)


if __name__ == "__main__":
    main()
