#!/usr/bin/env python3
"""Recompute the 2.5D validation denoising loss for saved epoch checkpoints.

The runs logged val_loss_denoising through DatasetTraverser while make_segment handed every
segment the episode's own info dict, so each chunk's full_res overwrote the previous one's and
every chunk of a volume was scored against the last chunk's HR slices. This replays
Trainer.test_upsampler (eval mode, fixed seed, the same chunking and batch size) with the fix,
one row per checkpoint, so the convergence curve can show a validation loss that is comparable
with the training loss. The sampled val PSNR in val_metrics.csv was never affected.

Usage:
    python scripts/recompute_val_loss.py --val <dir with low_res/test and full_res/test> \
        --ckpt-dir <agent_versions dir> --out val_loss_fixed.csv [--override k=v ...]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import Agent  # noqa: E402
from data import Dataset, MRIHdf5Dataset  # noqa: E402
from data.utils import DatasetTraverser  # noqa: E402
from trainer import fixed_rng  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val", type=Path, required=True, help="Directory holding low_res/test and full_res/test")
    p.add_argument("--ckpt-dir", type=Path, required=True, help="Directory of agent_epoch_*.pt files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config-name", default="trainer_mri")
    p.add_argument("--override", nargs="*", default=[], help="Hydra overrides, e.g. the both-neighbours option")
    p.add_argument("--batch-size", type=int, default=4, help="Segments per batch, as in training")
    p.add_argument("--val-seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "mps":
        # Denoiser.forward upsamples the conditioning bilinearly on MPS; training used bicubic on CUDA.
        # Keep bicubic here (it falls back to the CPU with PYTORCH_ENABLE_MPS_FALLBACK=1), or the
        # numbers are about 3 dB worse than the run's and not comparable with its training loss.
        _interpolate = F.interpolate
        F.interpolate = lambda *a, **k: _interpolate(*a, **{**k, "mode": "bicubic"}) if k.get("mode") == "bilinear" else _interpolate(*a, **k)
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    inner = cfg.agent.upsampler.inner_model
    chunk = inner.num_steps_conditioning + 1 + cfg.upsampler.training.num_autoregressive_steps + getattr(inner, "num_next_slices_conditioning", 0)

    full_res = MRIHdf5Dataset(args.val / "full_res")
    val = Dataset(args.val / "low_res" / "test", full_res, "val", cache_in_ram=True)
    val.load_from_default_path()
    agent = Agent(instantiate(cfg.agent, num_actions=0)).to(device)
    agent.setup_training(instantiate(cfg.upsampler.sigma_distribution))

    ckpts = sorted(args.ckpt_dir.glob("agent_epoch_*.pt"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_loss_denoising", "one_step_psnr", "num_batches"])
        writer.writeheader()
        for ck in ckpts:
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            sd = {k.replace("agent.", ""): v for k, v in sd.items() if k.startswith("agent.")} or sd
            agent.load_state_dict(sd, strict=True)
            agent.eval()
            losses, psnrs = [], []
            with torch.no_grad(), fixed_rng(args.val_seed):
                for batch in DatasetTraverser(val, args.batch_size, chunk):
                    _, m = agent.upsampler(batch.to(device))
                    losses.append(m["loss_denoising"])
                    psnrs.append(m["psnr"])
            epoch = int(re.search(r"(\d+)", ck.stem).group(1))
            row = {"epoch": epoch, "val_loss_denoising": sum(losses) / len(losses), "one_step_psnr": sum(psnrs) / len(psnrs), "num_batches": len(losses)}
            writer.writerow(row)
            f.flush()
            print(f"epoch {epoch}: val loss {row['val_loss_denoising']:.4f}, one-step PSNR {row['one_step_psnr']:.2f} dB", flush=True)


if __name__ == "__main__":
    main()
