"""Denoising error of a trained 2.5D model as a function of the noise level sigma.

Replays the training forward pass (Denoiser.forward: same conditioning, same chunking, same
masking) on the validation volumes, but with every target noised at one fixed sigma instead of
a sigma drawn from the training distribution. For each sigma on a log grid it records the
EDM-weighted denoising loss, the image-space MSE of the denoised output and its PSNR, next to
the training density of that sigma (ln sigma ~ N(P_mean, P_std^2)) and the sampler's step
sigmas. Plot with scripts/make_per_sigma_figure.py.

Usage:
    python scripts/per_sigma_error.py --val <dir with low_res/test and full_res/test> \
        --checkpoint agent_best_val.pt --out per_sigma.csv [--override k=v ...]
"""
from __future__ import annotations

import argparse
import csv
import math
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
from models.diffusion.edm_sampler import EDMSamplerConfig, build_sigmas  # noqa: E402
from trainer import fixed_rng  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val", type=Path, required=True, help="Directory holding low_res/test and full_res/test")
    p.add_argument("--checkpoint", type=Path, required=True, help="agent_best_val.pt or an agent_epoch_*.pt")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config-name", default="trainer_mri")
    p.add_argument("--override", nargs="*", default=[], help="Hydra overrides, e.g. the both-neighbours option")
    p.add_argument("--batch-size", type=int, default=4, help="Segments per batch, as in training")
    p.add_argument("--sigma-min", type=float, default=2e-3)
    p.add_argument("--sigma-max", type=float, default=20.0)
    p.add_argument("--num-sigmas", type=int, default=25)
    p.add_argument("--sampler-steps", type=int, default=20, help="Step sigmas of the shared sampler to mark")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: the first available)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    if device.type == "mps":
        # Denoiser.forward picks bilinear conditioning on MPS; training used bicubic on CUDA.
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
    sigma_cfg = instantiate(cfg.upsampler.sigma_distribution)
    agent.setup_training(sigma_cfg)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k.replace("agent.", ""): v for k, v in sd.items() if k.startswith("agent.")} or sd
    agent.load_state_dict(sd, strict=True)
    agent.eval()
    denoiser = agent.upsampler

    p_mean, p_std = sigma_cfg.loc, sigma_cfg.scale
    step_sigmas = build_sigmas(args.sampler_steps, EDMSamplerConfig.sigma_min, EDMSamplerConfig.sigma_max, EDMSamplerConfig.rho, torch.device("cpu")).tolist()

    sigmas = torch.logspace(math.log10(args.sigma_min), math.log10(args.sigma_max), args.num_sigmas).tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sigma", "loss_denoising", "mse_denoised", "psnr_denoised", "train_density", "num_batches"])
        writer.writeheader()
        for sigma in sigmas:
            # One sigma for every target and the same noise seed for each sigma, so the curve depends on sigma only.
            denoiser.sample_sigma_training = lambda n, dev, s=sigma: torch.full((n,), s, device=dev)
            losses, psnrs, mses = [], [], []
            with torch.no_grad(), fixed_rng(args.seed):
                for batch in DatasetTraverser(val, args.batch_size, chunk):
                    _, m = denoiser(batch.to(device))
                    losses.append(m["loss_denoising"])
                    psnrs.append(m["psnr"])
                    # PSNR is on the [-1, 1] range (peak 2), so MSE = 4 / 10^(PSNR/10).
                    mses.append(4.0 / (10 ** (m["psnr"] / 10)))
            ln_s = math.log(sigma)
            density = math.exp(-0.5 * ((ln_s - p_mean) / p_std) ** 2) / (p_std * math.sqrt(2 * math.pi))
            row = {
                "sigma": sigma,
                "loss_denoising": sum(losses) / len(losses),
                "mse_denoised": sum(mses) / len(mses),
                "psnr_denoised": sum(psnrs) / len(psnrs),
                "train_density": density,
                "num_batches": len(losses),
            }
            writer.writerow(row)
            f.flush()
            print(f"sigma {sigma:8.4f}: loss {row['loss_denoising']:.4f}, denoised PSNR {row['psnr_denoised']:.2f} dB", flush=True)

    with args.out.with_suffix(".sampler_sigmas.txt").open("w") as f:
        f.write("\n".join(f"{s:.6g}" for s in step_sigmas) + "\n")
    print(f"wrote {args.out} and the {args.sampler_steps} sampler sigmas next to it")


if __name__ == "__main__":
    main()
