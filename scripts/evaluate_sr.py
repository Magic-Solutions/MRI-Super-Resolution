#!/usr/bin/env python3
"""Evaluate one super-resolution method on a test split and write per-slice metrics.

Every method is scored the same way, so the CSVs can be compared with
``scripts/compare_sr.py``:

- Slices whose HR is entirely zero are skipped: they hold no anatomy, and the
  first submission's preprocessor (src/raw_data/scripts/main.py) dropped them.
- PSNR and SSIM are 2-D per slice on [-1, 1]. ``psnr_fg`` is PSNR over the
  pixels where the HR slice is non-zero, so background does not dominate.
- Predictions are clamped to [-1, 1].
- Interpolation (the bilinear/bicubic baselines and the 2.5D model's
  conditioning) runs on CPU, so results do not depend on the device. MPS has
  no bicubic, and the first submission's "bicubic" baseline was bilinear for
  that reason.

Methods: bilinear, bicubic, 25d, 3d, edsr, swin2sr, ressrdiff (needs --code-dir, a Res-SRDiff clone).

Usage:
    python scripts/evaluate_sr.py --method bicubic --data-dir <split> --out bicubic.csv
    python scripts/evaluate_sr.py --method 25d --checkpoint agent_best_val.pt --data-dir <split> --out 25d.csv
    python scripts/evaluate_sr.py --method 3d --checkpoint model_best_val.pt --patch-size 64 --data-dir <split> --out 3d.csv
    python scripts/evaluate_sr.py --method edsr --checkpoint edsr_best_val.pt --data-dir <split> --out edsr.csv

``<split>`` holds ``low_res/test`` and ``full_res/test`` (e.g. the ``evaluation``
folder of processed_data_mri_resub).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import Dataset, MRIHdf5Dataset  # noqa: E402
from metrics import lpips_distance, psnr, ssim  # noqa: E402

METHODS = ["bilinear", "bicubic", "25d", "3d", "edsr", "swin2sr", "ressrdiff"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--data-dir", type=Path, required=True, help="Folder with low_res/test and full_res/test")
    p.add_argument("--out", type=Path, required=True, help="Per-slice CSV; a .json with the settings is written next to it")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--name", default=None, help="Method name in the CSV (default: --method)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=20, help="Sampling steps for the diffusion models")
    p.add_argument("--lpips", action="store_true", help="Also compute LPIPS (downloads AlexNet weights on first use)")
    p.add_argument("--max-volumes", type=int, default=0, help="Evaluate only the first N volumes (0 = all)")
    p.add_argument("--volumes", nargs="+", default=None, metavar="NAME",
                   help="Evaluate only these volumes (HDF5 file names without .hdf5, e.g. sub_2794_ses_1_t1_sagittal)")
    p.add_argument("--save-predictions", type=Path, default=None, metavar="DIR",
                   help="Also write each volume's clamped prediction as uint8 HDF5 (<DIR>/<name>.hdf5, "
                        "datasets frame_<i>_x indexed like the HR files) for the qualitative figures")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                   help="auto picks cuda, then mps, then cpu")
    # 2.5D
    p.add_argument("--sampler", choices=["shared", "legacy"], default="shared",
                   help="shared: deterministic EDM sampler (Euler unless --order 2); legacy: the `sampler` config")
    p.add_argument("--order", type=int, choices=[1, 2], default=1)
    p.add_argument("--config-name", default="trainer_mri")
    p.add_argument("--override", action="append", default=[], help="Hydra override for the 2.5D model config")
    # 3D
    p.add_argument("--patch-size", type=int, default=64, help="LR patch size for sliding-window 3D inference")
    p.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--depths", type=int, nargs="+", default=[2, 2, 2, 2])
    p.add_argument("--attn-depths", type=int, nargs="+", default=[0, 0, 0, 1])
    p.add_argument("--cond-channels", type=int, default=1024)
    # EDSR / Swin2SR
    p.add_argument("--pretrained", action="store_true", help="Use the DIV2K weights instead of --checkpoint")
    # Res-SRDiff
    p.add_argument("--code-dir", type=Path, default=None, help="Clone of github.com/mosaf/Res-SRDiff (src/ressrdiff_adapter.py)")
    return p.parse_args()


def pick_device(choice: str = "auto") -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_volumes(data_dir: Path, max_volumes: int, volumes: list[str] | None = None):
    """Yield (name, lr, hr) per test volume; lr (T, 1, h, w) and hr (T, 1, H, W) in [-1, 1] on CPU.

    ``volumes``, when given, keeps only the volumes with those names (the HDF5 file name without
    ``.hdf5``); ``max_volumes`` still bounds how many episodes are looked at.
    """
    lr_dir, hr_dir = data_dir / "low_res" / "test", data_dir / "full_res" / "test"
    lr_ds = Dataset(lr_dir, MRIHdf5Dataset(hr_dir), name="test")
    lr_ds.load_from_default_path()
    n = lr_ds.num_episodes if max_volumes <= 0 else min(max_volumes, lr_ds.num_episodes)
    for ep_id in range(n):
        ep = lr_ds.load_episode(ep_id)
        lr = ep.obs.float().div(255).mul(2).sub(1) if ep.obs.dtype == torch.uint8 else ep.obs.float()
        hdf5_name = ep.info["original_file_id"].split("/")[-1]
        if volumes is not None and hdf5_name.replace(".hdf5", "") not in volumes:
            continue
        with h5py.File(hr_dir / hdf5_name, "r") as f:
            keys = sorted([k for k in f.keys() if k.endswith("_x")], key=lambda k: int(k.split("_")[1]))
            hr = np.stack([np.asarray(f[k]) for k in keys])
        hr = torch.from_numpy(hr).float().div(255).mul(2).sub(1).unsqueeze(1)
        assert lr.size(0) == hr.size(0), f"{hdf5_name}: {lr.size(0)} LR vs {hr.size(0)} HR slices"
        yield hdf5_name.replace(".hdf5", ""), lr, hr


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Interpolation:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def predict_volume(self, lr: torch.Tensor) -> torch.Tensor:
        return F.interpolate(lr, scale_factor=2, mode=self.mode)


class Model25d:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from agent import Agent
        from models.diffusion import DiffusionSampler, EDMSamplerConfig

        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
            cfg = compose(config_name=args.config_name, overrides=args.override)
        agent = Agent(instantiate(cfg.agent, num_actions=0))
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = {k.replace("agent.", ""): v for k, v in state.items() if k.startswith("agent.")} or state
        agent.load_state_dict(state, strict=True)
        self.upsampler = agent.upsampler.to(device).eval()
        self.device = device
        inner = self.upsampler.cfg.inner_model
        self.n_prev, self.n_next = inner.num_steps_conditioning, inner.num_next_slices_conditioning
        self.sampler_name = args.sampler
        if args.sampler == "legacy":
            sampler_cfg = instantiate(cfg.sampler)
            sampler_cfg.num_steps_denoising = args.num_steps
            self.legacy = DiffusionSampler(self.upsampler, sampler_cfg)
        else:
            self.shared_cfg = EDMSamplerConfig(num_steps=args.num_steps, order=args.order)

    @torch.no_grad()
    def predict_volume(self, lr: torch.Tensor) -> torch.Tensor:
        from models.diffusion import edm_sample, upsampled_condition

        out = []
        for si in range(lr.size(0)):
            prev = upsampled_condition(lr, si, self.n_prev, self.n_next, 2, "bicubic").to(self.device)
            if self.sampler_name == "legacy":
                sr, _ = self.legacy.sample(prev, None)
            else:
                b, t, c, h, w = prev.shape
                cond = prev.reshape(b, t * c, h, w)
                sr = edm_sample(lambda x, sigma: self.upsampler.denoise(x, sigma, None, cond, None),
                                torch.Size((b, c, h, w)), self.shared_cfg, self.device)
            out.append(sr.cpu())
        return torch.cat(out)


class Model3d:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        from models.diffusion.denoiser import SigmaDistributionConfig

        ri3d = load_module(ROOT / "scripts" / "run_inference_3d.py", "run_inference_3d")
        model = ri3d.build_model(args.channels, args.depths, args.attn_depths, args.cond_channels, 2)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False), strict=True)
        model.setup_training(SigmaDistributionConfig(loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=20))
        self.model = model.to(device).eval()
        self.device, self.patch_size, self.num_steps = device, args.patch_size, args.num_steps

    @torch.no_grad()
    def predict_volume(self, lr: torch.Tensor) -> torch.Tensor:
        from models.diffusion.patchwise_3d import patchwise_sample

        vol = lr[:, 0].unsqueeze(0).unsqueeze(0)  # (1, 1, T, h, w)
        pred = patchwise_sample(self.model, vol, self.device, self.patch_size, self.patch_size // 4, 2,
                                self.num_steps, verbose=False)
        return pred[0, 0].unsqueeze(1)  # (T, 1, H, W)


class Model2dBaseline:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        if args.pretrained:
            baselines = load_module(ROOT / "scripts" / "evaluate_baselines.py", "evaluate_baselines")
            self.model = (baselines.load_edsr if args.method == "edsr" else baselines.load_swin2sr)(device)
        else:
            tsr = load_module(ROOT / "src" / "train_swin2sr.py", "train_swin2sr")
            self.model = tsr.BUILDERS[args.method](device)
            self.model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False), strict=True)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict_volume(self, lr: torch.Tensor) -> torch.Tensor:
        out = []
        for si in range(lr.size(0)):
            x = lr[si : si + 1].add(1).div(2).repeat(1, 3, 1, 1).to(self.device)
            y = self.model(x)
            y = y.reconstruction if hasattr(y, "reconstruction") else y
            out.append(y.clamp(0, 1).mean(dim=1, keepdim=True).mul(2).sub(1).cpu())
        return torch.cat(out)


class ModelResSRDiff:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        import ressrdiff_adapter

        self.adapter = ressrdiff_adapter
        self.rmodel = ressrdiff_adapter.build_model(args.code_dir, ressrdiff_adapter.make_flags(), device)
        self.rmodel.ema_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False), strict=True)
        self.model = self.rmodel.ema_model

    @torch.no_grad()
    def predict_volume(self, lr: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.adapter.sample(self.rmodel, self.adapter.lq_from_lr(lr[si : si + 1])).cpu()
                          for si in range(lr.size(0))])


def psnr_foreground(pred: torch.Tensor, target: torch.Tensor) -> float:
    mask = target > -1  # HR pixels that were non-zero before normalization to [-1, 1]
    mse = (pred - target)[mask].pow(2).mean().item()
    return 100.0 if mse < 1e-10 else 10 * math.log10(4.0 / mse)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    args = parse_args()
    if args.method in ("25d", "3d", "ressrdiff") or (args.method in ("edsr", "swin2sr") and not args.pretrained):
        assert args.checkpoint is not None, f"--checkpoint is required for --method {args.method}"
    if args.method == "ressrdiff":
        assert args.code_dir is not None, "--code-dir is required for --method ressrdiff"
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    name = args.name or args.method
    print(f"Method {name} on {device}; data {args.data_dir}")

    if args.method in ("bilinear", "bicubic"):
        method = Interpolation(args.method)
    elif args.method == "25d":
        method = Model25d(args, device)
    elif args.method == "3d":
        method = Model3d(args, device)
    elif args.method == "ressrdiff":
        method = ModelResSRDiff(args, device)
    else:
        method = Model2dBaseline(args, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "subject", "slice", "psnr", "psnr_fg", "ssim", "fg_fraction"] + (["lpips"] if args.lpips else [])
    t0 = time.time()
    n_slices = n_blank = 0
    inference_seconds = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for subject, lr, hr in load_volumes(args.data_dir, args.max_volumes, args.volumes):
            torch.manual_seed(args.seed)
            t_volume = time.time()
            pred = method.predict_volume(lr).clamp(-1, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds.append(time.time() - t_volume)
            if args.save_predictions is not None:
                # Store the prediction the metrics below are computed from, as 0-255 uint8 like the
                # HR files (same frame_<i>_x keys, so slice i of the HR file is slice i here).
                args.save_predictions.mkdir(parents=True, exist_ok=True)
                as_uint8 = pred[:, 0].add(1).div(2).mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
                with h5py.File(args.save_predictions / f"{subject}.hdf5", "w") as f_pred:
                    for si in range(as_uint8.shape[0]):
                        f_pred.create_dataset(f"frame_{si}_x", data=as_uint8[si])
            psnrs = []
            for si in range(hr.size(0)):
                p, h = pred[si : si + 1], hr[si : si + 1]
                if (h > -1).sum() == 0:
                    n_blank += 1
                    continue
                row = {"method": name, "subject": subject, "slice": si, "psnr": psnr(p, h),
                       "psnr_fg": psnr_foreground(p, h), "ssim": ssim(p, h),
                       "fg_fraction": (h > -1).float().mean().item()}
                if args.lpips:
                    row["lpips"] = lpips_distance(p, h)
                writer.writerow(row)
                psnrs.append(row["psnr"])
                n_slices += 1
            print(f"  {subject}: {len(psnrs)} slices, PSNR {np.mean(psnrs):.2f} dB")
    elapsed = time.time() - t0
    print(f"Wrote {n_slices} slices ({n_blank} all-zero HR slices skipped) to {args.out} in {elapsed:.0f}s")
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    model = getattr(method, "upsampler", None) or getattr(method, "model", None)
    settings.update(name=name, device=str(device), git_sha=git_sha(), seconds=round(elapsed, 1),
                    num_slices=n_slices, num_blank_skipped=n_blank,
                    num_parameters=sum(p.numel() for p in model.parameters()) if model is not None else 0,
                    inference_seconds_per_volume=round(float(np.mean(inference_seconds)), 3),
                    peak_gpu_memory_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None)
    args.out.with_suffix(".json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
