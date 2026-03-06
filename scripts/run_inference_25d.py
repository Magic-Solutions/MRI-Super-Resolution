#!/usr/bin/env python3
"""Run 2.5D upsampler inference on a few MRI slices and save comparison images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from agent import Agent
from data.dataset import Dataset, MRIHdf5Dataset
from envs.world_model_env import DiffusionSampler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("src/processed_data_mri"))
    p.add_argument("--config-name", default="trainer_mri")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--out-dir", type=Path, default=Path("inference_output"))
    return p.parse_args()


def to_uint8(t: torch.Tensor) -> np.ndarray:
    return t.clamp(-1, 1).add(1).div(2).mul(255).byte().cpu().squeeze(0).numpy()


def save_comparison(gt_hr: np.ndarray, gen_hr: np.ndarray, bicubic: np.ndarray, path: Path) -> None:
    from PIL import Image

    def arr_to_pil(a: np.ndarray) -> Image.Image:
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        return Image.fromarray(a, mode="L")

    imgs = [arr_to_pil(gt_hr), arr_to_pil(gen_hr), arr_to_pil(bicubic)]
    gap = 4
    w_total = sum(im.width for im in imgs) + gap * 2
    h_total = max(im.height for im in imgs)
    canvas = Image.new("L", (w_total, h_total + 20), 200)

    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    x = 0
    for im, label in zip(imgs, ["GT HR", "Generated HR", "Bicubic LR"]):
        canvas.paste(im, (x, 20))
        draw.text((x + 2, 2), label, fill=0)
        x += im.width + gap
    canvas.save(path)


def main() -> None:
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
    up_factor = upsampler.cfg.upsampling_factor
    n_cond = cfg.agent.upsampler.inner_model.num_steps_conditioning
    up_mode = "bilinear" if device.type == "mps" else "bicubic"

    sampler_cfg = instantiate(cfg.world_model_env.diffusion_sampler_upsampling)
    sampler_cfg.num_steps_denoising = args.num_steps
    sampler = DiffusionSampler(upsampler, sampler_cfg)

    lr_dir = args.data_dir / "low_res" / "train"
    hr_dir = args.data_dir / "full_res" / "train"
    hr_dataset = MRIHdf5Dataset(hr_dir)
    lr_dataset = Dataset(lr_dir, dataset_full_res=hr_dataset, name="inference")
    lr_dataset.load_from_default_path()
    print(f"Dataset: {lr_dataset.num_episodes} episodes, {lr_dataset.num_steps} steps")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {args.num_samples} samples with {args.num_steps} denoising steps...")

    rng = np.random.default_rng(42)
    ep_ids = rng.choice(lr_dataset.num_episodes, size=min(args.num_samples, lr_dataset.num_episodes), replace=False)

    import h5py
    for i, ep_id in enumerate(ep_ids):
        ep = lr_dataset.load_episode(int(ep_id))
        obs_lr = ep.obs.float().div(255).mul(2).sub(1) if ep.obs.dtype == torch.uint8 else ep.obs.float()
        t = obs_lr.size(0)
        mid = t // 2

        context_start = max(0, mid - n_cond)
        context = obs_lr[context_start:mid].unsqueeze(0).to(device)
        if context.size(1) < n_cond:
            pad = n_cond - context.size(1)
            context = F.pad(context, [0, 0, 0, 0, 0, 0, pad, 0])

        target_lr = obs_lr[mid].unsqueeze(0).to(device)
        _, c, h, w = target_lr.shape

        target_lr_up = F.interpolate(target_lr, scale_factor=up_factor, mode=up_mode).unsqueeze(1)
        context_up = F.interpolate(
            context.reshape(-1, c, h, w), scale_factor=up_factor, mode=up_mode
        ).reshape(1, n_cond, c, up_factor * h, up_factor * w)
        prev = torch.cat((context_up, target_lr_up), dim=1)

        with torch.no_grad():
            sr_frame, _ = sampler.sample(prev, None)

        gen_hr = to_uint8(sr_frame[0])
        bicubic = to_uint8(F.interpolate(target_lr, scale_factor=up_factor, mode=up_mode)[0])

        file_id = ep.info.get("original_file_id", "")
        if file_id.startswith("train/"):
            file_id = file_id[len("train/"):]
        elif file_id.startswith("test/"):
            file_id = file_id[len("test/"):]
        hr_path = hr_dir / file_id if file_id else None
        if hr_path and hr_path.exists():
            with h5py.File(hr_path, "r") as f:
                keys = sorted([k for k in f.keys() if k.endswith("_x")], key=lambda k: int(k.split("_")[1]))
                gt_hr = np.asarray(f[keys[mid]])
        else:
            gt_hr = np.full_like(gen_hr, 128)

        out_path = args.out_dir / f"sample_{i:03d}_ep{ep_id}_slice{mid}.png"
        save_comparison(gt_hr, gen_hr, bicubic, out_path)
        print(f"  [{i+1}/{len(ep_ids)}] Saved {out_path.name}")

    print(f"Done! Output: {args.out_dir}/")


if __name__ == "__main__":
    main()
