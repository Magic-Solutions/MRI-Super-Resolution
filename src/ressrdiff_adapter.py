"""Glue for Res-SRDiff (Safari et al., https://github.com/mosaf/Res-SRDiff), the MRI-specific diffusion comparator (E6).

The Res-SRDiff code is not copied into this repo: the job scripts clone it at a
pinned commit and pass its folder as ``code_dir``. This module builds its model
and diffusion with the settings below, and feeds it our slices the way all our
models see them: intensities x / 255 * 2 - 1, and the LR slice upsampled to HR
size with bicubic interpolation on CPU (Res-SRDiff runs with sf = 1, so it
expects the LR image already at HR size).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

RESSRDIFF_COMMIT = "2a0e0a8c6de924f61f496f1eaad9557038971c99"

# Defaults of Res-SRDiff's script/main.py, with its brain settings (model_channels 64) and our 256 x 256 slices.
DEFAULT_FLAGS = dict(
    model="Super_resolution", model_name="unet_swin", train=True, resume=False, num_gpus=1, cuda=True,
    image_size=256, in_channels=1, out_channels=1, model_channels=64, num_res_blocks=[2, 2, 2, 2],
    attention_resolutions=(64, 32, 16, 8), cond_lq=True, dropout=0.0,
    sf=1.0, schedule_name="exponential", schedule_kwargs={"power": 0.3}, etas_end=0.99, steps=4,
    min_noise_level=0.2, kappa=2.0, weighted_mse=False, predict_type="xstart", timestep_respacing=None,
    scale_factor=1.0, normalize_input=False, latent_flag=False,
    batch_size=16, lr=5e-5, betas_G=(0.9, 0.999), weight_decay=1e-5, ema_rate=0.999, loss_coef=[4.0, 1.0],
    use_amp=False, out_dir="ressrdiff_out",
)


def make_flags(**overrides) -> argparse.Namespace:
    flags = dict(DEFAULT_FLAGS)
    flags.update(overrides)
    return argparse.Namespace(**flags)


def lq_from_lr(lr: torch.Tensor) -> torch.Tensor:
    """(N, 1, h, w) LR in [-1, 1] -> (N, 1, 2h, 2w) bicubic upsampling on CPU, clamped to [-1, 1]."""
    return F.interpolate(lr.cpu().float(), scale_factor=2, mode="bicubic").clamp(-1, 1)


def build_model(code_dir: Path, flags: argparse.Namespace, device: torch.device):
    """Return Res-SRDiff's ``build.Model``: network, EMA copy, diffusion, RAdam optimizer and LPIPS loss."""
    script_dir = str(Path(code_dir) / "script")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from build import Model  # noqa: E402  (from the Res-SRDiff clone)

    return Model(name="Super_resolution", device=device, data_loader=None, test_data_loader=None, FLAGS=flags)


@torch.no_grad()
def sample(rmodel, lq: torch.Tensor, use_ema: bool = True) -> torch.Tensor:
    """Super-resolve (N, 1, H, W) upsampled LR in [-1, 1]; returns (N, 1, H, W) in [-1, 1]."""
    net = rmodel.ema_model if use_ema else rmodel.model
    net.eval()
    lq = lq.to(rmodel.device)
    out = rmodel.base_diffusion.p_sample_loop(
        y=lq, model=net, first_stage_model=None, noise=None, noise_repeat=False, clip_denoised=True,
        denoised_fn=None, model_kwargs={"lq": lq}, progress=False,
    )
    return out.clamp(-1, 1)


def load_ema_into_model(rmodel) -> None:
    rmodel.ema_model.load_state_dict(rmodel.ema_state)
