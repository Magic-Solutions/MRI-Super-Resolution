"""Conditioning frames for sampling one slice with the 2.5D upsampler.

The layout matches ``Denoiser.forward``: the ``num_prev`` LR slices before the
target, the ``num_next`` LR slices after it, then the target's own LR slice.
Neighbours outside the volume are zero frames, like the zero padding of
training segments.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def slice_condition_frames(obs_lr: Tensor, si: int, num_prev: int, num_next: int) -> Tensor:
    """obs_lr: (T, C, h, w) in [-1, 1]. Returns (num_prev + num_next + 1, C, h, w)."""
    t = obs_lr.size(0)
    ids = list(range(si - num_prev, si)) + list(range(si + 1, si + 1 + num_next)) + [si]
    zero = torch.zeros_like(obs_lr[0])
    return torch.stack([obs_lr[j] if 0 <= j < t else zero for j in ids])


def upsampled_condition(obs_lr: Tensor, si: int, num_prev: int, num_next: int, up_factor: int, up_mode: str) -> Tensor:
    """Conditioning frames for slice ``si``, upsampled to HR: (1, num_prev + num_next + 1, C, H, W)."""
    frames = slice_condition_frames(obs_lr, si, num_prev, num_next)
    return F.interpolate(frames, scale_factor=up_factor, mode=up_mode).unsqueeze(0)
