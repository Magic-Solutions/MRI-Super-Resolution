"""3D inner model for volumetric MRI super-resolution."""

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..blocks3d import Conv3x3_3d, FourierFeatures3d, GroupNorm3d, UNet3d


@dataclass
class InnerModel3dConfig:
    img_channels: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]


class InnerModel3d(nn.Module):
    """3D UNet for volumetric diffusion upsampling.

    Input: bicubic-upsampled LR volume (1 ch) + noisy HR target (1 ch) = 2 channels.
    """

    def __init__(self, cfg: InnerModel3dConfig) -> None:
        super().__init__()
        self.noise_emb = FourierFeatures3d(cfg.cond_channels)
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        in_channels = 2 * cfg.img_channels  # bicubic LR + noisy target
        self.conv_in = Conv3x3_3d(in_channels, cfg.channels[0])

        self.unet = UNet3d(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm3d(cfg.channels[0])
        self.conv_out = Conv3x3_3d(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_target: Tensor, c_noise: Tensor, lr_upsampled: Tensor) -> Tensor:
        cond = self.cond_proj(self.noise_emb(c_noise))
        x = self.conv_in(torch.cat((lr_upsampled, noisy_target), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x
