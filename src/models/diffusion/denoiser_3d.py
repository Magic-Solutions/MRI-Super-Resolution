"""3D diffusion-based upsampler for volumetric MRI super-resolution.

Operates on 3D patches: takes a low-res patch, bicubic-upsamples it, and
refines via iterative denoising to produce the high-res output.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .inner_model_3d import InnerModel3d, InnerModel3dConfig
from .denoiser import SigmaDistributionConfig


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners3d:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor


@dataclass
class Denoiser3dConfig:
    inner_model: InnerModel3dConfig
    sigma_data: float
    sigma_offset_noise: float
    upsampling_factor: int


class Denoiser3d(nn.Module):
    def __init__(self, cfg: Denoiser3dConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel3d(cfg.inner_model)
        self.sample_sigma_training = None

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

    def apply_noise(self, x: Tensor, sigma: Tensor) -> Tensor:
        b, c = x.shape[:2]
        offset_noise = self.cfg.sigma_offset_noise * torch.randn(
            b, c, *([1] * (x.ndim - 2)), device=self.device
        )
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor) -> Conditioners3d:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        ndim = 5  # (B, C, D, H, W)
        return Conditioners3d(
            add_dims(c_in, ndim),
            add_dims(c_out, ndim),
            add_dims(c_skip, ndim),
            add_dims(c_noise, 1),
        )

    def forward(
        self,
        lr_volume: Tensor,
        hr_volume: Tensor,
    ) -> Tuple[Tensor, dict]:
        """Training forward pass.

        Args:
            lr_volume: (B, 1, D, H, W) low-res volume.
            hr_volume: (B, 1, D', H', W') high-res volume (target).
        """
        b = lr_volume.size(0)
        uf = self.cfg.upsampling_factor
        _, _, d, h, w = lr_volume.shape
        lr_up = F.interpolate(
            lr_volume, size=(d, h * uf, w * uf),
            mode="trilinear", align_corners=False,
        )

        sigma = self.sample_sigma_training(b, self.device)
        noisy_hr = self.apply_noise(hr_volume, sigma)

        cs = self.compute_conditioners(sigma)
        rescaled_lr = lr_up / self.cfg.sigma_data
        rescaled_noise = noisy_hr * cs.c_in
        model_output = self.inner_model(rescaled_noise, cs.c_noise.reshape(b), rescaled_lr)

        target = (hr_volume - cs.c_skip * noisy_hr) / cs.c_out
        loss = F.mse_loss(model_output, target)
        return loss, {"loss_3d_upsampler": loss.item()}

    @torch.no_grad()
    def denoise(
        self,
        noisy_hr: Tensor,
        sigma: Tensor,
        lr_up: Tensor,
    ) -> Tensor:
        b = noisy_hr.size(0)
        cs = self.compute_conditioners(sigma)
        rescaled_lr = lr_up / self.cfg.sigma_data
        rescaled_noise = noisy_hr * cs.c_in
        model_output = self.inner_model(rescaled_noise, cs.c_noise.reshape(b), rescaled_lr)
        d = cs.c_skip * noisy_hr + cs.c_out * model_output
        return d.clamp(-1, 1).add(1).div(2).mul(255).byte().float().div(255).mul(2).sub(1)

    @torch.no_grad()
    def sample(
        self,
        lr_volume: Tensor,
        num_steps: int = 20,
        sigma_min: float = 2e-3,
        sigma_max: float = 5.0,
        rho: int = 7,
    ) -> Tensor:
        """Generate HR volume from LR via iterative denoising."""
        b, c, vol_d, h, w = lr_volume.shape
        uf = self.cfg.upsampling_factor
        lr_up = F.interpolate(
            lr_volume, size=(vol_d, h * uf, w * uf),
            mode="trilinear", align_corners=False,
        )

        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        l = torch.linspace(0, 1, num_steps, device=self.device)
        sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
        sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))

        x = torch.randn(b, c, vol_d, uf * h, uf * w, device=self.device)
        for sigma_cur, sigma_next in zip(sigmas[:-1], sigmas[1:]):
            denoised = self.denoise(x, sigma_cur.expand(b), lr_up)
            d_cur = (x - denoised) / sigma_cur
            x = x + d_cur * (sigma_next - sigma_cur)
        return x
