"""Shared deterministic EDM sampler used by both the 2.5D and the 3D model.

Using one sampler for every model keeps the sigma schedule, solver and step
count identical, so differences in results come from the architecture.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass
class EDMSamplerConfig:
    num_steps: int = 20
    sigma_min: float = 2e-3
    sigma_max: float = 5.0
    rho: int = 7
    order: int = 1  # 1: Euler, 2: Heun (Karras et al. 2022, Alg. 1)
    # The original DIAMOND samplers start from unit-variance noise rather than
    # noise at sigma_max. Keep False to reproduce earlier results.
    scale_init_noise: bool = False


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))


@torch.no_grad()
def edm_sample(
    denoise_fn: Callable[[Tensor, Tensor], Tensor],
    shape: torch.Size,
    cfg: EDMSamplerConfig,
    device: torch.device,
    generator: torch.Generator = None,
) -> Tensor:
    """Sample with the deterministic EDM ODE (no churn).

    ``denoise_fn(x, sigma)`` returns the denoised estimate for noisy input ``x``
    at noise level ``sigma`` (a tensor of shape ``(B,)``), with any
    conditioning already bound in.
    """
    b = shape[0]
    sigmas = build_sigmas(cfg.num_steps, cfg.sigma_min, cfg.sigma_max, cfg.rho, device)
    x = torch.randn(shape, device=device, generator=generator)
    if cfg.scale_init_noise:
        x = x * sigmas[0]
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        denoised = denoise_fn(x, sigma.expand(b))
        d = (x - denoised) / sigma
        x_next = x + d * (next_sigma - sigma)
        if cfg.order == 2 and next_sigma > 0:
            denoised_2 = denoise_fn(x_next, next_sigma.expand(b))
            d_2 = (x_next - denoised_2) / next_sigma
            x_next = x + (d + d_2) / 2 * (next_sigma - sigma)
        x = x_next
    return x
