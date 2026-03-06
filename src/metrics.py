"""Image quality metrics for super-resolution evaluation.

All functions expect tensors in [-1, 1] range with shape (B, C, ...).
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def psnr(pred: Tensor, target: Tensor) -> float:
    """Peak Signal-to-Noise Ratio for tensors in [-1, 1]."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    max_val = 2.0  # range is [-1, 1]
    return 10 * torch.log10(torch.tensor(max_val**2 / mse)).item()


def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    return g / g.sum()


def ssim(
    pred: Tensor,
    target: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Structural Similarity Index for tensors in [-1, 1].

    Computed per-channel, averaged over batch and channels.
    Supports both 2D (B, C, H, W) and 3D (B, C, D, H, W).
    """
    C1 = (0.01 * 2.0) ** 2  # (K1 * L)^2, L=2 for [-1,1]
    C2 = (0.03 * 2.0) ** 2

    ndim = pred.ndim - 2  # spatial dims (2 or 3)
    c = pred.size(1)
    device = pred.device

    kernel_1d = _gaussian_kernel_1d(window_size, sigma, device)
    if ndim == 2:
        kernel = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(c, 1, -1, -1)
        conv_fn = F.conv2d
    else:
        kernel = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(c, 1, -1, -1, -1)
        conv_fn = F.conv3d

    pad = window_size // 2

    mu_pred = conv_fn(pred, kernel, padding=pad, groups=c)
    mu_target = conv_fn(target, kernel, padding=pad, groups=c)

    mu_pred_sq = mu_pred**2
    mu_target_sq = mu_target**2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = conv_fn(pred**2, kernel, padding=pad, groups=c) - mu_pred_sq
    sigma_target_sq = conv_fn(target**2, kernel, padding=pad, groups=c) - mu_target_sq
    sigma_cross = conv_fn(pred * target, kernel, padding=pad, groups=c) - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / (
        (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )

    return ssim_map.mean().item()
