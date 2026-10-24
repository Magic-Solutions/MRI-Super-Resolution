"""Low-resolution simulation in k-space, as an alternative to 2x2 block averaging.

``degrade_kspace`` keeps the central 1/factor of k-space along each in-plane
axis (truncation, which is what a lower acquisition matrix does and which
produces Gibbs ringing), optionally adds complex Gaussian noise, and returns
the magnitude image. Complex noise gives Rician-distributed noise in the
magnitude, as in real MR magnitude images. The LR grid is aligned with the
block-averaged one, so models trained on block averaging see no spatial shift.
"""

from __future__ import annotations

import numpy as np


def degrade_kspace(hr: np.ndarray, factor: int = 2, noise_std: float = 0.0,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """hr: (H, W) image in uint8 intensity units. Returns the (H/factor, W/factor) LR magnitude as uint8.

    noise_std is the standard deviation of the real and imaginary noise in the
    same intensity units (0-255).
    """
    h, w = hr.shape
    assert h % factor == 0 and w % factor == 0, f"{hr.shape} is not divisible by {factor}"
    lh, lw = h // factor, w // factor
    k = np.fft.fft2(hr.astype(np.float64))
    # Shift by (factor - 1) / 2 HR pixels so LR pixel j sits at the centre of HR pixels factor*j .. factor*j + factor - 1,
    # as with block averaging (and interpolate(align_corners=False)). Without it the LR image is offset by half a pixel.
    s = (factor - 1) / 2
    k = k * np.exp(2j * np.pi * s * (np.fft.fftfreq(h)[:, None] + np.fft.fftfreq(w)[None, :]))
    k = np.fft.fftshift(k)
    top, left = (h - lh) // 2, (w - lw) // 2
    k_lr = k[top : top + lh, left : left + lw]
    # The inverse FFT of the cropped spectrum is factor^2 brighter; rescale so intensities match the HR image.
    lr = np.fft.ifft2(np.fft.ifftshift(k_lr)) / factor**2
    if noise_std > 0:
        rng = rng if rng is not None else np.random.default_rng()
        lr = lr + noise_std * (rng.standard_normal(lr.shape) + 1j * rng.standard_normal(lr.shape))
    return np.clip(np.floor(np.abs(lr)), 0, 255).astype(np.uint8)
