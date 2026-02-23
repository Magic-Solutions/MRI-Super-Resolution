"""Shared helpers for depth visualization."""

from __future__ import annotations

import cv2
import numpy as np


def colorize_inverse_depth_uint8(
    depth_u8: np.ndarray,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> np.ndarray:
    """Colorize uint8 depth using inverse-depth percentile scaling.

    Args:
        depth_u8: Depth image as uint8 array of shape (H, W), where 0 means invalid.
        lower_percentile: Lower percentile for robust scaling.
        upper_percentile: Upper percentile for robust scaling.

    Returns:
        RGB uint8 image of shape (H, W, 3).
    """
    if depth_u8.ndim != 2:
        raise ValueError(f"Expected depth_u8 with shape (H, W), got {depth_u8.shape}")

    depth_f = depth_u8.astype(np.float32)
    valid = depth_f > 0
    inv_vis = np.zeros_like(depth_f, dtype=np.float32)
    if valid.any():
        inv_depth = 1.0 / np.maximum(depth_f[valid], 1.0)
        p_lo, p_hi = np.percentile(inv_depth, [lower_percentile, upper_percentile])
        if p_hi > p_lo:
            inv_vis[valid] = np.clip((inv_depth - p_lo) / (p_hi - p_lo), 0.0, 1.0)

    inv_u8 = (inv_vis * 255.0).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(inv_u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
