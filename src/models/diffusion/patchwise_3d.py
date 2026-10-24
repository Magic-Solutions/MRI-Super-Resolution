"""Sliding-window inference for the 3D model, shared by evaluation and training-time validation."""

import torch

from .denoiser_3d import Denoiser3d


@torch.no_grad()
def patchwise_sample(
    model: Denoiser3d,
    lr_vol: torch.Tensor,
    device: torch.device,
    patch_size: int,
    overlap: int,
    uf: int,
    num_steps: int,
    verbose: bool = True,
) -> torch.Tensor:
    """Run patch-based 3D inference with overlap blending."""
    _, _, d, h, w = lr_vol.shape
    stride = patch_size - overlap
    hr_d, hr_h, hr_w = d, h * uf, w * uf

    output = torch.zeros(1, 1, hr_d, hr_h, hr_w)
    counts = torch.zeros(1, 1, hr_d, hr_h, hr_w)

    d_starts = list(range(0, max(d - patch_size, 0) + 1, stride))
    if d_starts[-1] + patch_size < d:
        d_starts.append(d - patch_size)
    h_starts = list(range(0, max(h - patch_size, 0) + 1, stride))
    if h_starts[-1] + patch_size < h:
        h_starts.append(h - patch_size)
    w_starts = list(range(0, max(w - patch_size, 0) + 1, stride))
    if w_starts[-1] + patch_size < w:
        w_starts.append(w - patch_size)

    total_patches = len(d_starts) * len(h_starts) * len(w_starts)
    patch_idx = 0

    for ds in d_starts:
        for hs in h_starts:
            for ws in w_starts:
                de = min(ds + patch_size, d)
                he = min(hs + patch_size, h)
                we = min(ws + patch_size, w)

                lr_patch = lr_vol[:, :, ds:de, hs:he, ws:we].to(device)
                pred_patch = model.sample(lr_patch, num_steps=num_steps).cpu()

                output[:, :, ds:de, hs * uf:he * uf, ws * uf:we * uf] += pred_patch
                counts[:, :, ds:de, hs * uf:he * uf, ws * uf:we * uf] += 1

                patch_idx += 1
                if verbose and (patch_idx % 10 == 0 or patch_idx == total_patches):
                    print(f"    patch {patch_idx}/{total_patches}")

    counts = counts.clamp(min=1)
    return output / counts
