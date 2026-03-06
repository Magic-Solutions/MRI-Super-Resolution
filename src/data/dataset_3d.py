"""Patch-based 3D dataset for volumetric MRI super-resolution.

Loads full LR/HR NIfTI volumes and extracts random 3D patches during training.
For evaluation, provides full volumes.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class MRIPatchDataset3d(TorchDataset):
    """Dataset that yields random 3D patch pairs (LR, HR) from MRI volumes.

    LR patches have size ``(patch_size, patch_size, patch_size)`` and HR patches
    are ``(patch_size, patch_size * uf, patch_size * uf)`` (upsampling in H/W only).
    Values are normalized to [-1, 1].
    """

    def __init__(
        self,
        lr_dir: Path,
        hr_dir: Path,
        patch_size: int = 64,
        upsampling_factor: int = 2,
        patches_per_volume: int = 8,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.upsampling_factor = upsampling_factor
        self.patches_per_volume = patches_per_volume

        self._lr_volumes, self._hr_volumes = _discover_volume_pairs(lr_dir, hr_dir)
        self.num_volumes = len(self._lr_volumes)

    def __len__(self) -> int:
        return self.num_volumes * self.patches_per_volume

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx = idx // self.patches_per_volume

        lr_vol = _load_episode_as_volume(self._lr_volumes[vol_idx])
        hr_vol = _load_hdf5_as_volume(self._hr_volumes[vol_idx])

        ps = self.patch_size
        uf = self.upsampling_factor
        lr_d, lr_h, lr_w = lr_vol.shape
        hr_d, hr_h, hr_w = hr_vol.shape

        eff_d = min(lr_d, hr_d)
        eff_h = min(lr_h, hr_h // uf)
        eff_w = min(lr_w, hr_w // uf)

        d_max = max(1, eff_d - ps)
        h_max = max(1, eff_h - ps)
        w_max = max(1, eff_w - ps)

        d0 = np.random.randint(0, d_max)
        h0 = np.random.randint(0, h_max)
        w0 = np.random.randint(0, w_max)

        d_end = min(d0 + ps, eff_d)
        h_end = min(h0 + ps, eff_h)
        w_end = min(w0 + ps, eff_w)

        lr_patch = lr_vol[d0:d_end, h0:h_end, w0:w_end]
        hr_patch = hr_vol[d0:d_end, h0 * uf:h_end * uf, w0 * uf:w_end * uf]

        lr_patch = _pad_to(lr_patch, (ps, ps, ps))
        hr_patch = _pad_to(hr_patch, (ps, ps * uf, ps * uf))

        lr_t = torch.from_numpy(lr_patch.copy()).unsqueeze(0).float().div(255).mul(2).sub(1)
        hr_t = torch.from_numpy(hr_patch.copy()).unsqueeze(0).float().div(255).mul(2).sub(1)

        return {"lr": lr_t, "hr": hr_t}


class MRIVolumeDataset3d(TorchDataset):
    """Dataset that yields full LR/HR volume pairs for evaluation."""

    def __init__(
        self,
        lr_dir: Path,
        hr_dir: Path,
        upsampling_factor: int = 2,
    ) -> None:
        super().__init__()
        self.upsampling_factor = upsampling_factor
        self._lr_volumes, self._hr_volumes = _discover_volume_pairs(lr_dir, hr_dir)

    def __len__(self) -> int:
        return len(self._lr_volumes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lr_vol = _load_episode_as_volume(self._lr_volumes[idx])
        hr_vol = _load_hdf5_as_volume(self._hr_volumes[idx])

        lr_t = torch.from_numpy(lr_vol.copy()).unsqueeze(0).float().div(255).mul(2).sub(1)
        hr_t = torch.from_numpy(hr_vol.copy()).unsqueeze(0).float().div(255).mul(2).sub(1)

        return {"lr": lr_t, "hr": hr_t}


def _pad_to(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Zero-pad array to target shape if smaller."""
    pads = []
    for cur, tgt in zip(arr.shape, target_shape):
        pads.append((0, max(0, tgt - cur)))
    if all(p == (0, 0) for p in pads):
        return arr
    return np.pad(arr, pads, mode="constant", constant_values=0)


def _discover_volume_pairs(
    lr_dir: Path,
    hr_dir: Path,
) -> Tuple[List[Path], List[Path]]:
    """Match LR .pt files to HR .hdf5 files.

    Uses ``episode_mapping.json`` cache in lr_dir if available (fast),
    otherwise falls back to reading ``original_file_id`` from each .pt file.
    """
    import json

    lr_dir = Path(lr_dir)
    hr_dir = Path(hr_dir)

    hr_by_key: Dict[str, Path] = {}
    for f in hr_dir.rglob("*.hdf5"):
        key = f"{f.parent.name}/{f.name}"
        hr_by_key[key] = f

    mapping_path = lr_dir / "episode_mapping.json"
    if mapping_path.exists():
        with open(mapping_path) as fin:
            mapping = json.load(fin)
        lr_out: List[Path] = []
        hr_out: List[Path] = []
        for rel_path, file_id in sorted(mapping.items()):
            lr_path = lr_dir / rel_path
            if lr_path.exists() and file_id in hr_by_key:
                lr_out.append(lr_path)
                hr_out.append(hr_by_key[file_id])
        return lr_out, hr_out

    lr_out = []
    hr_out = []
    for f in sorted(lr_dir.rglob("*.pt")):
        if f.name == "info.pt":
            continue
        d = torch.load(f, weights_only=False, map_location="cpu")
        file_id = d.get("info", {}).get("original_file_id", "")
        if file_id in hr_by_key:
            lr_out.append(f)
            hr_out.append(hr_by_key[file_id])

    return lr_out, hr_out


def _load_episode_as_volume(path: Path) -> np.ndarray:
    """Load a .pt episode file and reshape obs (N, 1, H, W) -> (N, H, W) uint8."""
    d = torch.load(path, weights_only=False, map_location="cpu")
    obs = d["obs"]  # (N, 1, H, W) uint8
    return obs.squeeze(1).numpy()


def _load_hdf5_as_volume(path: Path) -> np.ndarray:
    """Load an HDF5 file with frame_N_x keys into a (N, H, W) uint8 volume."""
    with h5py.File(path, "r") as f:
        keys = sorted(
            [k for k in f.keys() if k.endswith("_x")],
            key=lambda k: int(k.split("_")[1]),
        )
        slices = [np.asarray(f[k]) for k in keys]
    return np.stack(slices)
