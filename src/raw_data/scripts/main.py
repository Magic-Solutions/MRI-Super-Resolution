"""Process FOMO60K NIfTI volumes into low-res / high-res training pairs.

Scans a FOMO60K dataset folder (e.g. PT015_NKI) for NIfTI files and produces
paired low-resolution and high-resolution data suitable for super-resolution
training.

Modes:
  2D  – Extract 2D slices along a single axis (axial, coronal, or sagittal).
         Each slice becomes one sample stored as a grayscale image tensor.
  3D  – Keep full 3D volumes.  Each volume becomes one sample stored as a
         3D tensor.

Output layout (mirrors the existing processed_data format):
    <out_dir>/
        full_res/<split>/<subject>_<ses>_<seq>.hdf5   (high-res)
        low_res/<split>/info.pt
        low_res/<split>/NNN/NN/N/N.pt                 (low-res)
        samples/                                       (visual QC)

Usage:
    python main.py <out_dir> <fomo_dir> [--mode 2d --axis axial]
    python main.py <out_dir> <fomo_dir> --mode 3d
"""

import gc
import random
import shutil
from collections import Counter
from pathlib import Path
import argparse

import h5py
import nibabel as nib
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent

AXIS_MAP = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}

DEFAULT_SCALE_FACTOR = 4
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SEED = 42


def load_volume(path: Path) -> np.ndarray:
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize to [0, 255] uint8 using the 1st–99th percentile of nonzero voxels."""
    mask = volume > 0
    if not mask.any():
        return np.zeros_like(volume, dtype=np.uint8)
    vmin, vmax = np.percentile(volume[mask], [1, 99])
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.uint8)
    clipped = np.clip(volume, vmin, vmax)
    normed = (clipped - vmin) / (vmax - vmin) * 255.0
    return normed.astype(np.uint8)


def downsample_2d(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2D image by averaging non-overlapping blocks."""
    h, w = image.shape
    new_h = h // factor * factor
    new_w = w // factor * factor
    cropped = image[:new_h, :new_w].astype(np.float32)
    blocks = cropped.reshape(new_h // factor, factor, new_w // factor, factor)
    return blocks.mean(axis=(1, 3)).astype(np.uint8)


def downsample_3d(volume: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 3D volume by averaging non-overlapping blocks."""
    s0, s1, s2 = volume.shape
    n0 = s0 // factor * factor
    n1 = s1 // factor * factor
    n2 = s2 // factor * factor
    cropped = volume[:n0, :n1, :n2].astype(np.float32)
    blocks = cropped.reshape(
        n0 // factor, factor,
        n1 // factor, factor,
        n2 // factor, factor,
    )
    return blocks.mean(axis=(1, 3, 5)).astype(np.uint8)


def get_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    slc = np.take(volume, idx, axis=axis)
    return np.rot90(slc)


def validate_2d_pair_shapes(
    slices_hi: list[np.ndarray],
    slices_lo: list[np.ndarray],
    scale_factor: int,
) -> tuple[bool, str]:
    if not slices_hi or not slices_lo:
        return False, "empty hi/lo slice list"
    if len(slices_hi) != len(slices_lo):
        return False, f"length mismatch hi={len(slices_hi)} lo={len(slices_lo)}"

    hi_shapes = {slc.shape for slc in slices_hi}
    lo_shapes = {slc.shape for slc in slices_lo}
    if len(hi_shapes) != 1:
        return False, f"inconsistent HR slice shapes: {sorted(hi_shapes)}"
    if len(lo_shapes) != 1:
        return False, f"inconsistent LR slice shapes: {sorted(lo_shapes)}"

    hi_h, hi_w = next(iter(hi_shapes))
    lo_h, lo_w = next(iter(lo_shapes))
    if hi_h != lo_h * scale_factor or hi_w != lo_w * scale_factor:
        return (
            False,
            f"HR/LR ratio mismatch: hi={(hi_h, hi_w)} lo={(lo_h, lo_w)} "
            f"scale_factor={scale_factor}",
        )
    return True, ""


def validate_3d_pair_shapes(
    volume_hi: np.ndarray,
    volume_lo: np.ndarray,
    scale_factor: int,
) -> tuple[bool, str]:
    if volume_hi.ndim != 3 or volume_lo.ndim != 3:
        return False, f"expected 3D tensors, got hi={volume_hi.shape} lo={volume_lo.shape}"
    if (
        volume_hi.shape[0] != volume_lo.shape[0] * scale_factor
        or volume_hi.shape[1] != volume_lo.shape[1] * scale_factor
        or volume_hi.shape[2] != volume_lo.shape[2] * scale_factor
    ):
        return (
            False,
            f"HR/LR ratio mismatch: hi={volume_hi.shape} lo={volume_lo.shape} "
            f"scale_factor={scale_factor}",
        )
    return True, ""


def get_episode_path(directory: Path, episode_id: int) -> Path:
    n = 3
    powers = np.arange(n)
    subfolders = np.floor((episode_id % 10 ** (1 + powers)) / 10**powers) * 10**powers
    subfolders = [int(x) for x in subfolders[::-1]]
    subfolders = "/".join([f"{x:0{n - i}d}" for i, x in enumerate(subfolders)])
    return directory / subfolders / f"{episode_id}.pt"


def discover_nifti_files(
    fomo_dir: Path,
    sequence_filter: str | None = None,
) -> list[Path]:
    """Find all .nii.gz files under a FOMO dataset directory.

    Optionally filter by sequence prefix (e.g. 't1' to skip dwi/t2).
    """
    all_files = sorted(fomo_dir.rglob("*.nii.gz"))
    if sequence_filter:
        all_files = [f for f in all_files if f.name.startswith(sequence_filter)]
    return all_files


def train_test_split(
    files: list[Path],
    test_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Split files into train/test by subject to avoid data leakage."""
    subject_files: dict[str, list[Path]] = {}
    for f in files:
        sub_dir = None
        for parent in f.parents:
            if parent.name.startswith("sub_"):
                sub_dir = parent.name
                break
        key = sub_dir or f.stem
        subject_files.setdefault(key, []).append(f)

    subjects = sorted(subject_files.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_test = max(1, int(len(subjects) * test_ratio))
    test_subjects = set(subjects[:n_test])

    train_files: list[Path] = []
    test_files: list[Path] = []
    for subj in subjects:
        if subj in test_subjects:
            test_files.extend(subject_files[subj])
        else:
            train_files.extend(subject_files[subj])

    return train_files, test_files


def save_full_res_hdf5_2d(
    path: Path,
    slices: list[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for i, slc in enumerate(slices):
            f.create_dataset(f"frame_{i}_x", data=slc, dtype=np.uint8)


def save_full_res_hdf5_3d(
    path: Path,
    volume: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("volume_x", data=volume, dtype=np.uint8)


def save_low_res_episode_2d(
    split_dir: Path,
    episode_id: int,
    slices_lo: list[np.ndarray],
    source_id: str,
) -> None:
    n = len(slices_lo)
    obs_list = []
    for slc in slices_lo:
        t = torch.from_numpy(slc).unsqueeze(0)  # (1, H, W)
        obs_list.append(t)
    obs = torch.stack(obs_list)  # (N, 1, H, W)

    episode = {
        "obs": obs,
        "info": {"original_file_id": source_id},
    }
    path = get_episode_path(split_dir, episode_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(episode, tmp)
    tmp.rename(path)


def save_low_res_episode_3d(
    split_dir: Path,
    episode_id: int,
    volume_lo: np.ndarray,
    source_id: str,
) -> None:
    obs = torch.from_numpy(volume_lo).unsqueeze(0)  # (1, D, H, W)

    episode = {
        "obs": obs,
        "info": {"original_file_id": source_id},
    }
    path = get_episode_path(split_dir, episode_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(episode, tmp)
    tmp.rename(path)


def load_info(info_path: Path) -> dict:
    if info_path.is_file():
        return torch.load(info_path, weights_only=False)
    return {
        "is_static": True,
        "num_episodes": 0,
        "num_steps": 0,
        "start_idx": np.array([], dtype=np.int64),
        "lengths": np.array([], dtype=np.int64),
        "counter_rew": Counter(),
        "counter_end": Counter(),
    }


def save_info(info_path: Path, info: dict) -> None:
    info_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(info, info_path)


def save_sample_slice(
    sample_dir: Path,
    name: str,
    slice_hi: np.ndarray,
    slice_lo: np.ndarray,
) -> None:
    """Save a sample slice pair as .npy for quick visual QC."""
    import matplotlib.pyplot as plt

    sample_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor("black")

    axes[0].imshow(slice_hi, cmap="gray")
    axes[0].set_title("High-res", color="white")
    axes[0].axis("off")

    axes[1].imshow(slice_lo, cmap="gray")
    axes[1].set_title("Low-res", color="white")
    axes[1].axis("off")

    fig.suptitle(name, color="white", fontsize=11)
    fig.tight_layout()
    fig.savefig(sample_dir / f"{name}.png", dpi=120, facecolor="black")
    plt.close(fig)


def process_2d(
    files: list[Path],
    split: str,
    out_dir: Path,
    axis: int,
    axis_name: str,
    scale_factor: int,
    info: dict,
    save_samples: bool = False,
) -> dict:
    full_res_dir = out_dir / "full_res" / split
    split_dir = out_dir / "low_res" / split
    sample_dir = out_dir / "samples"

    expected_hi_shape: tuple[int, int] | None = None
    expected_lo_shape: tuple[int, int] | None = None

    for file_idx, nii_path in enumerate(files):
        rel = nii_path.relative_to(nii_path.parents[2])
        stem = str(rel).replace("/", "_").replace(".nii.gz", "")
        print(f"  [{file_idx + 1}/{len(files)}] {rel}")

        volume = load_volume(nii_path)
        volume_u8 = normalize_volume(volume)
        n_slices = volume_u8.shape[axis]

        slices_hi = []
        slices_lo = []
        for i in range(n_slices):
            slc = get_slice(volume_u8, axis, i)
            if slc.sum() == 0:
                continue
            slc_lo = downsample_2d(slc, scale_factor)
            h_lo, w_lo = slc_lo.shape
            slc_hi = slc[: h_lo * scale_factor, : w_lo * scale_factor]
            slices_hi.append(slc_hi)
            slices_lo.append(slc_lo)

        if not slices_hi:
            print(f"    Skipped (all blank slices)")
            continue

        is_valid, reason = validate_2d_pair_shapes(slices_hi, slices_lo, scale_factor)
        if not is_valid:
            print(f"    Skipped (shape validation failed: {reason})")
            continue

        curr_hi_shape = slices_hi[0].shape
        curr_lo_shape = slices_lo[0].shape
        if expected_hi_shape is None:
            expected_hi_shape = curr_hi_shape
            expected_lo_shape = curr_lo_shape
            print(f"    Reference shape set: HR={expected_hi_shape}, LR={expected_lo_shape}")
        elif curr_hi_shape != expected_hi_shape or curr_lo_shape != expected_lo_shape:
            print(
                "    Skipped (mismatched sample shape: "
                f"expected HR={expected_hi_shape}, LR={expected_lo_shape}; "
                f"got HR={curr_hi_shape}, LR={curr_lo_shape})"
            )
            continue

        episode_id = info["num_episodes"]
        hdf5_name = f"{stem}_{axis_name}.hdf5"
        hdf5_path = full_res_dir / hdf5_name
        file_id = f"{split}/{hdf5_name}"

        save_full_res_hdf5_2d(hdf5_path, slices_hi)
        save_low_res_episode_2d(split_dir, episode_id, slices_lo, file_id)

        n_samples = len(slices_hi)
        info["start_idx"] = np.concatenate(
            (info["start_idx"], np.array([info["num_steps"]]))
        )
        info["lengths"] = np.concatenate(
            (info["lengths"], np.array([n_samples]))
        )
        info["num_steps"] += n_samples
        info["num_episodes"] += 1

        if save_samples and file_idx == 0:
            mid = len(slices_hi) // 2
            save_sample_slice(
                sample_dir,
                f"{stem}_{axis_name}_slice{mid}",
                slices_hi[mid],
                slices_lo[mid],
            )

        print(f"    {n_samples} slices -> episode {episode_id}")

        del volume, volume_u8, slices_hi, slices_lo
        gc.collect()

    return info


def process_3d(
    files: list[Path],
    split: str,
    out_dir: Path,
    scale_factor: int,
    info: dict,
    save_samples: bool = False,
) -> dict:
    full_res_dir = out_dir / "full_res" / split
    split_dir = out_dir / "low_res" / split
    sample_dir = out_dir / "samples"

    expected_hi_shape: tuple[int, int, int] | None = None
    expected_lo_shape: tuple[int, int, int] | None = None

    for file_idx, nii_path in enumerate(files):
        rel = nii_path.relative_to(nii_path.parents[2])
        stem = str(rel).replace("/", "_").replace(".nii.gz", "")
        print(f"  [{file_idx + 1}/{len(files)}] {rel}")

        volume = load_volume(nii_path)
        volume_u8 = normalize_volume(volume)

        if volume_u8.sum() == 0:
            print(f"    Skipped (blank volume)")
            continue

        volume_lo = downsample_3d(volume_u8, scale_factor)
        s = volume_lo.shape
        volume_hi = volume_u8[
            : s[0] * scale_factor,
            : s[1] * scale_factor,
            : s[2] * scale_factor,
        ]

        is_valid, reason = validate_3d_pair_shapes(volume_hi, volume_lo, scale_factor)
        if not is_valid:
            print(f"    Skipped (shape validation failed: {reason})")
            continue

        if expected_hi_shape is None:
            expected_hi_shape = volume_hi.shape
            expected_lo_shape = volume_lo.shape
            print(f"    Reference shape set: HR={expected_hi_shape}, LR={expected_lo_shape}")
        elif volume_hi.shape != expected_hi_shape or volume_lo.shape != expected_lo_shape:
            print(
                "    Skipped (mismatched sample shape: "
                f"expected HR={expected_hi_shape}, LR={expected_lo_shape}; "
                f"got HR={volume_hi.shape}, LR={volume_lo.shape})"
            )
            continue

        episode_id = info["num_episodes"]
        hdf5_name = f"{stem}_3d.hdf5"
        hdf5_path = full_res_dir / hdf5_name
        file_id = f"{split}/{hdf5_name}"

        save_full_res_hdf5_3d(hdf5_path, volume_hi)
        save_low_res_episode_3d(split_dir, episode_id, volume_lo, file_id)

        info["start_idx"] = np.concatenate(
            (info["start_idx"], np.array([info["num_steps"]]))
        )
        info["lengths"] = np.concatenate((info["lengths"], np.array([1])))
        info["num_steps"] += 1
        info["num_episodes"] += 1

        if save_samples and file_idx == 0:
            mid_ax = volume_hi.shape[2] // 2
            slc_hi = np.rot90(volume_hi[:, :, mid_ax])
            slc_lo = np.rot90(volume_lo[:, :, mid_ax // scale_factor])
            save_sample_slice(sample_dir, f"{stem}_3d_axial_mid", slc_hi, slc_lo)

        print(
            f"    volume {volume_hi.shape} -> {volume_lo.shape} "
            f"(episode {episode_id})"
        )

        del volume, volume_u8, volume_hi, volume_lo
        gc.collect()

    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process FOMO60K NIfTI data into low/high-res training pairs."
    )
    parser.add_argument(
        "out_dir", type=Path,
        help="Root output directory (e.g. src/processed_data_mri)",
    )
    parser.add_argument(
        "fomo_dir", type=Path,
        help="Path to FOMO60K dataset folder (e.g. src/raw_data/FOMO60K/PT015_NKI)",
    )
    parser.add_argument(
        "--mode",
        choices=["2d", "3d"],
        default="2d",
        help="Processing mode: 2d (slice-based) or 3d (volume-based). Default: 2d",
    )
    parser.add_argument(
        "--axis",
        choices=["axial", "coronal", "sagittal"],
        default="axial",
        help="Slice axis for 2d mode (ignored in 3d mode). Default: axial",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=DEFAULT_SCALE_FACTOR,
        help=f"Downsampling factor for low-res data. Default: {DEFAULT_SCALE_FACTOR}",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="t1",
        help="NIfTI filename prefix filter (e.g. 't1', 't2', 'dwi'). Default: t1",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Fraction of subjects held out for test. Default: {DEFAULT_TEST_RATIO}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for train/test split. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "-n", "--max-files",
        type=int,
        default=None,
        help="Limit total number of NIfTI files to process (before train/test split).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.fomo_dir.is_dir():
        raise FileNotFoundError(f"FOMO directory not found: {args.fomo_dir}")

    if args.out_dir.exists():
        print(f"Clearing existing {args.out_dir} ...")
        shutil.rmtree(args.out_dir)

    print(f"FOMO data:     {args.fomo_dir}")
    print(f"Output:        {args.out_dir}")
    print(f"Mode:          {args.mode}")
    if args.mode == "2d":
        print(f"Axis:          {args.axis}")
    print(f"Scale factor:  {args.scale_factor}x")
    print(f"Sequence:      {args.sequence}*")
    print(f"Test ratio:    {args.test_ratio}")

    nifti_files = discover_nifti_files(args.fomo_dir, args.sequence)
    if not nifti_files:
        print("No NIfTI files found. Check --sequence filter and fomo_dir path.")
        return

    if args.max_files is not None:
        nifti_files = nifti_files[: args.max_files]

    print(f"\nUsing {len(nifti_files)} NIfTI files")

    train_files, test_files = train_test_split(
        nifti_files, args.test_ratio, args.seed
    )
    print(f"  Train: {len(train_files)} files")
    print(f"  Test:  {len(test_files)} files")

    for split, files in [("train", train_files), ("test", test_files)]:
        print(f"\n{'#' * 60}")
        print(f"# {split.upper()} ({len(files)} files)")
        print(f"{'#' * 60}")

        if not files:
            print("  (empty)")
            continue

        split_dir = args.out_dir / "low_res" / split
        info_path = split_dir / "info.pt"
        info = load_info(info_path)

        if args.mode == "2d":
            axis_idx = AXIS_MAP[args.axis]
            info = process_2d(
                files, split, args.out_dir, axis_idx, args.axis,
                args.scale_factor, info, save_samples=(split == "train"),
            )
        else:
            info = process_3d(
                files, split, args.out_dir,
                args.scale_factor, info, save_samples=(split == "train"),
            )

        save_info(info_path, info)
        print(
            f"\n  {split}: {info['num_episodes']} episodes, "
            f"{info['num_steps']} samples"
        )

    print(f"\n{'=' * 60}")
    print(f"Dataset written to {args.out_dir}")
    for split in ("train", "test"):
        info_path = args.out_dir / "low_res" / split / "info.pt"
        if info_path.exists():
            info = torch.load(info_path, weights_only=False)
            print(
                f"  {split}: {info['num_episodes']} episodes, "
                f"{info['num_steps']} samples"
            )
            print(f"    full_res: {args.out_dir / 'full_res' / split}")
            print(f"    low_res:  {args.out_dir / 'low_res' / split}")
        else:
            print(f"  {split}: (empty)")


if __name__ == "__main__":
    main()
