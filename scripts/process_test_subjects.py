#!/usr/bin/env python3
"""Process held-out NKI subjects into the test split.

Reads the existing training episodes to identify which subjects are already
used, then picks N new subjects from the raw FOMO60K data and processes them
into the test directories (low_res/test, full_res/test) WITHOUT touching
the training data.

Usage:
    python scripts/process_test_subjects.py \
        --num-subjects 20 \
        --scale-factor 2 \
        --axis sagittal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "raw_data" / "scripts"))

from main import (
    discover_nifti_files,
    load_info,
    process_2d,
    save_info,
    AXIS_MAP,
)
from data.dataset import Dataset


def get_training_subject_ids(lr_train_dir: Path) -> set[str]:
    """Extract subject IDs from all episodes in the training set."""
    ds = Dataset(lr_train_dir, dataset_full_res=None, name="train")
    ds.load_from_default_path()

    subject_ids: set[str] = set()
    for i in range(ds.num_episodes):
        ep = ds.load_episode(i)
        fid = ep.info.get("original_file_id", "")
        name = fid.split("/")[-1].replace(".hdf5", "")
        parts = name.split("_ses_")
        if parts:
            subject_ids.add(parts[0])
    return subject_ids


def pick_test_subjects(
    fomo_dir: Path,
    sequence: str,
    exclude: set[str],
    num_subjects: int,
    seed: int,
) -> list[Path]:
    """Select NIfTI files from subjects NOT in the exclusion set.

    Only includes files that actually exist on disk (not directory stubs).
    """
    import random

    all_files = [
        f for f in discover_nifti_files(fomo_dir, sequence)
        if f.is_file() and f.stat().st_size > 0
    ]

    subject_files: dict[str, list[Path]] = {}
    for f in all_files:
        sub_dir = None
        for parent in f.parents:
            if parent.name.startswith("sub_"):
                sub_dir = parent.name
                break
        if sub_dir is None:
            continue
        if sub_dir in exclude:
            continue
        subject_files.setdefault(sub_dir, []).append(f)

    available = sorted(subject_files.keys())
    rng = random.Random(seed)
    rng.shuffle(available)
    chosen = available[:num_subjects]

    files: list[Path] = []
    for subj in sorted(chosen):
        files.extend(sorted(subject_files[subj]))

    print(f"Available subjects (excl. training): {len(available)}")
    print(f"Chosen test subjects ({len(chosen)}): {', '.join(sorted(chosen))}")
    print(f"Total NIfTI files to process: {len(files)}")
    return files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=Path("src/processed_data_mri"))
    p.add_argument("--fomo-dir", type=Path, default=Path("src/raw_data/FOMO60K/PT015_NKI"))
    p.add_argument("--num-subjects", type=int, default=20)
    p.add_argument("--scale-factor", type=int, default=2)
    p.add_argument("--axis", choices=["axial", "coronal", "sagittal"], default="sagittal")
    p.add_argument("--sequence", default="t1")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    lr_train_dir = args.processed_dir / "low_res" / "train"
    lr_test_dir = args.processed_dir / "low_res" / "test"
    info_path = lr_test_dir / "info.pt"

    if not lr_train_dir.exists():
        raise FileNotFoundError(f"Training data not found: {lr_train_dir}")
    if not args.fomo_dir.is_dir():
        raise FileNotFoundError(f"FOMO directory not found: {args.fomo_dir}")

    print("Reading existing training subjects...")
    train_subjects = get_training_subject_ids(lr_train_dir)
    print(f"  {len(train_subjects)} training subjects to exclude\n")

    test_files = pick_test_subjects(
        args.fomo_dir, args.sequence, train_subjects,
        args.num_subjects, args.seed,
    )

    if not test_files:
        print("No test files found. Check --fomo-dir and --sequence.")
        return

    info = load_info(info_path)
    axis_idx = AXIS_MAP[args.axis]

    print(f"\nProcessing test split ({args.axis}, scale={args.scale_factor}x)...")
    info = process_2d(
        test_files,
        split="test",
        out_dir=args.processed_dir,
        axis=axis_idx,
        axis_name=args.axis,
        scale_factor=args.scale_factor,
        info=info,
        save_samples=False,
    )

    save_info(info_path, info)
    print(f"\nTest set: {info['num_episodes']} episodes, {info['num_steps']} samples")
    print(f"  full_res: {args.processed_dir / 'full_res' / 'test'}")
    print(f"  low_res:  {lr_test_dir}")


if __name__ == "__main__":
    main()
