#!/usr/bin/env python3
"""Turn FOMO60K NKI T1 scans into the processed MRI dataset format.

This reproduces gs://hendrik_mri_super_resolution/mri_denoiser/processed_data_mri
bit-exactly (checked on sub_8112 ses_1 and sub_10915 ses_1):

  - the raw ``t1.nii.gz`` is RAS, (176, 256, 256), sagittal along axis 0;
    no resampling, cropping or padding
  - p1, p99 = 1st / 99th percentile of the NON-ZERO voxels of the volume
  - HR = uint8((clip(vol, p1, p99) - p1) / (p99 - p1) * 255), truncating
  - HR frame i = rot90(HR[i])
  - LR = floor of the 2x2 block mean of each HR frame

Every session is written to ``<out>/{low_res,full_res}/train`` with
``info.pt`` and ``episode_mapping.json``; ``scripts/make_split.py`` then splits
that pool by subject.

FOMO60K is now ``FOMO-MRI/FOMO50K`` on the Hugging Face Hub. It is gated:
accept its terms on the website and run ``huggingface-cli login`` first.

Usage:
    # 1. pick subjects (one session each) from the Hub file listing
    python scripts/preprocess_fomo.py select --num-subjects 650 --seed 0 --sessions sessions.txt
    # 2. download only those T1 scans
    python scripts/preprocess_fomo.py download --sessions sessions.txt --fomo-root ~/data/FOMO50K
    # 3. preprocess them into a pool, then split it by subject
    python scripts/preprocess_fomo.py preprocess --sessions sessions.txt --fomo-root ~/data/FOMO50K \
        --out processed_data_mri_pool
    python scripts/make_split.py --src processed_data_mri_pool --out processed_data_mri_resub \
        --num-test 70 --num-val 6 --num-train 500 --require-num-slices 176

On a tight disk, skip step 2 and pass ``--fetch --delete-raw`` to step 3 so
only one raw scan is on disk at a time, and pass ``--link`` to make_split.py.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch

from make_split import episode_path, write_index

REPO_ID = "FOMO-MRI/FOMO50K"
NKI_DIR = "PT015_NKI"
T1_RE = re.compile(rf"^{NKI_DIR}/sub_(\d+)/ses_(\d+)/t1\.nii\.gz$")
SESSION_RE = re.compile(r"^sub_(\d+)/ses_(\d+)$")
EXPECTED_SHAPE = (176, 256, 256)


def normalize(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float64)
    p1, p99 = np.percentile(vol[vol > 0], [1, 99])
    x = np.clip(vol, p1, p99)
    x = (x - p1) / (p99 - p1) * 255
    return x.astype(np.uint8)


def hr_frames(vol: np.ndarray) -> np.ndarray:
    x = normalize(vol)
    return np.stack([np.rot90(x[i]) for i in range(x.shape[0])])


def lr_frames(hr: np.ndarray) -> np.ndarray:
    n, h, w = hr.shape
    blocks = hr.reshape(n, h // 2, 2, w // 2, 2).astype(np.float64).mean(axis=(2, 4))
    return np.floor(blocks).astype(np.uint8)


def read_sessions(path: Path) -> list[tuple[str, int]]:
    out = []
    for line in path.read_text().split():
        m = SESSION_RE.match(line)
        if m is None:
            raise SystemExit(f"Bad line in {path}: {line!r} (expected sub_<id>/ses_<n>)")
        out.append((m[1], int(m[2])))
    return out


def t1_path(sub: str, ses: int) -> str:
    return f"{NKI_DIR}/sub_{sub}/ses_{ses}/t1.nii.gz"


def cmd_select(args: argparse.Namespace) -> None:
    from huggingface_hub import list_repo_files

    sessions: dict[str, list[int]] = defaultdict(list)
    for f in list_repo_files(REPO_ID, repo_type="dataset"):
        m = T1_RE.match(f)
        if m:
            sessions[m[1]].append(int(m[2]))
    subjects = sorted(sessions, key=int)
    print(f"{len(subjects)} NKI subjects, {sum(map(len, sessions.values()))} sessions with a T1")
    if args.num_subjects > len(subjects):
        raise SystemExit(f"Only {len(subjects)} subjects available")

    rng = random.Random(args.seed)
    rng.shuffle(subjects)
    chosen = sorted(subjects[:args.num_subjects], key=int)
    lines = [f"sub_{sub}/ses_{min(sessions[sub])}" for sub in chosen]  # lowest session number
    args.sessions.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} sessions to {args.sessions}")


def cmd_download(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    patterns = [t1_path(sub, ses) for sub, ses in read_sessions(args.sessions)]
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=args.fomo_root, allow_patterns=patterns)
    missing = [p for p in patterns if not (args.fomo_root / p).is_file()]
    print(f"Downloaded {len(patterns) - len(missing)}/{len(patterns)} T1 scans to {args.fomo_root}")
    for p in missing:
        print(f"  missing {p}")


def cmd_preprocess(args: argparse.Namespace) -> None:
    lr_out = args.out / "low_res" / "train"
    hr_out = args.out / "full_res" / "train"
    hr_out.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    lengths: list[int] = []
    for sub, ses in read_sessions(args.sessions):
        src = args.fomo_root / t1_path(sub, ses)
        if args.fetch and not src.is_file():
            from huggingface_hub import hf_hub_download
            hf_hub_download(REPO_ID, t1_path(sub, ses), repo_type="dataset", local_dir=args.fomo_root)
        vol = np.asanyarray(nib.load(src).dataobj)
        if args.delete_raw:
            src.unlink()
        if not args.any_shape and vol.shape != EXPECTED_SHAPE:
            print(f"  skip sub_{sub}/ses_{ses}: shape {vol.shape}")
            continue
        hr = hr_frames(vol)
        lr = lr_frames(hr)

        name = f"sub_{sub}_ses_{ses}_t1_sagittal.hdf5"
        with h5py.File(hr_out / name, "w") as f:
            for i, frame in enumerate(hr):
                f.create_dataset(f"frame_{i}_x", data=frame)

        file_id = f"train/{name}"
        dst = episode_path(lr_out, len(lengths))
        dst.parent.mkdir(parents=True, exist_ok=True)
        obs = torch.from_numpy(np.ascontiguousarray(lr[:, None]))
        torch.save({"obs": obs, "info": {"original_file_id": file_id}}, dst)
        mapping[str(dst.relative_to(lr_out))] = file_id
        lengths.append(len(hr))

    summary = write_index(lr_out, mapping, lengths)
    print(f"Wrote {summary['num_volumes']} volumes ({summary['num_slices']} slices) to {args.out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="Pick subjects from the Hub listing, one session each")
    s.add_argument("--num-subjects", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--sessions", type=Path, required=True, help="Output list of sub_<id>/ses_<n>")
    s.set_defaults(func=cmd_select)

    d = sub.add_parser("download", help="Download the listed T1 scans")
    d.add_argument("--sessions", type=Path, required=True)
    d.add_argument("--fomo-root", type=Path, required=True)
    d.set_defaults(func=cmd_download)

    r = sub.add_parser("preprocess", help="Write the listed scans as one processed pool")
    r.add_argument("--sessions", type=Path, required=True)
    r.add_argument("--fomo-root", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--any-shape", action="store_true", help=f"Keep volumes that are not {EXPECTED_SHAPE}")
    r.add_argument("--fetch", action="store_true", help="Download each scan just before processing it")
    r.add_argument("--delete-raw", action="store_true", help="Delete each raw scan once it is loaded (saves disk)")
    r.set_defaults(func=cmd_preprocess)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
