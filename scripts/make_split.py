#!/usr/bin/env python3
"""Build a subject-level train / val / test split from processed MRI data.

Reads an existing processed dataset (``low_res/train`` episodes, their
``episode_mapping.json`` and the matching ``full_res/train`` HDF5 files) and
writes two new processed datasets that both the 2.5D and the 3D trainers read
the same way:

    <out>/training/    low_res/{train,test}  full_res/{train,test}
    <out>/evaluation/  low_res/test          full_res/test
    <out>/split.json

``training/*/test`` holds the VALIDATION subjects, because the 2.5D trainer
evaluates on ``<data>/test`` during training and ``train_3d.py`` takes it via
``--lr-test/--hr-test``. The held-out TEST subjects live only in
``evaluation/`` so they are never seen while training or selecting checkpoints.

Each split lists every one of its episodes in both ``info.pt`` and
``episode_mapping.json``, so the 2.5D trainer (which reads ``info.pt``) and the
3D trainer (which reads ``episode_mapping.json``) see exactly the same volumes.

Usage:
    python scripts/make_split.py \
        --src processed_data_mri_pool --out processed_data_mri_resub \
        --num-test 70 --num-val 6 --num-train 500 --seed 0 --require-num-slices 176
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

FILE_RE = re.compile(r"sub_(?P<sub>\d+)_ses_(?P<ses>\d+)_")


def episode_path(directory: Path, episode_id: int) -> Path:
    """Same layout as ``data.Dataset._get_episode_path``."""
    n = 3
    powers = np.arange(n)
    subfolders = np.floor((episode_id % 10 ** (1 + powers)) / 10**powers) * 10**powers
    subfolders = [int(x) for x in subfolders[::-1]]
    subfolders = "/".join([f"{x:0{n - i}d}" for i, x in enumerate(subfolders)])
    return directory / subfolders / f"{episode_id}.pt"


def load_sessions(src: Path) -> dict[str, list[tuple[int, Path, Path]]]:
    """Return {subject_id: [(session, lr_episode_path, hr_hdf5_path), ...]}."""
    lr_dir = src / "low_res" / "train"
    hr_dir = src / "full_res" / "train"
    with open(lr_dir / "episode_mapping.json") as f:
        mapping = json.load(f)

    sessions: dict[str, list[tuple[int, Path, Path]]] = defaultdict(list)
    for rel_path, file_id in sorted(mapping.items()):
        lr_path = lr_dir / rel_path
        hr_path = hr_dir / Path(file_id).name
        m = FILE_RE.search(Path(file_id).name)
        if m is None or not lr_path.is_file() or not hr_path.is_file():
            print(f"  skip {rel_path} -> {file_id} (missing file or unexpected name)")
            continue
        sessions[m["sub"]].append((int(m["ses"]), lr_path, hr_path))
    for sub in sessions:
        sessions[sub].sort()
    return sessions


def write_split(
    items: list[tuple[str, int, Path, Path]],
    lr_out: Path,
    hr_out: Path,
    split_dir_name: str,
    link: bool = False,
) -> dict:
    """Copy episodes (renumbered from 0) and HDF5 files; write info.pt and episode_mapping.json."""
    lr_out.mkdir(parents=True, exist_ok=True)
    hr_out.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    lengths: list[int] = []
    for episode_id, (_, _, lr_path, hr_path) in enumerate(items):
        d = torch.load(lr_path, weights_only=False, map_location="cpu")
        file_id = f"{split_dir_name}/{hr_path.name}"
        d.setdefault("info", {})["original_file_id"] = file_id
        dst = episode_path(lr_out, episode_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(d, dst)
        if link:
            os.link(hr_path, hr_out / hr_path.name)  # same file system; no extra disk
        else:
            shutil.copy2(hr_path, hr_out / hr_path.name)
        mapping[str(dst.relative_to(lr_out))] = file_id
        lengths.append(int(d["obs"].shape[0]))
    return write_index(lr_out, mapping, lengths)


def write_index(lr_out: Path, mapping: dict[str, str], lengths: list[int]) -> dict:
    """Write the info.pt (2.5D trainer) and episode_mapping.json (3D trainer) of one split."""
    lengths_arr = np.array(lengths, dtype=np.int64)
    info = {
        "is_static": True,
        "num_episodes": len(lengths),
        "num_steps": int(lengths_arr.sum()),
        "start_idx": np.concatenate(([0], np.cumsum(lengths_arr)[:-1])).astype(np.int64) if lengths else np.array([], dtype=np.int64),
        "lengths": lengths_arr,
        "counter_rew": Counter(),
        "counter_end": Counter(),
    }
    torch.save(info, lr_out / "info.pt")
    with open(lr_out / "episode_mapping.json", "w") as f:
        json.dump(mapping, f, indent=1)
    return {"num_volumes": info["num_episodes"], "num_slices": info["num_steps"]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="Existing processed dataset (expects low_res/train, full_res/train)")
    p.add_argument("--out", type=Path, required=True, help="Output root for training/, evaluation/ and split.json")
    p.add_argument("--num-test", type=int, default=30)
    p.add_argument("--num-val", type=int, default=15)
    p.add_argument("--num-train", type=int, default=None, help="Cap on training subjects (default: all remaining)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all-train-sessions", action="store_true",
                   help="Keep every session of training subjects (val/test always use one session per subject)")
    p.add_argument("--require-num-slices", type=int, default=None,
                   help="Only keep volumes with exactly this many slices (176 for the NKI T1 scans)")
    p.add_argument("--link", action="store_true",
                   help="Hard-link the HDF5 files instead of copying them (src and out on the same disk)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sessions = load_sessions(args.src)

    if args.require_num_slices is not None:
        kept: dict[str, list[tuple[int, Path, Path]]] = {}
        for sub, items in sessions.items():
            ok = [it for it in items
                  if torch.load(it[1], weights_only=False, map_location="cpu")["obs"].shape[0] == args.require_num_slices]
            if ok:
                kept[sub] = ok
        print(f"Slice filter ({args.require_num_slices}): {len(kept)}/{len(sessions)} subjects kept")
        sessions = kept

    subjects = sorted(sessions, key=int)
    print(f"{len(subjects)} subjects, {sum(len(v) for v in sessions.values())} sessions in {args.src}")
    if args.num_test + args.num_val >= len(subjects):
        raise SystemExit(f"Not enough subjects for {args.num_test} test + {args.num_val} val")

    rng = random.Random(args.seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    test_subs = sorted(shuffled[:args.num_test], key=int)
    val_subs = sorted(shuffled[args.num_test:args.num_test + args.num_val], key=int)
    train_subs = shuffled[args.num_test + args.num_val:]
    if args.num_train is not None:
        train_subs = train_subs[:args.num_train]  # cap before sorting so the subset stays random
    train_subs = sorted(train_subs, key=int)

    def pick(subs: list[str], all_sessions: bool) -> list[tuple[str, int, Path, Path]]:
        out = []
        for sub in subs:
            chosen = sessions[sub] if all_sessions else sessions[sub][:1]  # lowest session number
            out.extend((sub, ses, lr, hr) for ses, lr, hr in chosen)
        return out

    splits = {
        "train": pick(train_subs, args.all_train_sessions),
        "val": pick(val_subs, False),
        "test": pick(test_subs, False),
    }

    summary = {
        "train": write_split(splits["train"], args.out / "training/low_res/train", args.out / "training/full_res/train", "train", args.link),
        "val": write_split(splits["val"], args.out / "training/low_res/test", args.out / "training/full_res/test", "test", args.link),
        "test": write_split(splits["test"], args.out / "evaluation/low_res/test", args.out / "evaluation/full_res/test", "test", args.link),
    }

    split_record = {
        "source": str(args.src),
        "seed": args.seed,
        "one_session_per_subject": {"train": not args.all_train_sessions, "val": True, "test": True},
        "require_num_slices": args.require_num_slices,
        "note": "training/*/test holds the validation subjects; evaluation/*/test holds the held-out test subjects.",
        **{
            name: {
                "subjects": [int(s) for s in subs],
                "sessions": [f"sub_{sub}_ses_{ses}" for sub, ses, _, _ in splits[name]],
                **summary[name],
            }
            for name, subs in (("train", train_subs), ("val", val_subs), ("test", test_subs))
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "split.json", "w") as f:
        json.dump(split_record, f, indent=1)

    assert not (set(train_subs) & set(val_subs) or set(train_subs) & set(test_subs) or set(val_subs) & set(test_subs))
    for name in ("train", "val", "test"):
        print(f"{name:>5}: {len(split_record[name]['subjects'])} subjects, "
              f"{summary[name]['num_volumes']} volumes, {summary[name]['num_slices']} slices")
    print(f"Wrote {args.out / 'split.json'}")


if __name__ == "__main__":
    main()
