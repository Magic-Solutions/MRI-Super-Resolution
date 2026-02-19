"""Convert MKV + actions ndjson into the DIAMOND processed_data format.

Produces the same directory layout as process_csgo_tar_files.py:
  out_dir/
    full_res/
      <stem>.hdf5          # HDF5 with frame_i_x, frame_i_y, frame_i_xaux, frame_i_helperarr
    low_res/
      train/
        info.pt
        000/00/0/0.pt
      test/
        ...

Usage:
    python convert_to_csgo.py <mkv> <actions_ndjson> <out_dir> [--split test]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as T


FULL_RES = (150, 280)
LOW_RES = (30, 56)
ACTION_DIM = 51  # 11 keys + 2 clicks + 23 mouse_x + 15 mouse_y
XAUX_DIM = 54
HELPER_DIM = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mkv", type=Path, help="Input .mkv file (RGB stream)")
    parser.add_argument("actions", type=Path, help="Actions .ndjson file")
    parser.add_argument("out_dir", type=Path, help="Root output directory (e.g. src/processed_data_omgrab)")
    parser.add_argument(
        "--split", default="train", choices=["train", "test"],
        help="Which split to write into (default: train)",
    )
    parser.add_argument(
        "--episode-id", type=int, default=None,
        help="Override episode ID (default: auto-increment based on existing episodes)",
    )
    return parser.parse_args()


def load_actions(path: Path) -> dict[int, list[float]]:
    actions: dict[int, list[float]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            actions[entry["f"]] = entry["action"]
    return actions


def extract_frames(mkv_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(mkv_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {mkv_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def get_episode_path(directory: Path, episode_id: int) -> Path:
    n = 3
    powers = np.arange(n)
    subfolders = np.floor((episode_id % 10 ** (1 + powers)) / 10**powers) * 10**powers
    subfolders = [int(x) for x in subfolders[::-1]]
    subfolders = "/".join([f"{x:0{n - i}d}" for i, x in enumerate(subfolders)])
    return directory / subfolders / f"{episode_id}.pt"


def save_full_res_hdf5(
    path: Path,
    frames: list[np.ndarray],
    actions: dict[int, list[float]],
    n: int,
) -> None:
    """Save in the same HDF5 format as the original CSGO dataset.

    Each frame has 4 datasets: frame_{i}_x (H,W,3 uint8), frame_{i}_y (51, float64),
    frame_{i}_xaux (54, float64), frame_{i}_helperarr (2, float64).
    The x frames are resized to 150x280 to match CSGO full_res.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for i in range(n):
            rgb = cv2.resize(frames[i], (FULL_RES[1], FULL_RES[0]), interpolation=cv2.INTER_AREA)
            f.create_dataset(f"frame_{i}_x", data=rgb, dtype=np.uint8)

            act = np.zeros(ACTION_DIM, dtype=np.float64)
            if i in actions:
                act[:] = actions[i]
            f.create_dataset(f"frame_{i}_y", data=act)

            f.create_dataset(f"frame_{i}_xaux", data=np.zeros(XAUX_DIM, dtype=np.float64))
            f.create_dataset(f"frame_{i}_helperarr", data=np.zeros(HELPER_DIM, dtype=np.float64))


def save_episode(path: Path, obs: torch.Tensor, act: torch.Tensor, info: dict) -> None:
    n = obs.size(0)
    episode = {
        "obs": obs,
        "act": act,
        "rew": torch.zeros(n, dtype=torch.float32),
        "end": torch.zeros(n, dtype=torch.uint8),
        "trunc": torch.zeros(n, dtype=torch.uint8),
        "info": info,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(episode, tmp)
    tmp.rename(path)


def load_info(info_path: Path) -> dict:
    if info_path.is_file():
        return torch.load(info_path, weights_only=False)
    return {
        "is_static": False,
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


def main() -> None:
    args = parse_args()
    full_res_dir = args.out_dir / "full_res"
    low_res_dir = args.out_dir / "low_res"
    split_dir = low_res_dir / args.split
    info_path = split_dir / "info.pt"

    print(f"Extracting RGB frames from {args.mkv} ...")
    frames = extract_frames(args.mkv)
    n_frames = len(frames)
    print(f"  {n_frames} frames, resolution {frames[0].shape[1]}x{frames[0].shape[0]}")

    print(f"Loading actions from {args.actions} ...")
    actions = load_actions(args.actions)
    n_actions = len(actions)
    print(f"  {n_actions} action entries")

    if n_frames != n_actions:
        print(f"WARNING: frame count ({n_frames}) != action count ({n_actions}), using min")
    n = min(n_frames, n_actions)

    # --- full_res HDF5 ---
    hdf5_path = full_res_dir / f"{args.mkv.stem}.hdf5"
    print(f"Saving full_res HDF5 to {hdf5_path} ...")
    save_full_res_hdf5(hdf5_path, frames, actions, n)
    print(f"  {n} frames, {FULL_RES[0]}x{FULL_RES[1]}, 4 datasets per frame")

    # --- low_res .pt episode ---
    obs_list = []
    for i in range(n):
        t = torch.from_numpy(frames[i]).permute(2, 0, 1).float()
        t = T.resize(t, LOW_RES, interpolation=T.InterpolationMode.BICUBIC)
        obs_list.append(t.clamp(0, 255).byte())
    obs = torch.stack(obs_list)
    print(f"  low_res obs: {obs.shape}, dtype={obs.dtype}")

    act = torch.zeros(n, ACTION_DIM, dtype=torch.float64)
    for i in range(n):
        if i in actions:
            act[i] = torch.tensor(actions[i], dtype=torch.float64)

    info = load_info(info_path)
    episode_id = args.episode_id if args.episode_id is not None else info["num_episodes"]

    episode_path = get_episode_path(split_dir, episode_id)
    source_id = f"{args.mkv.stem}"
    save_episode(episode_path, obs, act, info={"original_file_id": source_id})
    print(f"  Saved low_res episode {episode_id} to {episode_path}")

    info["start_idx"] = np.concatenate((info["start_idx"], np.array([info["num_steps"]])))
    info["lengths"] = np.concatenate((info["lengths"], np.array([n])))
    info["num_steps"] += n
    info["num_episodes"] += 1
    info["counter_rew"].update(Counter({0.0: n}))
    info["counter_end"].update(Counter({0: n}))

    save_info(info_path, info)
    print(f"  Updated {info_path}: {info['num_episodes']} episodes, {info['num_steps']} steps")
    print("Done.")


if __name__ == "__main__":
    main()
