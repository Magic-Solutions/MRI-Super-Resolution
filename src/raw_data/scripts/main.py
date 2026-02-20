"""Batch-process all MKV + hand-landmark pairs into DIAMOND training data.

Scans raw_data/train/ and raw_data/test/ for file pairs:
    <uuid>.mkv + <uuid>_hands.ndjson

For each split:
  1. Infers actions from hand landmarks (headless, no GUI)
  2. Converts to DIAMOND processed_data format (full_res HDF5 + low_res .pt)
  3. Splits clips longer than CHUNK_SIZE frames into multiple episodes

Usage:
    python main.py <out_dir> [--chunk-size 1000]
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import shutil

import av
import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as T

RAW_DIR = Path(__file__).resolve().parent.parent

FULL_RES = (150, 280)
LOW_RES = (30, 56)
CHUNK_SIZE = 1000

# Action vector layout (51 dims)
N_KEYS = 11
N_CLICKS = 2
N_MOUSE_X = 23
N_MOUSE_Y = 15
ACTION_DIM = N_KEYS + N_CLICKS + N_MOUSE_X + N_MOUSE_Y
MOUSE_X_ZERO_IDX = 11
MOUSE_Y_ZERO_IDX = 7
XAUX_DIM = 54
HELPER_DIM = 2

# MediaPipe hand landmark indices
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

DIRECTION_THRESHOLD = 0.008
GRIP_THRESHOLD = 1.1
SMOOTHING = 0.4


# ---------------------------------------------------------------------------
# Action inference from hand landmarks (headless)
# ---------------------------------------------------------------------------

@dataclass
class ActionState:
    prev_x: float | None = None
    prev_y: float | None = None
    smooth_dx: float = 0.0
    smooth_dy: float = 0.0
    pending_dx: float = 0.0
    pending_dy: float = 0.0
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    grip: bool = False


def get_right_hand(hands: list) -> dict | None:
    for h in hands:
        if h.get("handedness") == "Right":
            return h
    return None


def detect_grip(landmarks: list) -> bool:
    wrist = np.array(landmarks[WRIST][:2])
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    mcps = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    tip_dists = [np.linalg.norm(np.array(landmarks[t][:2]) - wrist) for t in tips]
    mcp_dists = [np.linalg.norm(np.array(landmarks[m][:2]) - wrist) for m in mcps]
    ratios = [td / md if md > 1e-6 else 2.0 for td, md in zip(tip_dists, mcp_dists)]
    return np.mean(ratios) < GRIP_THRESHOLD


def update_actions(state: ActionState, landmarks: list | None) -> None:
    state.left = state.pending_dx < -DIRECTION_THRESHOLD
    state.right = state.pending_dx > DIRECTION_THRESHOLD
    state.up = state.pending_dy < -DIRECTION_THRESHOLD
    state.down = state.pending_dy > DIRECTION_THRESHOLD

    if landmarks is None:
        state.prev_x = state.prev_y = None
        state.smooth_dx = state.smooth_dy = 0.0
        state.pending_dx = state.pending_dy = 0.0
        state.grip = False
        return

    state.grip = detect_grip(landmarks)
    x, y = landmarks[INDEX_MCP][0], landmarks[INDEX_MCP][1]
    if state.prev_x is not None:
        raw_dx = x - state.prev_x
        raw_dy = y - state.prev_y
        state.smooth_dx = SMOOTHING * raw_dx + (1 - SMOOTHING) * state.smooth_dx
        state.smooth_dy = SMOOTHING * raw_dy + (1 - SMOOTHING) * state.smooth_dy

    state.pending_dx = state.smooth_dx
    state.pending_dy = state.smooth_dy
    state.prev_x, state.prev_y = x, y


def encode_action(state: ActionState) -> list[float]:
    vec = [0.0] * ACTION_DIM
    if state.up:
        vec[0] = 1.0
    if state.left:
        vec[1] = 1.0
    if state.down:
        vec[2] = 1.0
    if state.right:
        vec[3] = 1.0
    if state.grip:
        vec[4] = 1.0
    vec[N_KEYS + N_CLICKS + MOUSE_X_ZERO_IDX] = 1.0
    vec[N_KEYS + N_CLICKS + N_MOUSE_X + MOUSE_Y_ZERO_IDX] = 1.0
    return vec


def infer_actions(hands_path: Path, n_frames: int) -> list[list[float]]:
    """Run action inference headlessly over all frames."""
    hands_by_frame: dict[int, list] = {}
    with open(hands_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            hands_by_frame[entry["f"]] = entry["h"]

    state = ActionState()
    actions: list[list[float]] = []
    for i in range(n_frames):
        hands = hands_by_frame.get(i, [])
        right = get_right_hand(hands)
        update_actions(state, right["landmarks"] if right else None)
        actions.append(encode_action(state))
    return actions


# ---------------------------------------------------------------------------
# Dataset writing
# ---------------------------------------------------------------------------

def extract_frames(mkv_path: Path) -> list[np.ndarray]:
    """Extract RGB frames from MKV track 0 using OpenCV."""
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


def extract_depth_frames(mkv_path: Path) -> list[np.ndarray]:
    """Extract 16-bit depth frames from MKV track 1 (FFV1, gray16le) using pyav.

    Returns a list of uint16 grayscale arrays (H, W).
    """
    container = av.open(str(mkv_path))
    stream = container.streams.video[1]
    frames = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()  # (H, W) uint16 for gray16le
        frames.append(arr)
    container.close()
    return frames


def normalize_depth_to_uint8(
    depth_frames: list[np.ndarray],
    lo_pct: float = 3.0,
    hi_pct: float = 97.0,
) -> list[np.ndarray]:
    """Normalize 16-bit depth frames to 8-bit per-frame with percentile clipping.

    Zeros (invalid/no-return pixels) are kept as zero.  Non-zero values are
    clipped to [lo_pct, hi_pct] percentiles of the non-zero pixels in each
    frame, then linearly mapped to [1, 255].
    """
    out = []
    for d in depth_frames:
        result = np.zeros(d.shape, dtype=np.uint8)
        mask = d > 0
        if not mask.any():
            out.append(result)
            continue
        vals = d[mask].astype(np.float32)
        p_lo = np.percentile(vals, lo_pct)
        p_hi = np.percentile(vals, hi_pct)
        if p_hi <= p_lo:
            out.append(result)
            continue
        clipped = np.clip(d.astype(np.float32), p_lo, p_hi)
        normed = (clipped - p_lo) / (p_hi - p_lo)
        result[mask] = (normed[mask] * 254 + 1).astype(np.uint8)
        out.append(result)
    return out


def merge_rgbd(
    rgb_frames: list[np.ndarray],
    depth_frames: list[np.ndarray],
) -> list[np.ndarray]:
    """Merge RGB (H, W, 3) and depth (H_d, W_d) into RGBD (H, W, 4).

    Depth is resized to match RGB dimensions before concatenation.
    """
    merged = []
    for rgb, depth in zip(rgb_frames, depth_frames):
        h, w = rgb.shape[:2]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        rgbd = np.concatenate([rgb, depth[:, :, np.newaxis]], axis=2)
        merged.append(rgbd)
    return merged


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
    actions: list[list[float]],
    start: int,
    end: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for out_i, src_i in enumerate(range(start, end)):
            rgb = cv2.resize(frames[src_i], (FULL_RES[1], FULL_RES[0]), interpolation=cv2.INTER_AREA)
            f.create_dataset(f"frame_{out_i}_x", data=rgb, dtype=np.uint8)
            act = np.array(actions[src_i], dtype=np.float64)
            f.create_dataset(f"frame_{out_i}_y", data=act)
            f.create_dataset(f"frame_{out_i}_xaux", data=np.zeros(XAUX_DIM, dtype=np.float64))
            f.create_dataset(f"frame_{out_i}_helperarr", data=np.zeros(HELPER_DIM, dtype=np.float64))


def save_low_res_episode(
    split_dir: Path,
    episode_id: int,
    frames: list[np.ndarray],
    actions: list[list[float]],
    start: int,
    end: int,
    source_id: str,
) -> None:
    n = end - start
    obs_list = []
    for i in range(start, end):
        t = torch.from_numpy(frames[i]).permute(2, 0, 1).float()
        t = T.resize(t, LOW_RES, interpolation=T.InterpolationMode.BICUBIC)
        obs_list.append(t.clamp(0, 255).byte())
    obs = torch.stack(obs_list)

    act = torch.zeros(n, ACTION_DIM, dtype=torch.float64)
    for j, i in enumerate(range(start, end)):
        act[j] = torch.tensor(actions[i], dtype=torch.float64)

    episode = {
        "obs": obs,
        "act": act,
        "rew": torch.zeros(n, dtype=torch.float32),
        "end": torch.zeros(n, dtype=torch.uint8),
        "trunc": torch.zeros(n, dtype=torch.uint8),
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def find_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for mkv in sorted(raw_dir.glob("*.mkv")):
        hands = mkv.with_name(f"{mkv.stem}_hands.ndjson")
        if hands.exists():
            pairs.append((mkv, hands))
        else:
            print(f"  SKIP {mkv.name}: no matching _hands.ndjson")
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Root output directory (e.g. src/processed_data_omgrab)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Max frames per episode (default: {CHUNK_SIZE})")
    return parser.parse_args()


def process_split(
    split: str,
    raw_dir: Path,
    out_dir: Path,
    chunk_size: int,
) -> None:
    """Process a single split (train or test)."""
    split_raw = raw_dir / split
    if not split_raw.is_dir():
        print(f"\n  Skipping '{split}': {split_raw} does not exist")
        return

    full_res_dir = out_dir / "full_res"
    split_dir = out_dir / "low_res" / split
    info_path = split_dir / "info.pt"

    if split_dir.exists():
        print(f"  Clearing existing {split_dir} ...")
        shutil.rmtree(split_dir)

    print(f"\nScanning {split_raw} for MKV + hands.ndjson pairs ...")
    pairs = find_pairs(split_raw)
    if not pairs:
        print("  No file pairs found.")
        return
    print(f"  Found {len(pairs)} pair(s)\n")

    info = load_info(info_path)

    for mkv_path, hands_path in pairs:
        stem = mkv_path.stem
        print(f"{'='*60}")
        print(f"Processing: {stem}")

        print(f"  Extracting RGB frames from {mkv_path.name} ...")
        rgb_frames = extract_frames(mkv_path)
        n_rgb = len(rgb_frames)
        print(f"    {n_rgb} RGB frames, {rgb_frames[0].shape[1]}x{rgb_frames[0].shape[0]}")

        print(f"  Extracting depth frames from {mkv_path.name} (track 1) ...")
        depth_frames_raw = extract_depth_frames(mkv_path)
        n_depth = len(depth_frames_raw)
        print(f"    {n_depth} depth frames, {depth_frames_raw[0].shape[1]}x{depth_frames_raw[0].shape[0]}")

        n_frames = min(n_rgb, n_depth)
        rgb_frames = rgb_frames[:n_frames]
        depth_frames_raw = depth_frames_raw[:n_frames]

        print(f"  Normalizing depth (16-bit -> 8-bit) ...")
        depth_frames_u8 = normalize_depth_to_uint8(depth_frames_raw)

        print(f"  Merging RGB + Depth -> RGBD ...")
        frames = merge_rgbd(rgb_frames, depth_frames_u8)
        print(f"    {len(frames)} RGBD frames, shape {frames[0].shape}")

        print(f"  Inferring actions from {hands_path.name} ...")
        actions = infer_actions(hands_path, n_frames)
        print(f"    {len(actions)} actions inferred")

        n_full_chunks = n_frames // chunk_size
        tail = n_frames - n_full_chunks * chunk_size

        if n_full_chunks == 0:
            print(f"  SKIP: clip too short ({n_frames}f < {chunk_size})")
            continue

        if tail > 0:
            print(f"  {n_full_chunks} full chunk(s), discarding {tail} leftover frames")
        else:
            print(f"  {n_full_chunks} full chunk(s)")

        for chunk_idx in range(n_full_chunks):
            start = chunk_idx * chunk_size
            end = start + chunk_size
            episode_id = info["num_episodes"]

            hdf5_subdir = "omgrab"
            hdf5_name = f"omgrab_{episode_id}.hdf5"
            hdf5_path = full_res_dir / hdf5_subdir / hdf5_name
            file_id = f"{hdf5_subdir}/{hdf5_name}"
            save_full_res_hdf5(hdf5_path, frames, actions, start, end)

            save_low_res_episode(
                split_dir, episode_id, frames, actions, start, end, file_id,
            )

            info["start_idx"] = np.concatenate((info["start_idx"], np.array([info["num_steps"]])))
            info["lengths"] = np.concatenate((info["lengths"], np.array([chunk_size])))
            info["num_steps"] += chunk_size
            info["num_episodes"] += 1
            info["counter_rew"].update(Counter({0.0: chunk_size}))
            info["counter_end"].update(Counter({0: chunk_size}))

            ep_path = get_episode_path(split_dir, episode_id)
            print(f"    chunk {chunk_idx}: frames [{start}:{end}] -> episode "
                  f"{episode_id} ({ep_path.name})")

    save_info(info_path, info)

    print(f"\n  {split} done: {info['num_episodes']} episodes, {info['num_steps']} steps")


def main() -> None:
    args = parse_args()
    full_res_dir = args.out_dir / "full_res"

    if full_res_dir.exists():
        print(f"Clearing existing {full_res_dir} ...")
        shutil.rmtree(full_res_dir)

    for split in ("train", "test"):
        print(f"\n{'#'*60}")
        print(f"# Split: {split}")
        print(f"{'#'*60}")
        process_split(split, RAW_DIR, args.out_dir, args.chunk_size)

    print(f"\n{'='*60}")
    print(f"Dataset written to {args.out_dir}")
    print(f"  full_res: {full_res_dir}")
    for split in ("train", "test"):
        split_dir = args.out_dir / "low_res" / split
        info_path = split_dir / "info.pt"
        if info_path.exists():
            info = torch.load(info_path, weights_only=False)
            print(f"  {split}: {info['num_episodes']} episodes, {info['num_steps']} steps")
        else:
            print(f"  {split}: (empty)")


if __name__ == "__main__":
    main()
