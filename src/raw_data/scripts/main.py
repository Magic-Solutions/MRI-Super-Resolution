"""Batch-process MKV files into DIAMOND training data.

Scans raw_data/train/ and raw_data/test/ for .mkv files containing:
    - Track 0: RGB video (H.264)
    - Track 1: Depth video (FFV1, gray16le)
    - Track 3: Hand landmarks (SRT subtitle, JSON per packet)

For each split:
  1. Extracts hand landmarks from subtitle stream and infers actions
  2. Converts to DIAMOND processed_data format (full_res HDF5 + low_res .pt)
  3. Splits clips longer than CHUNK_SIZE frames into multiple episodes

Usage:
    python main.py <out_dir> [--chunk-size 1000]
"""

import gc
import json
import random
import subprocess
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
DEPTH_MIN_MM = 200
DEPTH_MAX_MM = 3000

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

GRIP_THRESHOLD = 1.1
VELOCITY_EMA_ALPHA = 0.65
VELOCITY_GAIN = 1.5
SPEED_ON_THRESHOLD = 0.005
SPEED_OFF_THRESHOLD = 0.0035
DIAGONAL_RATIO_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Action inference from hand landmarks (headless)
# ---------------------------------------------------------------------------

@dataclass
class ActionState:
    prev_x: float | None = None
    prev_y: float | None = None
    smooth_dx: float = 0.0
    smooth_dy: float = 0.0
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    grip: bool = False
    moving: bool = False
    ema_samples: int = 0


def get_right_hand(hands: list) -> dict | None:
    # Prefer explicit right hand, but gracefully fall back when handedness
    # labels are delayed, flipped, or missing in early frames.
    if not hands:
        return None
    for h in hands:
        if h.get("handedness") == "Right":
            return h
    if len(hands) == 1:
        return hands[0]
    return max(hands, key=lambda h: float(h.get("score", 0.0)))


def detect_grip(landmarks: list) -> bool:
    wrist = np.array(landmarks[WRIST][:2])
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    mcps = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    tip_dists = [np.linalg.norm(np.array(landmarks[t][:2]) - wrist) for t in tips]
    mcp_dists = [np.linalg.norm(np.array(landmarks[m][:2]) - wrist) for m in mcps]
    ratios = [td / md if md > 1e-6 else 2.0 for td, md in zip(tip_dists, mcp_dists)]
    return np.mean(ratios) < GRIP_THRESHOLD


def get_tracking_point(landmarks: list) -> tuple[float, float]:
    """Use a palm-center proxy to reduce single-keypoint jitter."""
    track_ids = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    pts = np.array([landmarks[i][:2] for i in track_ids], dtype=np.float32)
    center = pts.mean(axis=0)
    return float(center[0]), float(center[1])


def movement_from_velocity(dx: float, dy: float, was_moving: bool) -> tuple[bool, bool, bool, bool, bool]:
    """Map smoothed image-space velocity to movement keys with hysteresis."""
    speed = float(np.hypot(dx, dy))
    moving = speed > (SPEED_OFF_THRESHOLD if was_moving else SPEED_ON_THRESHOLD)
    if not moving:
        return False, False, False, False, False

    ax, ay = abs(dx), abs(dy)
    dominant = max(ax, ay)
    subordinate = min(ax, ay)
    ratio = subordinate / (dominant + 1e-8)

    if ratio < DIAGONAL_RATIO_THRESHOLD:
        # Minor axis is too weak: emit only dominant cardinal direction.
        if ay >= ax:
            up = dy < 0.0
            down = dy > 0.0
            left = False
            right = False
        else:
            left = dx < 0.0
            right = dx > 0.0
            up = False
            down = False
    else:
        # Axes are comparable: allow true diagonals.
        up = dy < 0.0
        down = dy > 0.0
        left = dx < 0.0
        right = dx > 0.0

    return left, right, up, down, moving


def update_actions(state: ActionState, landmarks: list | None) -> None:
    if landmarks is None:
        state.prev_x = state.prev_y = None
        state.smooth_dx = state.smooth_dy = 0.0
        state.ema_samples = 0
        state.grip = False
        state.moving = False
        state.left = state.right = state.up = state.down = False
        return

    state.grip = detect_grip(landmarks)
    x, y = get_tracking_point(landmarks)
    if state.prev_x is not None:
        raw_dx = x - state.prev_x
        raw_dy = y - state.prev_y
        state.smooth_dx = VELOCITY_EMA_ALPHA * raw_dx + (1 - VELOCITY_EMA_ALPHA) * state.smooth_dx
        state.smooth_dy = VELOCITY_EMA_ALPHA * raw_dy + (1 - VELOCITY_EMA_ALPHA) * state.smooth_dy
        state.ema_samples += 1
    state.prev_x, state.prev_y = x, y

    # Correct EMA startup bias so early movement is not artificially muted.
    if state.ema_samples > 0:
        bias = 1.0 - (1.0 - VELOCITY_EMA_ALPHA) ** state.ema_samples
        eff_dx = state.smooth_dx / max(bias, 1e-8)
        eff_dy = state.smooth_dy / max(bias, 1e-8)
    else:
        eff_dx = 0.0
        eff_dy = 0.0

    state.left, state.right, state.up, state.down, state.moving = movement_from_velocity(
        VELOCITY_GAIN * eff_dx, VELOCITY_GAIN * eff_dy, state.moving
    )


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


def extract_hand_landmarks(mkv_path: Path) -> dict[int, list]:
    """Extract hand landmarks from subtitle stream 3 using MKV timestamps.

    New export payloads no longer guarantee duplicated timing fields in JSON
    (e.g. f/t/dt), so we map packets to video frames using packet PTS and
    stream time bases.
    """
    def _extract_hands(entry: dict) -> list:
        if isinstance(entry.get("h"), list):
            return entry["h"]
        if isinstance(entry.get("hands"), list):
            return entry["hands"]
        payload = entry.get("payload")
        if isinstance(payload, dict):
            if isinstance(payload.get("h"), list):
                return payload["h"]
            if isinstance(payload.get("hands"), list):
                return payload["hands"]
        return []

    def _select_hand_stream_index(path: Path) -> int | None:
        probe = av.open(str(path))
        subtitle_indices = [i for i, s in enumerate(probe.streams) if s.type == "subtitle"]
        probe.close()
        for idx in subtitle_indices:
            c = av.open(str(path))
            stream = c.streams[idx]
            found = False
            checked = 0
            for pkt in c.demux(stream):
                if pkt.size == 0:
                    continue
                raw = bytes(pkt).decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                checked += 1
                if _extract_hands(entry):
                    found = True
                    break
                if checked >= 50:
                    break
            c.close()
            if found:
                return idx
        return None

    stream_idx = _select_hand_stream_index(mkv_path)
    if stream_idx is None:
        return {}

    container = av.open(str(mkv_path))
    video_stream = container.streams.video[0]
    stream = container.streams[stream_idx]
    fps = float(video_stream.average_rate) if video_stream.average_rate else 25.0
    video_start_time = (
        float(video_stream.start_time * video_stream.time_base)
        if video_stream.start_time is not None
        else 0.0
    )
    hands_by_frame: dict[int, list] = {}

    for pkt in container.demux(stream):
        if pkt.size == 0 or pkt.pts is None:
            continue
        raw = bytes(pkt).decode("utf-8").strip()
        if not raw:
            continue
        entry = json.loads(raw)
        pkt_time = float(pkt.pts * stream.time_base)
        frame_idx = int(round((pkt_time - video_start_time) * fps))
        if frame_idx < 0:
            continue
        hands_by_frame[frame_idx] = _extract_hands(entry)
    container.close()
    return hands_by_frame


def infer_actions(hands_by_frame: dict[int, list], n_frames: int) -> list[list[float]]:
    """Run action inference headlessly over all frames."""
    state = ActionState()
    actions: list[list[float]] = []
    latest_hands: list = []
    for i in range(n_frames):
        if i in hands_by_frame:
            latest_hands = hands_by_frame[i]
        hands = latest_hands
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
    """Extract 16-bit depth frames from MKV track 1 (FFV1, gray16le) via ffmpeg.

    Uses ffmpeg subprocess because pyav's FFV1 decoder chokes on keyframe
    parameter changes that ffmpeg handles gracefully.

    Returns a list of uint16 grayscale arrays (H, W).
    """
    probe = av.open(str(mkv_path))
    stream = probe.streams.video[1]
    w, h = stream.width, stream.height
    probe.close()

    proc = subprocess.run(
        [
            "ffmpeg", "-i", str(mkv_path),
            "-map", "0:v:1",
            "-f", "rawvideo", "-pix_fmt", "gray16le",
            "-v", "error",
            "pipe:1",
        ],
        capture_output=True,
    )

    frame_bytes = w * h * 2
    raw = proc.stdout
    n = len(raw) // frame_bytes
    frames = []
    for i in range(n):
        arr = np.frombuffer(raw, dtype=np.uint16, count=w * h, offset=i * frame_bytes)
        frames.append(arr.reshape(h, w).copy())
    return frames


def normalize_depth_to_uint8(
    depth_frames: list[np.ndarray],
    depth_bounds: tuple[float, float] = (DEPTH_MIN_MM, DEPTH_MAX_MM),
) -> list[np.ndarray]:
    """Normalize 16-bit depth frames to 8-bit using fixed physical bounds."""
    out = []
    for d in depth_frames:
        out.append(normalize_depth_frame(d, depth_bounds))
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


def iter_rgb_frames(mkv_path: Path):
    """Yield RGB frames one at a time from MKV track 0 using OpenCV."""
    cap = cv2.VideoCapture(str(mkv_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {mkv_path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def iter_depth_frames(mkv_path: Path):
    """Yield 16-bit depth frames one at a time from MKV track 1 via ffmpeg pipe."""
    probe = av.open(str(mkv_path))
    stream = probe.streams.video[1]
    w, h = stream.width, stream.height
    probe.close()

    frame_bytes = w * h * 2
    proc = subprocess.Popen(
        [
            "ffmpeg", "-i", str(mkv_path),
            "-map", "0:v:1",
            "-f", "rawvideo", "-pix_fmt", "gray16le",
            "-v", "error",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(h, w).copy()
            yield arr
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def normalize_depth_frame(
    d: np.ndarray,
    depth_bounds: tuple[float, float] = (DEPTH_MIN_MM, DEPTH_MAX_MM),
) -> np.ndarray:
    """Normalize one 16-bit depth frame to 8-bit with fixed mm bounds.

    Zero depth is treated as invalid and remains zero in the output.
    Non-zero values are clipped to [depth_min, depth_max] and mapped to [1, 255].
    """
    result = np.zeros(d.shape, dtype=np.uint8)
    mask = d > 0
    if not mask.any():
        return result
    p_lo, p_hi = depth_bounds
    if p_hi <= p_lo:
        return result
    clipped = np.clip(d.astype(np.float32), p_lo, p_hi)
    normed = (clipped - p_lo) / (p_hi - p_lo)
    result[mask] = (normed[mask] * 254 + 1).astype(np.uint8)
    return result


def merge_rgbd_frame(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Merge one RGB (H, W, 3) + depth (H_d, W_d) frame into RGBD (H, W, 4)."""
    h, w = rgb.shape[:2]
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([rgb, depth[:, :, np.newaxis]], axis=2)


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
# Sample frame preview
# ---------------------------------------------------------------------------

def save_sample_frame(
    sample_dir: Path,
    stem: str,
    idx: int,
    rgb_frames: list[np.ndarray],
    depth_frames_u8: list[np.ndarray],
    actions: list[list[float]],
) -> None:
    """Save one random frame at both resolutions for quick visual inspection."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    rgb = rgb_frames[idx]
    depth = depth_frames_u8[idx]

    rgb_lo = cv2.resize(rgb, (LOW_RES[1], LOW_RES[0]), interpolation=cv2.INTER_AREA)
    rgb_hi = cv2.resize(rgb, (FULL_RES[1], FULL_RES[0]), interpolation=cv2.INTER_AREA)
    depth_lo = cv2.resize(depth, (LOW_RES[1], LOW_RES[0]), interpolation=cv2.INTER_AREA)
    depth_hi = cv2.resize(depth, (FULL_RES[1], FULL_RES[0]), interpolation=cv2.INTER_AREA)

    prefix = f"{stem}_f{idx}"
    cv2.imwrite(str(sample_dir / f"{prefix}_rgb_lo.png"), cv2.cvtColor(rgb_lo, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(sample_dir / f"{prefix}_rgb_hi.png"), cv2.cvtColor(rgb_hi, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(sample_dir / f"{prefix}_depth_lo.png"), depth_lo)
    cv2.imwrite(str(sample_dir / f"{prefix}_depth_hi.png"), depth_hi)

    with open(sample_dir / f"{prefix}_action.json", "w") as f:
        json.dump({"frame": idx, "action": actions[idx]}, f, indent=2)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def find_mkvs(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("*.mkv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Root output directory (e.g. src/processed_data_omgrab)")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Root raw-data directory containing train/ and test/ (default: {RAW_DIR})",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Max frames per episode (default: {CHUNK_SIZE})")
    parser.add_argument(
        "--depth-min-mm",
        type=int,
        default=DEPTH_MIN_MM,
        help=f"Minimum valid depth in millimeters for clipping (default: {DEPTH_MIN_MM})",
    )
    parser.add_argument(
        "--depth-max-mm",
        type=int,
        default=DEPTH_MAX_MM,
        help=f"Maximum valid depth in millimeters for clipping (default: {DEPTH_MAX_MM})",
    )
    return parser.parse_args()


def process_split(
    split: str,
    raw_dir: Path,
    out_dir: Path,
    chunk_size: int,
    depth_bounds: tuple[float, float],
) -> None:
    """Process a single split (train or test).

    Streams RGB + depth frames and processes them in chunk-sized batches
    so that only ~chunk_size frames are in memory at a time.
    """
    split_raw = raw_dir / split
    if not split_raw.is_dir():
        print(f"\n  Skipping '{split}': {split_raw} does not exist")
        return

    full_res_dir = out_dir / "full_res" / split
    split_dir = out_dir / "low_res" / split
    info_path = split_dir / "info.pt"

    print(f"\nScanning {split_raw} for MKV files ...")
    mkvs = find_mkvs(split_raw)
    if not mkvs:
        print("  No MKV files found.")
        return
    print(f"  Found {len(mkvs)} file(s)")

    info = load_info(info_path)

    for mkv_path in mkvs:
        stem = mkv_path.stem
        print(f"{'='*60}")
        print(f"Processing: {stem}")

        print(f"  Extracting hand landmarks from {mkv_path.name} (track 3) ...")
        hands_by_frame = extract_hand_landmarks(mkv_path)
        print(f"    {len(hands_by_frame)} hand landmark packets (timestamp-mapped)")

        print(f"  Streaming RGB + depth in chunks of {chunk_size} ...")
        action_state = ActionState()
        chunk_rgbd: list[np.ndarray] = []
        chunk_actions: list[list[float]] = []
        sample_rgb: np.ndarray | None = None
        sample_depth_u8: np.ndarray | None = None
        sample_action: list[float] | None = None
        frame_idx = 0
        chunk_count = 0
        sample_saved = False
        sample_local_idx = random.randint(0, chunk_size - 1)
        latest_hands: list = []

        rgb_gen = iter_rgb_frames(mkv_path)
        depth_gen = iter_depth_frames(mkv_path)
        try:
            for rgb, depth_raw in zip(rgb_gen, depth_gen):
                depth_u8 = normalize_depth_frame(depth_raw, depth_bounds)
                rgbd = merge_rgbd_frame(rgb, depth_u8)

                if frame_idx in hands_by_frame:
                    latest_hands = hands_by_frame[frame_idx]
                hands = latest_hands
                right = get_right_hand(hands)
                update_actions(action_state, right["landmarks"] if right else None)
                action = encode_action(action_state)

                chunk_rgbd.append(rgbd)
                chunk_actions.append(action)

                local_idx = len(chunk_rgbd) - 1
                if not sample_saved and local_idx == sample_local_idx:
                    sample_rgb = rgb
                    sample_depth_u8 = depth_u8
                    sample_action = action

                frame_idx += 1

                if len(chunk_rgbd) == chunk_size:
                    if not sample_saved and sample_rgb is not None:
                        sample_dir = out_dir / "samples"
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        global_idx = chunk_count * chunk_size + sample_local_idx
                        prefix = f"{stem}_f{global_idx}"
                        rgb_lo = cv2.resize(sample_rgb, (LOW_RES[1], LOW_RES[0]), interpolation=cv2.INTER_AREA)
                        rgb_hi = cv2.resize(sample_rgb, (FULL_RES[1], FULL_RES[0]), interpolation=cv2.INTER_AREA)
                        d_lo = cv2.resize(sample_depth_u8, (LOW_RES[1], LOW_RES[0]), interpolation=cv2.INTER_AREA)
                        d_hi = cv2.resize(sample_depth_u8, (FULL_RES[1], FULL_RES[0]), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(str(sample_dir / f"{prefix}_rgb_lo.png"), cv2.cvtColor(rgb_lo, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(sample_dir / f"{prefix}_rgb_hi.png"), cv2.cvtColor(rgb_hi, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(sample_dir / f"{prefix}_depth_lo.png"), d_lo)
                        cv2.imwrite(str(sample_dir / f"{prefix}_depth_hi.png"), d_hi)
                        with open(sample_dir / f"{prefix}_action.json", "w") as f:
                            json.dump({"frame": global_idx, "action": sample_action}, f, indent=2)
                        print(f"  Saved sample frame {global_idx} to {sample_dir}")
                        sample_saved = True
                        del sample_rgb, sample_depth_u8, sample_action

                    episode_id = info["num_episodes"]
                    hdf5_name = f"{stem}_chunk{chunk_count}.hdf5"
                    hdf5_path = full_res_dir / hdf5_name
                    file_id = f"{split}/{hdf5_name}"
                    save_full_res_hdf5(hdf5_path, chunk_rgbd, chunk_actions, 0, chunk_size)
                    save_low_res_episode(
                        split_dir, episode_id, chunk_rgbd, chunk_actions,
                        0, chunk_size, file_id,
                    )

                    info["start_idx"] = np.concatenate((info["start_idx"], np.array([info["num_steps"]])))
                    info["lengths"] = np.concatenate((info["lengths"], np.array([chunk_size])))
                    info["num_steps"] += chunk_size
                    info["num_episodes"] += 1
                    info["counter_rew"].update(Counter({0.0: chunk_size}))
                    info["counter_end"].update(Counter({0: chunk_size}))

                    ep_path = get_episode_path(split_dir, episode_id)
                    print(f"    chunk {chunk_count}: frames "
                          f"[{chunk_count * chunk_size}:{(chunk_count + 1) * chunk_size}] "
                          f"-> episode {episode_id} ({ep_path.name})")

                    chunk_rgbd.clear()
                    chunk_actions.clear()
                    chunk_count += 1
                    gc.collect()
        finally:
            rgb_gen.close()
            depth_gen.close()

        tail = len(chunk_rgbd)
        total_frames = frame_idx
        if chunk_count == 0:
            print(f"  SKIP: clip too short ({total_frames}f < {chunk_size})")
        elif tail > 0:
            print(f"  {chunk_count} full chunk(s), discarding {tail} leftover frames")
        else:
            print(f"  {chunk_count} full chunk(s)")
        print(f"  {total_frames} frames streamed")

        del hands_by_frame
        chunk_rgbd.clear()
        chunk_actions.clear()
        gc.collect()

    save_info(info_path, info)

    print(f"\n  {split} done: {info['num_episodes']} episodes, {info['num_steps']} steps")


def main() -> None:
    args = parse_args()
    if args.depth_max_mm <= args.depth_min_mm:
        raise ValueError("--depth-max-mm must be greater than --depth-min-mm")
    depth_bounds = (float(args.depth_min_mm), float(args.depth_max_mm))

    if args.out_dir.exists():
        print(f"Clearing existing {args.out_dir} ...")
        shutil.rmtree(args.out_dir)

    print(f"Using raw data root: {args.raw_dir}")
    print(f"Depth normalization: clip [{args.depth_min_mm}, {args.depth_max_mm}] mm -> uint8")

    for split in ("train", "test"):
        print(f"\n{'#'*60}")
        print(f"# Split: {split}")
        print(f"{'#'*60}")
        process_split(split, args.raw_dir, args.out_dir, args.chunk_size, depth_bounds)

    print(f"\n{'='*60}")
    print(f"Dataset written to {args.out_dir}")
    for split in ("train", "test"):
        full_res_split = args.out_dir / "full_res" / split
        split_dir = args.out_dir / "low_res" / split
        info_path = split_dir / "info.pt"
        if info_path.exists():
            info = torch.load(info_path, weights_only=False)
            print(f"  {split}: {info['num_episodes']} episodes, {info['num_steps']} steps")
            print(f"    full_res: {full_res_split}")
            print(f"    low_res:  {split_dir}")
        else:
            print(f"  {split}: (empty)")


if __name__ == "__main__":
    main()
