#!/usr/bin/env python3
"""Preview inferred actions locally using the exact preprocessing logic.

This script runs a local MKV through the same hand/action inference path as
`src/raw_data/scripts/main.py` and overlays the resulting controls on RGB frames.
It is intended for fast tuning/debugging without GCS download or dataset writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.raw_data.scripts import main as pipeline


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mkv", type=Path, required=True, help="Path to local MKV recording")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write actions ndjson")
    parser.add_argument("--wait-ms", type=int, default=40, help="OpenCV wait delay (ms), default 40")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit for quick checks")
    return parser.parse_args()


def draw_landmarks(frame: np.ndarray, hand: dict, w: int, h: int) -> None:
    lm = hand["landmarks"]
    pts = [(int(l[0] * w), int(l[1] * h)) for l in lm]
    color = (0, 128, 255)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)

    tx, ty = pipeline.get_tracking_point(lm)
    cv2.circle(frame, (int(tx * w), int(ty * h)), 8, (0, 255, 255), 2, cv2.LINE_AA)


def draw_key(frame: np.ndarray, cx: int, cy: int, w: int, h: int, active: bool, label: str) -> None:
    color_bg = (0, 200, 80) if active else (60, 60, 60)
    color_border = (100, 255, 130) if active else (140, 140, 140)
    color_text = (255, 255, 255) if active else (180, 180, 180)

    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bg, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_border, 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5 if len(label) <= 2 else 0.4
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    cv2.putText(frame, label, (cx - tw // 2, cy + th // 2), font, scale, color_text, 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, state: pipeline.ActionState) -> None:
    fh, fw = frame.shape[:2]
    ks = 44
    pad = 6
    margin = 20

    cluster_cx = fw - margin - ks - pad - ks // 2
    cluster_cy = fh - margin - ks - pad - ks // 2

    draw_key(frame, cluster_cx, cluster_cy - ks - pad, ks, ks, state.up, "^")
    draw_key(frame, cluster_cx - ks - pad, cluster_cy, ks, ks, state.left, "<")
    draw_key(frame, cluster_cx, cluster_cy, ks, ks, state.down, "v")
    draw_key(frame, cluster_cx + ks + pad, cluster_cy, ks, ks, state.right, ">")

    space_w = 3 * ks + 2 * pad
    space_cy = cluster_cy + ks + pad
    draw_key(frame, cluster_cx, space_cy, space_w, ks - 8, state.grip, "SPACE")


def draw_hud_from_action(frame: np.ndarray, action: list[float]) -> None:
    """Render HUD from a precomputed action vector."""
    class HudState:
        pass

    s = HudState()
    s.up = bool(action[0] > 0.5)
    s.left = bool(action[1] > 0.5)
    s.down = bool(action[2] > 0.5)
    s.right = bool(action[3] > 0.5)
    s.grip = bool(action[4] > 0.5)
    s.prev_x = None
    s.prev_y = None
    draw_hud(frame, s)


def main() -> None:
    args = parse_args()
    if not args.mkv.is_file():
        raise FileNotFoundError(f"MKV not found: {args.mkv}")

    hands_by_frame = pipeline.extract_hand_landmarks(args.mkv)
    cap = cv2.VideoCapture(str(args.mkv))
    if not cap.isOpened():
        print(f"Cannot open {args.mkv}")
        sys.exit(1)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        # Fallback for containers where frame count metadata is missing.
        n_frames = max(hands_by_frame.keys(), default=-1) + 1
    precomputed_actions = pipeline.infer_actions(hands_by_frame, n_frames)

    action_records: list[dict] = []
    paused = False
    frame_idx = 0
    latest_hands: list = []

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            h, w = frame.shape[:2]
            if frame_idx in hands_by_frame:
                latest_hands = hands_by_frame[frame_idx]
            hands = latest_hands
            right = pipeline.get_right_hand(hands)
            if right:
                draw_landmarks(frame, right, w, h)

            if frame_idx < len(precomputed_actions):
                action = precomputed_actions[frame_idx]
            else:
                action = [0.0] * pipeline.ACTION_DIM
            action_records.append({"f": frame_idx, "action": action})

            draw_hud_from_action(frame, action)
            cv2.putText(
                frame,
                f"Frame {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            display = frame
            frame_idx += 1

        cv2.imshow("Local Action Preview (pipeline logic)", display)
        key = cv2.waitKey(args.wait_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    if args.out is not None:
        with open(args.out, "w", encoding="utf-8") as f:
            for rec in action_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(action_records)} frames to {args.out}")
    else:
        print(f"Processed {len(action_records)} frames (no output file requested)")


if __name__ == "__main__":
    main()

