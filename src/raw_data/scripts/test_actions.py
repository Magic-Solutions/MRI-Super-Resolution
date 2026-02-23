import argparse
import json
import sys

import cv2
import numpy as np

from src.raw_data.scripts import main as pipeline

DEFAULT_MKV_PATH = "421c63b0-1036-4300-b0a3-0b0d81f8c68b.mkv"
DEFAULT_OUT_PATH = "421c63b0-1036-4300-b0a3-0b0d81f8c68b_actions.ndjson"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local full-recording action visualizer. Uses the exact hand/action "
            "processing code path from src/raw_data/scripts/main.py, without download or dataset writing."
        )
    )
    parser.add_argument("--mkv", default=DEFAULT_MKV_PATH, help="Path to local MKV recording")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output NDJSON actions path")
    parser.add_argument("--no-write", action="store_true", help="Do not write output actions file")
    parser.add_argument("--wait-ms", type=int, default=40, help="OpenCV waitKey delay (default 40ms)")
    return parser.parse_args()


def draw_landmarks(frame: np.ndarray, hand: dict, w: int, h: int) -> None:
    lm = hand["landmarks"]
    pts = [(int(l[0] * w), int(l[1] * h)) for l in lm]
    color = (0, 128, 255)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)

    # Highlight tracking point used by the pipeline
    tx, ty = pipeline.get_tracking_point(lm)
    cv2.circle(frame, (int(tx * w), int(ty * h)), 8, (0, 255, 255), 2, cv2.LINE_AA)


def draw_key(
    frame: np.ndarray, cx: int, cy: int, w: int, h: int, active: bool, label: str
) -> None:
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
    cv2.putText(
        frame, label,
        (cx - tw // 2, cy + th // 2),
        font, scale, color_text, 1, cv2.LINE_AA,
    )


def draw_hud(frame: np.ndarray, state: pipeline.ActionState) -> None:
    fh, fw = frame.shape[:2]
    ks = 44
    pad = 6
    margin = 20

    # Center of the arrow key cluster
    cluster_cx = fw - margin - ks - pad - ks // 2
    cluster_cy = fh - margin - ks - pad - ks // 2

    # Arrow keys:  [up] on top, [left][down][right] on bottom row
    draw_key(frame, cluster_cx, cluster_cy - ks - pad, ks, ks, state.up, "^")
    draw_key(frame, cluster_cx - ks - pad, cluster_cy, ks, ks, state.left, "<")
    draw_key(frame, cluster_cx, cluster_cy, ks, ks, state.down, "v")
    draw_key(frame, cluster_cx + ks + pad, cluster_cy, ks, ks, state.right, ">")

    # Space bar below arrows
    space_w = 3 * ks + 2 * pad
    space_cy = cluster_cy + ks + pad
    draw_key(frame, cluster_cx, space_cy, space_w, ks - 8, state.grip, "SPACE")

    # Tracking dot indicator
    if state.prev_x is not None:
        ix = int(state.prev_x * fw)
        iy = int(state.prev_y * fh)
        size = 10
        cv2.line(frame, (ix - size, iy), (ix + size, iy), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (ix, iy - size), (ix, iy + size), (0, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    hands_by_frame = pipeline.extract_hand_landmarks(args.mkv)
    cap = cv2.VideoCapture(args.mkv)
    if not cap.isOpened():
        print(f"Cannot open {args.mkv}")
        sys.exit(1)

    state = pipeline.ActionState()
    action_records: list[dict] = []
    paused = False
    frame_idx = 0
    latest_hands: list = []

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if frame_idx in hands_by_frame:
                latest_hands = hands_by_frame[frame_idx]
            hands = latest_hands
            right = pipeline.get_right_hand(hands)

            if right:
                draw_landmarks(frame, right, w, h)
                pipeline.update_actions(state, right["landmarks"])
            else:
                pipeline.update_actions(state, None)

            action_records.append({
                "f": frame_idx,
                "action": pipeline.encode_action(state),
            })

            draw_hud(frame, state)

            cv2.putText(
                frame, f"Frame {frame_idx}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA,
            )

            display = frame
            frame_idx += 1

        cv2.imshow("Actions from Hand Tracking", display)

        key = cv2.waitKey(args.wait_ms) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    if not args.no_write:
        with open(args.out, "w", encoding="utf-8") as f:
            for rec in action_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(action_records)} frames to {args.out}")


if __name__ == "__main__":
    main()
