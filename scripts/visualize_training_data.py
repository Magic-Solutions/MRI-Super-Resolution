"""Visualize training data: RGB + depth side-by-side with action HUD overlay.

Reads full_res HDF5 files and renders a video showing:
  - Left: RGB frame with arrow keys + spacebar overlay
  - Right: Depth channel (inferno colormap)

Hand landmarks are NOT stored in HDF5 (only used during preprocessing),
so they cannot be overlaid here.

Usage:
    python scripts/visualize_training_data.py [--num-seconds 10] [--fps 15] [--out training_viz.mp4]
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

DATA_DIR = Path("src/processed_data_omgrab/full_res/train")

# Action vector layout (51 dims)
# [0:4]   directions: up=0, left=1, down=2, right=3  (arrow keys)
# [4]     space (jump/grip)
# [5:11]  unused in our setup (ctrl, shift, 1, 2, 3, r)
# [11:13] mouse clicks: left=11, right=12
# [13:36] mouse_x one-hot (23 bins), neutral at index 24
# [36:51] mouse_y one-hot (15 bins), neutral at index 43


def decode_action(act: np.ndarray) -> dict:
    return {
        "up": bool(act[0] > 0.5),
        "left": bool(act[1] > 0.5),
        "down": bool(act[2] > 0.5),
        "right": bool(act[3] > 0.5),
        "space": bool(act[4] > 0.5),
    }


def draw_key(frame: np.ndarray, cx: int, cy: int, w: int, h: int, active: bool, label: str) -> None:
    color_bg = (0, 200, 80) if active else (60, 60, 60)
    color_border = (100, 255, 130) if active else (140, 140, 140)
    color_text = (255, 255, 255) if active else (180, 180, 180)

    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bg, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_border, 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5 if len(label) <= 2 else 0.35
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    cv2.putText(frame, label, (cx - tw // 2, cy + th // 2), font, scale, color_text, 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, action: dict, depth_frozen: bool) -> None:
    fh, fw = frame.shape[:2]
    ks = 36  # key size
    pad = 4
    margin = 12

    # Arrow key cluster (bottom-right)
    cx = fw - margin - ks - pad - ks // 2
    cy = fh - margin - ks - pad - ks // 2

    # Arrow keys: [^] on top, [<][v][>] on bottom row
    draw_key(frame, cx, cy - ks - pad, ks, ks, action["up"], "^")
    draw_key(frame, cx - ks - pad, cy, ks, ks, action["left"], "<")
    draw_key(frame, cx, cy, ks, ks, action["down"], "v")
    draw_key(frame, cx + ks + pad, cy, ks, ks, action["right"], ">")

    # Space bar below arrows
    space_w = 3 * ks + 2 * pad
    space_cy = cy + ks + pad
    draw_key(frame, cx, space_cy, space_w, ks - 8, action["space"], "SPACE")

    # Depth frozen warning
    if depth_frozen:
        cv2.putText(frame, "DEPTH FROZEN", (8, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seconds", type=int, default=10)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--out", type=str, default="training_viz.mp4")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--scale", type=int, default=3, help="Upscale factor for display")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    if not hdf5_files:
        print(f"No HDF5 files in {data_dir}")
        return

    total_frames = args.num_seconds * args.fps
    print(f"Rendering {total_frames} frames ({args.num_seconds}s @ {args.fps}fps)")

    orig_h, orig_w = 150, 280
    scale = args.scale
    disp_h, disp_w = orig_h * scale, orig_w * scale
    gap = 4
    canvas_w = disp_w * 2 + gap
    canvas_h = disp_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (canvas_w, canvas_h))

    frame_count = 0
    for hdf5_path in hdf5_files:
        if frame_count >= total_frames:
            break

        f = h5py.File(hdf5_path, "r")
        n_frames = len([k for k in f.keys() if k.endswith("_x")])

        prev_depth = None
        depth_frozen = False

        for i in range(n_frames):
            if frame_count >= total_frames:
                break

            rgbd = f[f"frame_{i}_x"][:]  # (150, 280, 4) uint8
            act = f[f"frame_{i}_y"][:]   # (51,) float64

            rgb = rgbd[:, :, :3]
            depth = rgbd[:, :, 3]

            # Detect depth freeze
            if prev_depth is not None and np.array_equal(depth, prev_depth):
                depth_frozen = True
            else:
                depth_frozen = False
            prev_depth = depth.copy()

            # Upscale
            rgb_up = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            depth_up = cv2.resize(depth, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            depth_color = cv2.applyColorMap(depth_up, cv2.COLORMAP_INFERNO)

            # Convert RGB to BGR for OpenCV
            rgb_bgr = cv2.cvtColor(rgb_up, cv2.COLOR_RGB2BGR)

            # Draw action HUD
            action = decode_action(act)
            draw_hud(rgb_bgr, action, depth_frozen)

            # Frame counter
            cv2.putText(rgb_bgr, f"F:{frame_count}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Labels
            cv2.putText(rgb_bgr, "RGB", (disp_w - 50, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            label_color = (0, 0, 255) if depth_frozen else (255, 255, 255)
            cv2.putText(depth_color, "DEPTH", (disp_w - 80, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)

            # Compose canvas
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            canvas[:, :disp_w] = rgb_bgr
            canvas[:, disp_w + gap:] = depth_color

            writer.write(canvas)
            frame_count += 1

        f.close()
        print(f"  {hdf5_path.name}: {n_frames} frames, depth froze={depth_frozen} (total: {frame_count})")

    writer.release()
    print(f"Saved {frame_count} frames to {args.out}")


if __name__ == "__main__":
    main()
