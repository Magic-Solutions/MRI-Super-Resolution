#!/usr/bin/env python3
"""Render depth-channel histogram from processed HDF5 training data.

This inspects the depth channel exactly as stored in full_res HDF5 files
and reports the distribution both in:
  1) stored uint8 space [0, 255], and
  2) model input space after Dataset transform: x/255*2-1 in [-1, 1].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def select_evenly_spaced_files(files: list[Path], count: int) -> list[Path]:
    if count <= 0:
        return []
    if len(files) <= count:
        return files
    idxs = np.linspace(0, len(files) - 1, num=count, dtype=int).tolist()
    return [files[i] for i in idxs]


def draw_hist_panel(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    counts: np.ndarray,
    title: str,
    x_labels: tuple[str, str, str],
) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (x0 + 10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)

    chart_x = x0 + 14
    chart_y = y0 + 44
    chart_w = w - 28
    chart_h = h - 84
    cv2.rectangle(canvas, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (60, 60, 60), 1, cv2.LINE_AA)

    y_max = int(counts.max()) if counts.size > 0 else 0
    if y_max <= 0:
        cv2.putText(
            canvas,
            "No depth samples",
            (chart_x + 12, chart_y + chart_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    else:
        n = counts.shape[0]
        for i in range(n):
            x = chart_x + int(i * chart_w / n)
            bar_h = int((counts[i] / y_max) * (chart_h - 2))
            y = chart_y + chart_h - bar_h
            color = (80, 120, 240) if i > 0 else (70, 70, 200)
            cv2.line(canvas, (x, chart_y + chart_h), (x, y), color, 1, cv2.LINE_AA)

    cv2.putText(canvas, x_labels[0], (chart_x, chart_y + chart_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    mid_x = chart_x + chart_w // 2 - 10
    cv2.putText(canvas, x_labels[1], (mid_x, chart_y + chart_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    right_x = chart_x + chart_w - 24
    cv2.putText(canvas, x_labels[2], (right_x, chart_y + chart_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing full_res HDF5 chunks")
    parser.add_argument("--num-chunks", type=int, default=5, help="Use this many evenly spaced chunks")
    parser.add_argument("--depth-min-mm", type=int, default=200)
    parser.add_argument("--depth-max-mm", type=int, default=3000)
    parser.add_argument("--out", required=True, type=Path, help="Output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.data_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {args.data_dir}")

    selected = select_evenly_spaced_files(files, args.num_chunks)

    hist_u8 = np.zeros(256, dtype=np.int64)
    used_frames = 0
    skipped_non_depth = 0

    for path in selected:
        with h5py.File(path, "r") as f:
            frame_keys = sorted([k for k in f.keys() if k.endswith("_x")], key=lambda k: int(k.split("_")[1]))
            for key in frame_keys:
                frame = np.asarray(f[key])
                if frame.ndim != 3 or frame.shape[2] < 4:
                    skipped_non_depth += 1
                    continue
                depth_u8 = frame[:, :, 3]
                hist_u8 += np.bincount(depth_u8.reshape(-1), minlength=256)
                used_frames += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    width = 1400
    height = 760
    panel_w = (width - 60) // 2
    panel_h = 460
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    draw_hist_panel(
        canvas,
        20,
        90,
        panel_w,
        panel_h,
        hist_u8,
        "Stored depth channel (uint8 in HDF5)",
        ("0", "128", "255"),
    )

    # depth_model = depth_u8 / 255 * 2 - 1, so bins map 1:1 with uint8 bins.
    hist_model = hist_u8.copy()
    draw_hist_panel(
        canvas,
        40 + panel_w,
        90,
        panel_w,
        panel_h,
        hist_model,
        "Model input depth channel (float after x/255*2-1)",
        ("-1.00", "0.00", "1.00"),
    )

    total_px = int(hist_u8.sum())
    invalid_px = int(hist_u8[0])
    invalid_pct = (100.0 * invalid_px / total_px) if total_px > 0 else 0.0
    used_chunks = len(selected)
    subtitle = (
        f"Chunks used: {used_chunks} | Frames used: {used_frames} | "
        f"Depth mm clip at preprocess: [{args.depth_min_mm}, {args.depth_max_mm}] | "
        f"Zero-depth (invalid): {invalid_px:,}/{total_px:,} ({invalid_pct:.2f}%)"
    )
    cv2.putText(canvas, "Depth Preview Histogram", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (185, 185, 185), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Note: non-zero depth is clipped to [min_mm, max_mm], encoded to uint8 [1,255], then normalized to [-1,1] in the dataset loader.",
        (20, 590),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    if skipped_non_depth > 0:
        cv2.putText(
            canvas,
            f"Skipped {skipped_non_depth} frame tensors without depth channel.",
            (20, 620),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (120, 180, 255),
            1,
            cv2.LINE_AA,
        )

    ok = cv2.imwrite(str(args.out), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write histogram PNG: {args.out}")
    print(f"Saved depth histogram: {args.out}")


if __name__ == "__main__":
    main()
