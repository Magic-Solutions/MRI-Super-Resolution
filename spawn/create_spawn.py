"""Create spawn data from HDF5 training episodes for interactive inference.

Usage:
    python spawn/create_spawn.py --hdf5-dir src/processed_data_omgrab/full_res/omgrab --num-conditioning 4

This generates one spawn directory per HDF5 file, each containing:
    low_res.npy   (num_conditioning, 3, 30, 56) uint8
    full_res.npy  (num_conditioning, 3, 150, 280) uint8
    act.npy       (num_conditioning, 51) float
    next_act.npy  (1, 51) float
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image


LOW_RES_SIZE = (56, 30)  # (width, height) for PIL


def create_spawn_from_hdf5(
    hdf5_path: Path,
    output_dir: Path,
    num_conditioning: int,
    start_frame: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        full_res_frames = []
        low_res_frames = []
        acts = []

        for i in range(start_frame, start_frame + num_conditioning):
            frame = torch.tensor(f[f"frame_{i}_x"][:]).flip(2).permute(2, 0, 1)
            full_res_frames.append(frame)

            img = Image.fromarray(frame.permute(1, 2, 0).numpy())
            low = torch.tensor(np.array(img.resize(LOW_RES_SIZE, Image.BICUBIC))).permute(2, 0, 1)
            low_res_frames.append(low)

            acts.append(torch.tensor(f[f"frame_{i}_y"][:]))

        next_act = torch.tensor(f[f"frame_{start_frame + num_conditioning}_y"][:]).unsqueeze(0)

    np.save(output_dir / "full_res.npy", torch.stack(full_res_frames).numpy())
    np.save(output_dir / "low_res.npy", torch.stack(low_res_frames).numpy())
    np.save(output_dir / "act.npy", torch.stack(acts).numpy())
    np.save(output_dir / "next_act.npy", next_act.numpy())


def main():
    parser = argparse.ArgumentParser(description="Create spawn data from HDF5 episodes.")
    parser.add_argument("--hdf5-dir", type=str, required=True, help="Directory containing .hdf5 files.")
    parser.add_argument("--output-dir", type=str, default="spawn", help="Root output directory.")
    parser.add_argument("--num-conditioning", type=int, default=4, help="Number of conditioning frames (must match agent config).")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index within each episode.")
    args = parser.parse_args()

    hdf5_dir = Path(args.hdf5_dir)
    output_root = Path(args.output_dir)
    hdf5_files = sorted(hdf5_dir.glob("*.hdf5"))

    if not hdf5_files:
        print(f"No .hdf5 files found in {hdf5_dir}")
        return

    for idx, hdf5_path in enumerate(hdf5_files):
        spawn_dir = output_root / f"{idx:03d}"
        create_spawn_from_hdf5(hdf5_path, spawn_dir, args.num_conditioning, args.start_frame)
        print(f"[{idx}] {hdf5_path.name} -> {spawn_dir}")

        with h5py.File(hdf5_path, "r") as f:
            fr = np.load(spawn_dir / "full_res.npy")
            lr = np.load(spawn_dir / "low_res.npy")
            print(f"    full_res: {fr.shape} {fr.dtype}, low_res: {lr.shape} {lr.dtype}")

    print(f"\nCreated {len(hdf5_files)} spawn(s) in {output_root}/")


if __name__ == "__main__":
    main()
