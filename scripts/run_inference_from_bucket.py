"""Pull a training run from GCS and launch interactive inference.

Downloads the trainer config, checkpoint, and one HDF5 episode (for spawn
generation), then launches the interactive game.  No local data needed.

doe
GCS layout expected:
    gs://BUCKET/diamond/runs/{run_id}/
        training_run/hydra/{run_id}/
            config/..._config_trainer.yaml
            checkpoints/agent_versions/agent_epoch_XXXXX.pt
        processed_data/full_res/train/*.hdf5
"""

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUCKET = "gs://omgrab-dev-diamond-vertex-artifacts"

LOW_RES_SIZE = (56, 30)  # (width, height) for PIL
NUM_CONDITIONING = 4
NUM_SPAWNS = 3


def gcs_cp(src: str, dst: str) -> None:
    """Copy a file from GCS using whichever CLI is available."""
    for cmd in (["gcloud", "storage", "cp"], ["gsutil", "cp"]):
        if shutil.which(cmd[0]):
            result = subprocess.run(cmd + [src, dst], capture_output=True, text=True)
            if result.returncode == 0:
                return
            print(f"  ERROR ({cmd[0]}): {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
    print("ERROR: neither 'gcloud' nor 'gsutil' found. Install the Google Cloud SDK.", file=sys.stderr)
    sys.exit(1)


def gcs_ls(prefix: str) -> list[str]:
    """List objects under a GCS prefix."""
    for cmd in (["gcloud", "storage", "ls"], ["gsutil", "ls"]):
        if shutil.which(cmd[0]):
            result = subprocess.run(cmd + [prefix], capture_output=True, text=True)
            if result.returncode == 0:
                return [line.strip() for line in result.stdout.splitlines() if line.strip()]
            print(f"  ERROR ({cmd[0]}): {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
    print("ERROR: neither 'gcloud' nor 'gsutil' found.", file=sys.stderr)
    sys.exit(1)


def find_latest_checkpoint(gcs_ckpt_dir: str, epoch: int | None) -> str:
    """Find the checkpoint to download. If epoch given, use that; otherwise pick latest."""
    objects = gcs_ls(gcs_ckpt_dir)
    pts = [o for o in objects if o.endswith(".pt")]
    if not pts:
        print(f"ERROR: no checkpoints found under {gcs_ckpt_dir}")
        sys.exit(1)

    if epoch is not None:
        target = f"agent_epoch_{epoch:05d}.pt"
        matches = [p for p in pts if target in p]
        if not matches:
            available = [Path(p).name for p in pts]
            print(f"ERROR: epoch {epoch} not found. Available: {available}")
            sys.exit(1)
        return matches[0]

    def epoch_num(p: str) -> int:
        m = re.search(r"epoch_(\d+)", p)
        return int(m.group(1)) if m else -1

    return max(pts, key=epoch_num)


def parse_use_depth(config_path: Path) -> bool:
    """Quick check for use_depth in the downloaded trainer.yaml."""
    text = config_path.read_text()
    if "use_depth: true" in text:
        return True
    return False


def generate_spawn_from_hdf5(hdf5_path: Path, spawn_dir: Path, start_frame: int = 0) -> None:
    """Generate one spawn directory from an HDF5 episode."""
    spawn_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        full_res_frames, low_res_frames, acts = [], [], []

        for i in range(start_frame, start_frame + NUM_CONDITIONING):
            frame = torch.tensor(f[f"frame_{i}_x"][:]).permute(2, 0, 1)
            full_res_frames.append(frame)

            img = Image.fromarray(frame.permute(1, 2, 0).numpy())
            low = torch.tensor(
                np.array(img.resize(LOW_RES_SIZE, Image.Resampling.BICUBIC)),
            ).permute(2, 0, 1)
            low_res_frames.append(low)

            acts.append(torch.tensor(f[f"frame_{i}_y"][:]))

        next_act = torch.tensor(
            f[f"frame_{start_frame + NUM_CONDITIONING}_y"][:],
        ).unsqueeze(0)

    np.save(spawn_dir / "full_res.npy", torch.stack(full_res_frames).numpy())
    np.save(spawn_dir / "low_res.npy", torch.stack(low_res_frames).numpy())
    np.save(spawn_dir / "act.npy", torch.stack(acts).numpy())
    np.save(spawn_dir / "next_act.npy", next_act.numpy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_id", help="Training run ID (e.g. run-20260224-045328)")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"GCS bucket (default: {DEFAULT_BUCKET})")
    parser.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch (default: latest)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--size-multiplier", type=int, default=2)
    parser.add_argument("--world-model-env", type=str, default=None, help="Inference config (default: from trainer.yaml, usually 'fast'). Options: fast, higher_quality.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id

    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    gcs_base = f"{args.bucket}/diamond/runs/{run_id}/training_run/hydra/{run_id}"

    # 1. Download trainer.yaml
    config_path = run_dir / "trainer.yaml"
    if config_path.exists():
        print(f"[1/4] Config already downloaded: {config_path}")
    else:
        print(f"[1/4] Downloading config ...")
        config_objects = gcs_ls(f"{gcs_base}/config/")
        yaml_files = [o for o in config_objects if o.endswith("trainer.yaml")]
        if not yaml_files:
            print(f"  ERROR: no trainer.yaml found under {gcs_base}/config/")
            sys.exit(1)
        gcs_cp(yaml_files[0].strip(), str(config_path))
        print(f"       -> {config_path}")

    use_depth = parse_use_depth(config_path)
    print(f"       use_depth: {use_depth}")

    # 2. Download checkpoint
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    existing_pts = sorted(ckpt_dir.glob("*.pt"))

    if existing_pts and args.epoch is None:
        ckpt_path = existing_pts[-1]
        print(f"[2/4] Checkpoint already downloaded: {ckpt_path.name}")
    else:
        print(f"[2/4] Finding checkpoint (epoch={args.epoch or 'latest'}) ...")
        gcs_ckpt_prefix = f"{gcs_base}/checkpoints/agent_versions/"
        gcs_ckpt = find_latest_checkpoint(gcs_ckpt_prefix, args.epoch)
        ckpt_name = Path(gcs_ckpt).name
        ckpt_path = ckpt_dir / ckpt_name
        if ckpt_path.exists():
            print(f"       Already exists: {ckpt_path}")
        else:
            print(f"       Downloading {ckpt_name} ...")
            gcs_cp(gcs_ckpt.strip(), str(ckpt_path))
            print(f"       -> {ckpt_path}")

    # 3. Generate spawn data (download one HDF5 from the run's processed data)
    spawn_dir = run_dir / "spawn"
    existing_spawns = list(spawn_dir.iterdir()) if spawn_dir.exists() else []

    if existing_spawns:
        print(f"[3/4] Spawn data exists: {len(existing_spawns)} spawn(s)")
    else:
        gcs_hdf5_dir = f"{args.bucket}/diamond/runs/{run_id}/processed_data/full_res/train/"
        print(f"[3/4] Downloading HDF5 for spawn generation ...")
        hdf5_objects = gcs_ls(gcs_hdf5_dir)
        hdf5_files = [o for o in hdf5_objects if o.endswith(".hdf5")]
        if not hdf5_files:
            print(f"       ERROR: no HDF5 files found under {gcs_hdf5_dir}")
            sys.exit(1)

        hdf5_name = Path(hdf5_files[0]).name
        local_hdf5 = run_dir / hdf5_name
        if not local_hdf5.exists():
            print(f"       Downloading {hdf5_name} ...")
            gcs_cp(hdf5_files[0].strip(), str(local_hdf5))

        with h5py.File(local_hdf5, "r") as f:
            n_frames = sum(1 for k in f.keys() if k.endswith("_x"))
        max_start = n_frames - NUM_CONDITIONING - 1
        if max_start < 0:
            print(f"       ERROR: HDF5 has only {n_frames} frames, need at least {NUM_CONDITIONING + 1}")
            sys.exit(1)

        starts = sorted(random.sample(range(max_start + 1), min(NUM_SPAWNS, max_start + 1)))
        print(f"       Generating {len(starts)} spawns from {hdf5_name} (frames: {starts}) ...")
        for idx, start in enumerate(starts):
            generate_spawn_from_hdf5(local_hdf5, spawn_dir / f"{idx:03d}", start_frame=start)
            fr = np.load(spawn_dir / f"{idx:03d}" / "full_res.npy")
            lr = np.load(spawn_dir / f"{idx:03d}" / "low_res.npy")
            print(f"       [{idx}] start={start:>5d}  full_res: {fr.shape}  low_res: {lr.shape}")

        local_hdf5.unlink()

    # 4. Launch game
    print(f"\n[4/4] Launching inference ...")
    print(f"       Checkpoint: {ckpt_path.name}")
    print(f"       Spawn dir:  {spawn_dir}")
    print(f"       Depth mode: {use_depth}")
    print(f"       WM env:     {args.world_model_env or 'default (from trainer.yaml)'}")
    print()

    play_cmd = [
        sys.executable, str(REPO_ROOT / "src" / "play.py"),
        "--local-ckpt", str(ckpt_path),
        "--spawn-dir", str(spawn_dir),
        "--fps", str(args.fps),
        "--size-multiplier", str(args.size_multiplier),
    ]
    if use_depth:
        play_cmd.append("--use-depth")
    if args.world_model_env:
        play_cmd.extend(["--world-model-env", args.world_model_env])

    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    os.execve(
        sys.executable,
        play_cmd,
        env,
    )


if __name__ == "__main__":
    main()
