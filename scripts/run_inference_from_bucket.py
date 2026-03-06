#!/usr/bin/env python3
"""Download a training run from GCS and run MRI super-resolution inference.

Downloads the checkpoint from a Vertex training run, then runs the 2.5D
upsampler on local test data and saves comparison images + metrics.

GCS layout expected:
    gs://BUCKET/diamond/runs/{run_id}/
        training_run/hydra/{run_id}/
            checkpoints/agent_versions/agent_epoch_XXXXX.pt

Usage:
    python scripts/run_inference_from_bucket.py run-20260306-035738
    python scripts/run_inference_from_bucket.py run-20260306-035738 --epoch 5 --num-samples 10
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUCKET = "gs://omgrab-dev-diamond-vertex-artifacts"


def gcs_cp(src: str, dst: str) -> None:
    for cmd in (["gcloud", "storage", "cp"], ["gsutil", "cp"]):
        if shutil.which(cmd[0]):
            result = subprocess.run(cmd + [src, dst], capture_output=True, text=True)
            if result.returncode == 0:
                return
            print(f"  ERROR ({cmd[0]}): {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
    print("ERROR: neither 'gcloud' nor 'gsutil' found.", file=sys.stderr)
    sys.exit(1)


def gcs_ls(prefix: str) -> list[str]:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_id", help="Training run ID (e.g. run-20260306-035738)")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch (default: latest)")
    p.add_argument("--config-name", default="trainer_mri")
    p.add_argument("--data-dir", type=Path, default=Path("src/processed_data_mri"))
    p.add_argument("--num-samples", type=int, default=5, help="Number of inference samples to save")
    p.add_argument("--num-steps", type=int, default=1, help="Denoising steps (default: 1, matching training config)")
    p.add_argument("--evaluate", action="store_true", help="Also run full test-set evaluation with metrics")
    p.add_argument("--skip-slices", type=int, default=10, help="Skip edge slices during evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id

    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    gcs_base = f"{args.bucket}/diamond/runs/{run_id}/training_run/hydra/{run_id}"

    # 1. Download checkpoint
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    existing_pts = sorted(ckpt_dir.glob("*.pt"))

    if existing_pts and args.epoch is None:
        ckpt_path = existing_pts[-1]
        print(f"[1/3] Checkpoint already downloaded: {ckpt_path.name}")
    else:
        print(f"[1/3] Finding checkpoint (epoch={args.epoch or 'latest'}) ...")
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

    # 2. Run inference samples
    out_dir = run_dir / "inference_output"
    print(f"\n[2/3] Running inference ({args.num_samples} samples, {args.num_steps} denoising steps) ...")
    inference_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_inference_25d.py"),
        "--checkpoint", str(ckpt_path),
        "--data-dir", str(args.data_dir),
        "--config-name", args.config_name,
        "--num-samples", str(args.num_samples),
        "--num-steps", str(args.num_steps),
        "--out-dir", str(out_dir),
    ]
    result = subprocess.run(inference_cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print("  Inference failed.")
        sys.exit(1)
    print(f"       Samples saved to {out_dir}/")

    # 3. Run evaluation (optional)
    if args.evaluate:
        print(f"\n[3/3] Running full test-set evaluation ...")
        eval_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_25d.py"),
            "--checkpoint", str(ckpt_path),
            "--data-dir", str(args.data_dir),
            "--config-name", args.config_name,
            "--num-steps", str(args.num_steps),
            "--skip-slices", str(args.skip_slices),
            "--csv-out", str(run_dir / "eval_results.csv"),
        ]
        subprocess.run(eval_cmd, cwd=str(REPO_ROOT))
    else:
        print("\n[3/3] Skipping evaluation (use --evaluate to enable)")

    print(f"\nDone! Run artifacts: {run_dir}/")


if __name__ == "__main__":
    main()
