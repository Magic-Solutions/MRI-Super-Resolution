#!/usr/bin/env python3
"""Submit a Vertex AI Custom Job for MRI super-resolution training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def build_job_config(args: argparse.Namespace) -> dict:
    container_env = [
        {"name": "OUTPUT_URI", "value": args.output_uri},
        {"name": "RUN_NAME", "value": args.run_name},
        {"name": "PREPROCESSED_DATA_URI", "value": args.preprocessed_data_uri},
        {"name": "WANDB_SECRET_ID", "value": args.wandb_secret},
    ]
    if args.mode == "2.5d":
        container_env.append({"name": "CONFIG_NAME", "value": args.config_name})
        if args.train_overrides:
            container_env.append({"name": "TRAIN_OVERRIDES", "value": args.train_overrides})
        if args.init_checkpoint_uri:
            container_env.append({"name": "INIT_CHECKPOINT_URI", "value": args.init_checkpoint_uri})
        training_script = "scripts/run_training_job.sh"
    else:
        if args.train_3d_args:
            container_env.append({"name": "TRAIN_3D_ARGS", "value": args.train_3d_args})
        training_script = "scripts/run_training_job_3d.sh"
    if args.project:
        container_env.append({"name": "GCP_PROJECT", "value": args.project})

    return {
        "serviceAccount": args.service_account,
        "baseOutputDirectory": {"outputUriPrefix": args.output_uri},
        "workerPoolSpecs": [
            {
                "replicaCount": "1",
                "machineSpec": {
                    "machineType": args.machine_type,
                    "acceleratorType": args.accelerator_type,
                    "acceleratorCount": args.accelerator_count,
                },
                "diskSpec": {
                    "bootDiskType": args.boot_disk_type,
                    "bootDiskSizeGb": args.boot_disk_size_gb,
                },
                "containerSpec": {
                    "imageUri": args.image_uri,
                    "command": ["bash", training_script],
                    "env": container_env,
                },
            }
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument(
        "--preprocessed-data-uri",
        required=True,
        help="gs://... prefix containing already processed MRI dataset (low_res/full_res).",
    )
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--wandb-secret", required=True, help="Secret Manager secret ID containing WANDB_API_KEY")
    parser.add_argument("--mode", choices=["2.5d", "3d"], default="2.5d", help="Training mode: 2.5d (Hydra-based) or 3d (standalone)")
    parser.add_argument("--config-name", default="trainer_mri", help="Hydra config name for 2.5d mode")
    parser.add_argument("--machine-type", default="a2-highgpu-1g")
    parser.add_argument("--accelerator-type", default="NVIDIA_TESLA_A100")
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument("--boot-disk-type", default="pd-ssd")
    parser.add_argument("--boot-disk-size-gb", type=int, default=500)
    parser.add_argument("--train-overrides", default="", help="Hydra overrides for 2.5d mode")
    parser.add_argument("--train-3d-args", default="", help="Extra CLI args for train_3d.py in 3d mode")
    parser.add_argument("--init-checkpoint-uri", default=None, help="GCS URI of checkpoint to initialize from (2.5d only)")
    parser.add_argument("--wait-for-start", dest="wait_for_start", action="store_true", default=True, help="Wait and report until job enters RUNNING state")
    parser.add_argument("--no-wait-for-start", dest="wait_for_start", action="store_false", help="Return immediately after submission")
    parser.add_argument("--start-timeout-seconds", type=int, default=900, help="Max seconds to wait for RUNNING state")
    parser.add_argument("--poll-interval-seconds", type=int, default=15, help="Polling interval while waiting for start")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def describe_custom_job(custom_job_name: str, project: str, region: str) -> dict:
    cmd = [
        "gcloud",
        "ai",
        "custom-jobs",
        "describe",
        custom_job_name,
        "--project",
        project,
        "--region",
        region,
        "--format",
        "json",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def wait_for_job_start(custom_job_name: str, project: str, region: str, timeout_seconds: int, poll_interval_seconds: int) -> None:
    print(f"Waiting for job to start (timeout: {timeout_seconds}s, poll: {poll_interval_seconds}s)...")
    started = time.time()
    last_state = None
    terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"}
    while True:
        info = describe_custom_job(custom_job_name, project, region)
        state = info.get("state") or info.get("jobState") or "UNKNOWN"
        if state != last_state:
            print(f"  state: {state}")
            last_state = state

        if state == "JOB_STATE_RUNNING":
            print("Job is now running.")
            return
        if state in terminal_states:
            print(f"Job reached terminal state before RUNNING: {state}")
            return
        if time.time() - started >= timeout_seconds:
            print("Timed out waiting for RUNNING state. Job may still be queued/pending.")
            return
        time.sleep(max(1, poll_interval_seconds))


def main() -> None:
    args = parse_args()
    config = build_job_config(args)
    tmp_path = Path(f"/tmp/{args.run_name}_vertex_job.json")
    tmp_path.write_text(json.dumps(config, indent=2))
    print(f"Wrote config to {tmp_path}")
    print(json.dumps(config, indent=2))

    if args.dry_run:
        return

    cmd = [
        "gcloud",
        "ai",
        "custom-jobs",
        "create",
        "--display-name",
        args.display_name,
        "--region",
        args.region,
        "--project",
        args.project,
        "--config",
        str(tmp_path),
        "--format",
        "json",
    ]
    print("Submitting Vertex custom job... (this may take 30-90s)")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    try:
        job_info = json.loads(result.stdout)
    except json.JSONDecodeError:
        job_info = {}
    custom_job_name = job_info.get("name")
    if custom_job_name:
        print("\nVertex job submitted.")
        print(f"Custom job: {custom_job_name}")
        print("View status:")
        print(f"  gcloud ai custom-jobs describe {custom_job_name} --region {args.region} --project {args.project}")
        print("Tail logs:")
        print(
            "  gcloud logging read "
            f"\"resource.type=\\\"aiplatform.googleapis.com/CustomJob\\\" "
            f"AND resource.labels.job_id=\\\"{custom_job_name.split('/')[-1]}\\\"\" "
            f"--project {args.project} --limit 50 --format='value(timestamp,textPayload)'"
        )
        if args.wait_for_start:
            wait_for_job_start(
                custom_job_name,
                project=args.project,
                region=args.region,
                timeout_seconds=args.start_timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
            )


if __name__ == "__main__":
    main()
