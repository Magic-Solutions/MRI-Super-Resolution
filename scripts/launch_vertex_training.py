#!/usr/bin/env python3
"""Local launcher: submit a Vertex AI training job for MRI super-resolution."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from google.cloud import storage


def load_pulumi_outputs(pulumi_dir: Path) -> dict:
    if not (pulumi_dir / "Pulumi.yaml").exists():
        return {}
    try:
        result = subprocess.run(
            ["pulumi", "stack", "output", "--json"],
            cwd=str(pulumi_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    no_scheme = uri[len("gs://"):]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    blob = "" if len(parts) == 1 else parts[1]
    if not bucket:
        raise ValueError(f"Missing bucket name in URI: {uri}")
    if bucket == "..." or "." in {bucket[0], bucket[-1]}:
        raise ValueError(
            f"Invalid bucket name '{bucket}' in URI: {uri}. "
            "Use a real bucket like gs://my-bucket/path/."
        )
    return bucket, blob


def resolve_checkpoint_uri(run_name: str, artifact_bucket_uri: str, project: str | None) -> str:
    """Find the latest agent_epoch_*.pt checkpoint in a previous run."""
    prefix_uri = f"{artifact_bucket_uri.rstrip('/')}/runs/{run_name}/training_run/hydra/{run_name}/checkpoints/"
    bucket_name, prefix = parse_gs_uri(prefix_uri)
    client = storage.Client(project=project) if project else storage.Client()
    candidates: list[str] = []
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith(".pt") and "agent_epoch_" in blob.name:
            candidates.append(blob.name)
    if not candidates:
        raise ValueError(
            f"No agent_epoch_*.pt checkpoint found under gs://{bucket_name}/{prefix}\n"
            f"Check that run '{run_name}' completed and uploaded checkpoints."
        )
    candidates.sort()
    latest = candidates[-1]
    return f"gs://{bucket_name}/{latest}"


def build_train_override_string(args: argparse.Namespace) -> str:
    overrides: list[str] = []
    if args.epochs is not None:
        overrides.append(f"training.num_final_epochs={args.epochs}")
    if args.eval_every is not None:
        overrides.append(f"evaluation.every={args.eval_every}")
    if args.steps_per_epoch is not None:
        overrides.append(f"upsampler.training.steps_per_epoch={args.steps_per_epoch}")
        overrides.append(f"upsampler.training.steps_first_epoch={args.steps_per_epoch}")
    if args.upsampler_steps_per_epoch is not None:
        overrides.append(f"upsampler.training.steps_per_epoch={args.upsampler_steps_per_epoch}")
        overrides.append(f"upsampler.training.steps_first_epoch={args.upsampler_steps_per_epoch}")
    if args.grad_acc_steps is not None:
        overrides.append(f"upsampler.training.grad_acc_steps={args.grad_acc_steps}")
    if args.train_overrides:
        overrides.extend(shlex.split(args.train_overrides))
    return " ".join(overrides)


def build_and_push_latest_image(image_uri: str, project: str, region: str) -> None:
    if not image_uri.endswith(":latest"):
        raise ValueError(
            f"--push-latest requires an image tagged ':latest'. Got: {image_uri}"
        )
    repo_root = Path(__file__).resolve().parent.parent
    print(f"Building and pushing image with Cloud Build: {image_uri}")
    config_content = f"""steps:
- name: gcr.io/cloud-builders/docker
  args: ['build', '-f', 'Dockerfile.vertex', '-t', '{image_uri}', '.']
images:
- '{image_uri}'
"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(config_content)
        config_path = tmp.name
    try:
        subprocess.run(
            [
                "gcloud",
                "builds",
                "submit",
                "--project",
                project,
                "--region",
                region,
                "--config",
                config_path,
                ".",
            ],
            cwd=str(repo_root),
            check=True,
        )
    finally:
        Path(config_path).unlink(missing_ok=True)


def build_train_3d_args_string(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.epochs is not None:
        parts.append(f"--epochs {args.epochs}")
    if args.train_3d_args:
        parts.append(args.train_3d_args)
    return " ".join(parts)


def print_submission_summary(
    *,
    run_name: str,
    display_name: str,
    mode: str,
    project: str,
    region: str,
    service_account: str,
    image_uri: str,
    output_uri: str,
    preprocessed_data_uri: str,
    args: argparse.Namespace,
    extra_info: str,
) -> None:
    print("\n" + "=" * 72)
    print("Vertex job submission summary")
    print("=" * 72)
    print(f"Mode:            {mode}")
    print(f"Run name:        {run_name}")
    print(f"Display name:    {display_name}")
    print(f"Project/region:  {project} / {region}")
    print(f"Service account: {service_account}")
    print(f"Image:           {image_uri}")
    print(f"Output URI:      {output_uri}")
    print(f"Preprocessed:    {preprocessed_data_uri}")
    print(
        "Hardware:        "
        f"{args.machine_type}, {args.accelerator_type} x{args.accelerator_count}, "
        f"boot_disk={args.boot_disk_size_gb}GB"
    )
    print(f"Extra args:      {extra_info if extra_info else '(none)'}")
    print("=" * 72 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--service-account", default=None)
    parser.add_argument("--image-uri", default=None)
    parser.add_argument("--artifact-bucket-uri", default=None, help="Base URI gs://bucket/path for run outputs")
    parser.add_argument("--wandb-secret", default="wandb-api-key")
    parser.add_argument(
        "--pulumi-dir",
        type=Path,
        default=Path("infra/pulumi-gcp"),
        help="Pulumi stack directory used to auto-fill defaults",
    )
    parser.add_argument(
        "--no-pulumi-defaults",
        action="store_true",
        help="Disable auto-loading values from pulumi stack output",
    )
    parser.add_argument("--mode", choices=["2.5d", "3d"], default="2.5d", help="Training mode: 2.5d (slice-based) or 3d (volumetric patches)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--display-name-prefix", default="mri-train")
    parser.add_argument(
        "--preprocessed-data-uri",
        required=True,
        help="gs://... prefix for already-processed MRI dataset (expects low_res/full_res subdirectories).",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Use hydra config trainer_mri_smoke (shortcut for --config-name trainer_mri_smoke)")
    parser.add_argument("--config-name", default=None, help="Hydra config name (default: trainer_mri). Overrides --smoke-test. 2.5d only.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs (both modes)")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Set upsampler steps per epoch (2.5d only)")
    parser.add_argument("--upsampler-steps-per-epoch", type=int, default=None, help="Override upsampler steps per epoch only (2.5d only)")
    parser.add_argument("--eval-every", type=int, default=None, help="Override evaluation.every (2.5d only)")
    parser.add_argument("--from-checkpoint", default=None, help="Previous run name to resume from latest checkpoint (2.5d only)")
    parser.add_argument("--grad-acc-steps", type=int, default=None, help="Override grad_acc_steps (2.5d only)")
    parser.add_argument("--train-overrides", default="", help="Hydra overrides (2.5d only)")
    parser.add_argument("--train-3d-args", default="", help="Extra CLI args for train_3d.py (3d only, e.g. '--batch-size 4 --patch-size 32')")
    parser.add_argument("--machine-type", default="a2-highgpu-1g")
    parser.add_argument("--accelerator-type", default="NVIDIA_TESLA_A100")
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument("--boot-disk-size-gb", type=int, default=500)
    parser.add_argument(
        "--push-latest",
        action="store_true",
        help="Build and push Dockerfile.vertex to the resolved :latest image before other steps",
    )
    parser.add_argument("--yes", action="store_true", help="Submit without interactive confirmation")
    parser.add_argument("--dry-run", action="store_true", help="Prepare everything but do not submit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pulumi_outputs = {} if args.no_pulumi_defaults else load_pulumi_outputs(args.pulumi_dir)
    project = args.project or pulumi_outputs.get("project")
    region = args.region or pulumi_outputs.get("region")
    service_account = args.service_account or pulumi_outputs.get("runtimeServiceAccountEmail")

    artifact_bucket_uri = args.artifact_bucket_uri
    if artifact_bucket_uri is None:
        artifact_bucket = pulumi_outputs.get("artifactBucket")
        if artifact_bucket:
            artifact_bucket_uri = f"gs://{artifact_bucket}/diamond"

    image_uri = args.image_uri
    if image_uri is None:
        image_base = pulumi_outputs.get("artifactImageBase")
        if image_base:
            image_uri = f"{image_base}/diamond-train:latest"

    missing = []
    if not project:
        missing.append("--project")
    if not region:
        missing.append("--region")
    if not service_account:
        missing.append("--service-account")
    if not artifact_bucket_uri:
        missing.append("--artifact-bucket-uri")
    if not image_uri:
        missing.append("--image-uri")
    if missing:
        raise ValueError(
            "Missing required settings: "
            + ", ".join(missing)
            + ". Either pass them explicitly or configure Pulumi outputs."
        )

    if args.push_latest:
        build_and_push_latest_image(image_uri, project=project, region=region)

    mode = args.mode
    prefix = "3d" if mode == "3d" else "25d"
    run_name = args.run_name or datetime.utcnow().strftime(f"run-{prefix}-%Y%m%d-%H%M%S")
    display_name = f"{args.display_name_prefix}-{run_name}"
    output_uri = f"{artifact_bucket_uri.rstrip('/')}/runs/{run_name}"

    if mode == "2.5d":
        if args.config_name:
            config_name = args.config_name
        elif args.smoke_test:
            config_name = "trainer_mri_smoke"
        else:
            config_name = "trainer_mri"
        train_overrides = build_train_override_string(args)
        extra_info = f"config={config_name}  overrides={train_overrides or '(none)'}"
    else:
        config_name = "trainer_mri"
        train_overrides = ""
        train_3d_args = build_train_3d_args_string(args)
        extra_info = f"train_3d_args={train_3d_args or '(defaults)'}"

    init_checkpoint_uri: str | None = None
    if args.from_checkpoint and mode == "2.5d":
        print(f"Resolving latest checkpoint from run: {args.from_checkpoint}")
        init_checkpoint_uri = resolve_checkpoint_uri(
            args.from_checkpoint, artifact_bucket_uri, project
        )
        print(f"  -> {init_checkpoint_uri}")

    print_submission_summary(
        run_name=run_name,
        display_name=display_name,
        mode=mode,
        project=project,
        region=region,
        service_account=service_account,
        image_uri=image_uri,
        output_uri=output_uri,
        preprocessed_data_uri=args.preprocessed_data_uri,
        args=args,
        extra_info=extra_info,
    )

    if not args.yes:
        answer = input("Submit Vertex job? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return

    submit_cmd = [
        sys.executable,
        "scripts/submit_vertex_job.py",
        "--project",
        project,
        "--region",
        region,
        "--display-name",
        display_name,
        "--run-name",
        run_name,
        "--image-uri",
        image_uri,
        "--service-account",
        service_account,
        "--output-uri",
        output_uri,
        "--wandb-secret",
        args.wandb_secret,
        "--mode",
        mode,
        "--config-name",
        config_name,
        "--machine-type",
        args.machine_type,
        "--accelerator-type",
        args.accelerator_type,
        "--accelerator-count",
        str(args.accelerator_count),
        "--boot-disk-size-gb",
        str(args.boot_disk_size_gb),
        "--preprocessed-data-uri",
        args.preprocessed_data_uri,
        "--train-overrides",
        train_overrides,
    ]
    if mode == "3d":
        submit_cmd.extend(["--train-3d-args", train_3d_args])
    if init_checkpoint_uri:
        submit_cmd.extend(["--init-checkpoint-uri", init_checkpoint_uri])
    if args.dry_run:
        submit_cmd.append("--dry-run")
    subprocess.run(submit_cmd, check=True)


if __name__ == "__main__":
    main()
