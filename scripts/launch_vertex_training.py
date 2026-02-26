#!/usr/bin/env python3
"""Local launcher: select recordings, preview sample, and submit a Vertex training job."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import random
import shlex
import subprocess
import sys
import tempfile
import time
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
    no_scheme = uri[len("gs://") :]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    blob = "" if len(parts) == 1 else parts[1]
    if not bucket:
        raise ValueError(f"Missing bucket name in URI: {uri}")
    # Friendly early guard for placeholder examples like gs://.../file.mkv.
    if bucket == "..." or "." in {bucket[0], bucket[-1]}:
        raise ValueError(
            f"Invalid bucket name '{bucket}' in URI: {uri}. "
            "Use a real bucket like gs://my-bucket/path/file.mkv."
        )
    return bucket, blob


def parse_recording_arg(value: str) -> tuple[str, str]:
    # Supported forms:
    # - gs://bucket/path/file.mkv     (defaults to train)
    # - train=gs://bucket/path/file.mkv
    # - test=gs://bucket/path/file.mkv
    if "=" in value:
        split, uri = value.split("=", 1)
        split = split.strip().lower()
    else:
        split, uri = "train", value
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split '{split}' in --recording-uri {value}")
    if not uri.endswith(".mkv"):
        raise ValueError(f"Recording must end with .mkv: {uri}")
    return split, uri


def parse_manifest(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split()
        if len(fields) == 1:
            split, uri = "train", fields[0]
        elif len(fields) == 2:
            split, uri = fields
        else:
            raise ValueError(f"Bad manifest line: {raw}")
        split = split.lower()
        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split '{split}' in line: {raw}")
        out.append((split, uri))
    return out


def list_from_prefix(prefix_uri: str, project: str | None) -> list[tuple[str, str]]:
    bucket_name, prefix = parse_gs_uri(prefix_uri)
    client = storage.Client(project=project) if project else storage.Client()
    entries: list[tuple[str, str]] = []
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if not blob.name.endswith(".mkv"):
            continue
        uri = f"gs://{bucket_name}/{blob.name}"
        split = "test" if "/test/" in uri else "train"
        entries.append((split, uri))
    return entries


def dedupe(entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for item in entries:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def write_manifest(path: Path, entries: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{split} {uri}" for split, uri in entries]
    path.write_text("\n".join(lines) + "\n")


def download_blob_with_progress(blob: storage.Blob, out_path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob.reload()  # Ensure metadata (size) is available.
    total = blob.size or 0
    downloaded = 0
    last_print = 0.0
    started = time.time()

    with blob.open("rb") as src, open(out_path, "wb") as dst:
        while True:
            chunk = src.read(chunk_bytes)
            if not chunk:
                break
            dst.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 0.5:
                if total > 0:
                    pct = 100.0 * downloaded / total
                    print(f"    {pct:6.2f}% ({downloaded / 1_048_576:.1f}/{total / 1_048_576:.1f} MiB)", end="\r")
                else:
                    print(f"    downloaded {downloaded / 1_048_576:.1f} MiB", end="\r")
                last_print = now

    elapsed = max(time.time() - started, 1e-6)
    rate_mib_s = (downloaded / 1_048_576) / elapsed
    if total > 0:
        print(f"    100.00% ({downloaded / 1_048_576:.1f}/{total / 1_048_576:.1f} MiB) @ {rate_mib_s:.1f} MiB/s")
    else:
        print(f"    downloaded {downloaded / 1_048_576:.1f} MiB @ {rate_mib_s:.1f} MiB/s")


def download_preview_sample(
    entries: list[tuple[str, str]],
    sample_count: int,
    raw_dir: Path,
    project: str | None,
) -> None:
    sample = random.sample(entries, min(sample_count, len(entries)))
    client = storage.Client(project=project) if project else storage.Client()
    for split, uri in sample:
        bucket_name, blob_name = parse_gs_uri(uri)
        filename = blob_name.replace("/", "_")
        out_path = raw_dir / split / filename
        print(f"Preview sample download: {uri} -> {out_path}")
        download_blob_with_progress(client.bucket(bucket_name).blob(blob_name), out_path)


def run_preview_pipeline(args: argparse.Namespace, raw_dir: Path, processed_dir: Path) -> Path | None:
    preprocess_cmd = [
        sys.executable,
        "src/raw_data/scripts/main.py",
        str(processed_dir),
        "--raw-dir",
        str(raw_dir),
        "--chunk-size",
        str(args.preview_chunk_size),
    ]
    if args.use_depth:
        preprocess_cmd.extend(
            [
                "--use-depth",
                "--depth-min-mm",
                str(args.depth_min_mm),
                "--depth-max-mm",
                str(args.depth_max_mm),
            ]
        )
    subprocess.run(
        preprocess_cmd,
        check=True,
    )
    preview_data_dir = processed_dir / "full_res" / "train"
    if not preview_data_dir.exists() or not any(preview_data_dir.glob("*.hdf5")):
        preview_data_dir = processed_dir / "full_res" / "test"

    subprocess.run(
        [
            sys.executable,
            "scripts/visualize_training_data.py",
            "--num-seconds",
            str(args.preview_seconds),
            "--fps",
            str(args.preview_fps),
            "--num-chunks",
            str(args.preview_num_chunks),
            "--data-dir",
            str(preview_data_dir),
            "--out",
            str(args.preview_output),
        ],
        check=True,
    )
    if not args.use_depth:
        return None

    depth_hist_output = (
        args.preview_depth_hist_output
        if args.preview_depth_hist_output is not None
        else args.preview_output.with_name(f"{args.preview_output.stem}_depth_hist.png")
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/render_depth_histogram.py",
            "--data-dir",
            str(preview_data_dir),
            "--num-chunks",
            str(args.preview_num_chunks),
            "--depth-min-mm",
            str(args.depth_min_mm),
            "--depth-max-mm",
            str(args.depth_max_mm),
            "--out",
            str(depth_hist_output),
        ],
        check=True,
    )
    return depth_hist_output


def maybe_open_preview(preview_outputs: list[Path]) -> None:
    answer = input("Do you want to open the training preview? (Y/n) ").strip().lower()
    if answer not in {"", "y", "yes"}:
        return
    paths = [p for p in preview_outputs if p is not None]
    if sys.platform == "darwin":
        for path in paths:
            subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        for path in paths:
            subprocess.run(["xdg-open", str(path)], check=False)
    elif sys.platform == "win32":
        for path in paths:
            subprocess.run(["cmd", "/c", "start", "", str(path)], check=False)
    else:
        print(f"Unsupported platform for auto-open: {sys.platform}")


def upload_manifest(local_manifest: Path, output_uri: str, project: str | None) -> str:
    bucket_name, prefix = parse_gs_uri(output_uri)
    prefix = prefix.rstrip("/")
    dst_blob = f"{prefix}/recordings_manifest.txt"
    dst_uri = f"gs://{bucket_name}/{dst_blob}"
    client = storage.Client(project=project) if project else storage.Client()
    client.bucket(bucket_name).blob(dst_blob).upload_from_filename(str(local_manifest))
    return dst_uri


def print_submission_summary(
    *,
    run_name: str,
    display_name: str,
    config_name: str,
    project: str,
    region: str,
    service_account: str,
    image_uri: str,
    output_uri: str,
    manifest_local: Path,
    entries: list[tuple[str, str]],
    args: argparse.Namespace,
    train_overrides: str,
) -> None:
    n_train = sum(1 for split, _ in entries if split == "train")
    n_test = len(entries) - n_train
    print("\n" + "=" * 72)
    print("Vertex job submission summary")
    print("=" * 72)
    print(f"Run name:        {run_name}")
    print(f"Display name:    {display_name}")
    print(f"Hydra config:    {config_name}")
    print(f"Project/region:  {project} / {region}")
    print(f"Service account: {service_account}")
    print(f"Image:           {image_uri}")
    print(f"Output URI:      {output_uri}")
    print(f"Manifest local:  {manifest_local}")
    print(f"Recordings:      total={len(entries)} train={n_train} test={n_test}")
    print(
        "Hardware:        "
        f"{args.machine_type}, {args.accelerator_type} x{args.accelerator_count}, "
        f"boot_disk={args.boot_disk_size_gb}GB"
    )
    print(
        "Preprocess:      "
        f"chunk_size={args.chunk_size}, use_depth={'yes' if args.use_depth else 'no'}, "
        f"depth_clip=[{args.depth_min_mm}, {args.depth_max_mm}]"
    )
    print(f"Smoke test:      {'yes' if args.smoke_test else 'no'}")
    print(f"Train overrides: {train_overrides if train_overrides else '(none)'}")
    if entries:
        print("Selected recordings (first 5):")
        for split, uri in entries[:5]:
            print(f"  - {split}: {uri}")
        if len(entries) > 5:
            print(f"  ... and {len(entries) - 5} more")
    print("=" * 72 + "\n")


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
        overrides.append(f"denoiser.training.steps_per_epoch={args.steps_per_epoch}")
        overrides.append(f"upsampler.training.steps_per_epoch={args.steps_per_epoch}")
        overrides.append(f"denoiser.training.steps_first_epoch={args.steps_per_epoch}")
        overrides.append(f"upsampler.training.steps_first_epoch={args.steps_per_epoch}")
    if args.denoiser_steps_per_epoch is not None:
        overrides.append(f"denoiser.training.steps_per_epoch={args.denoiser_steps_per_epoch}")
        overrides.append(f"denoiser.training.steps_first_epoch={args.denoiser_steps_per_epoch}")
    if args.upsampler_steps_per_epoch is not None:
        overrides.append(f"upsampler.training.steps_per_epoch={args.upsampler_steps_per_epoch}")
        overrides.append(f"upsampler.training.steps_first_epoch={args.upsampler_steps_per_epoch}")
    if args.lr is not None:
        overrides.append(f"denoiser.optimizer.lr={args.lr}")
    if args.autoregressive_steps is not None:
        overrides.append(f"denoiser.training.num_autoregressive_steps={args.autoregressive_steps}")
    if args.grad_acc_steps is not None:
        overrides.append(f"denoiser.training.grad_acc_steps={args.grad_acc_steps}")
        overrides.append(f"upsampler.training.grad_acc_steps={args.grad_acc_steps}")
    if args.train_overrides:
        overrides.extend(shlex.split(args.train_overrides))
    overrides.append(f"use_depth={'true' if args.use_depth else 'false'}")
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
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--display-name-prefix", default="diamond-train")
    parser.add_argument("--recording-uri", action="append", default=[])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--prefix", action="append", default=[], help="GCS prefix URI to include recursively")
    parser.add_argument("--preview-sample-count", type=int, default=2)
    parser.add_argument("--preview-chunk-size", type=int, default=200)
    parser.add_argument(
        "--preview-num-chunks",
        type=int,
        default=5,
        help="Render this many evenly spaced preview chunks (or fewer if unavailable)",
    )
    parser.add_argument(
        "--preview-seconds",
        type=int,
        default=32,
        help="Total preview duration in seconds (default: 32, ~4x longer than before)",
    )
    parser.add_argument("--preview-fps", type=int, default=15)
    parser.add_argument("--preview-output", type=Path, default=Path("training_preview.mp4"))
    parser.add_argument(
        "--preview-depth-hist-output",
        type=Path,
        default=None,
        help="Optional output PNG for depth histogram preview (default: <preview-output>_depth_hist.png)",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        default=False,
        help="Enable depth channel preprocessing/training for this run",
    )
    parser.add_argument("--depth-min-mm", type=int, default=200)
    parser.add_argument("--depth-max-mm", type=int, default=1500)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--smoke-test", action="store_true", help="Use hydra config trainer_smoke (shortcut for --config-name trainer_smoke)")
    parser.add_argument("--config-name", default=None, help="Hydra config name (default: trainer). Overrides --smoke-test.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.num_final_epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Set denoiser/upsampler steps per epoch")
    parser.add_argument("--denoiser-steps-per-epoch", type=int, default=None, help="Override denoiser steps per epoch")
    parser.add_argument("--upsampler-steps-per-epoch", type=int, default=None, help="Override upsampler steps per epoch")
    parser.add_argument("--eval-every", type=int, default=None, help="Override evaluation.every")
    parser.add_argument("--from-checkpoint", default=None, help="Previous run name (e.g. run-20260225-060050) to resume from latest checkpoint")
    parser.add_argument("--lr", type=float, default=None, help="Override optimizer learning rate for both denoiser and upsampler")
    parser.add_argument("--autoregressive-steps", type=int, default=None, help="Override denoiser.training.num_autoregressive_steps")
    parser.add_argument("--grad-acc-steps", type=int, default=None, help="Override grad_acc_steps for both denoiser and upsampler")
    parser.add_argument("--train-overrides", default="")
    parser.add_argument("--machine-type", default="a2-highgpu-1g")
    parser.add_argument("--accelerator-type", default="NVIDIA_TESLA_A100")
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument("--boot-disk-size-gb", type=int, default=500)
    parser.add_argument("--skip-preview", action="store_true")
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

    run_name = args.run_name or datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    display_name = f"{args.display_name_prefix}-{run_name}"
    output_uri = f"{artifact_bucket_uri.rstrip('/')}/runs/{run_name}"
    if args.config_name:
        config_name = args.config_name
    elif args.smoke_test:
        config_name = "trainer_smoke"
    else:
        config_name = "trainer"
    train_overrides = build_train_override_string(args)

    init_checkpoint_uri: str | None = None
    if args.from_checkpoint:
        print(f"Resolving latest checkpoint from run: {args.from_checkpoint}")
        init_checkpoint_uri = resolve_checkpoint_uri(
            args.from_checkpoint, artifact_bucket_uri, project
        )
        print(f"  -> {init_checkpoint_uri}")

    entries: list[tuple[str, str]] = []
    for r in args.recording_uri:
        entries.append(parse_recording_arg(r))
    if args.manifest:
        entries.extend(parse_manifest(args.manifest))
    for prefix in args.prefix:
        entries.extend(list_from_prefix(prefix, project))
    entries = dedupe(entries)

    if not entries:
        raise ValueError("No recordings selected. Provide --recording-uri, --manifest, or --prefix.")

    with ExitStack() as stack:
        tmp_root = Path(".tmp")
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(
                    dir=str(tmp_root),
                    prefix=f"vertex_launch_{run_name}_",
                )
            )
        )
        raw_preview_dir = tmp_dir / "raw_data"
        processed_preview_dir = tmp_dir / "processed_preview"
        manifest_local = tmp_dir / "recordings_manifest.txt"

        write_manifest(manifest_local, entries)
        print(f"Selected {len(entries)} recording(s). Manifest: {manifest_local}")

        if not args.skip_preview:
            download_preview_sample(entries, args.preview_sample_count, raw_preview_dir, project)
            depth_hist_output = run_preview_pipeline(args, raw_preview_dir, processed_preview_dir)
            print(f"Preview video: {args.preview_output}")
            if depth_hist_output is not None:
                print(f"Preview depth histogram: {depth_hist_output}")
                print("Depth format: HDF5 stores depth as uint8 in channel 4 (index 3).")
                print("Model input: loader maps depth with x/255*2-1 to [-1, 1].")
            maybe_open_preview([args.preview_output, depth_hist_output] if depth_hist_output else [args.preview_output])

        print_submission_summary(
            run_name=run_name,
            display_name=display_name,
            config_name=config_name,
            project=project,
            region=region,
            service_account=service_account,
            image_uri=image_uri,
            output_uri=output_uri,
            manifest_local=manifest_local,
            entries=entries,
            args=args,
            train_overrides=train_overrides,
        )

        if not args.yes and not args.skip_preview:
            answer = input("Submit Vertex job with this recording set? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                print("Cancelled.")
                return

        manifest_uri = upload_manifest(manifest_local, output_uri, project)
        print(f"Uploaded run manifest to {manifest_uri}")

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
            "--recordings-manifest-uri",
            manifest_uri,
            "--output-uri",
            output_uri,
            "--wandb-secret",
            args.wandb_secret,
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
            "--chunk-size",
            str(args.chunk_size),
            "--depth-min-mm",
            str(args.depth_min_mm),
            "--depth-max-mm",
            str(args.depth_max_mm),
            "--train-overrides",
            train_overrides,
        ]
        if args.use_depth:
            submit_cmd.append("--use-depth")
        if init_checkpoint_uri:
            submit_cmd.extend(["--init-checkpoint-uri", init_checkpoint_uri])
        if args.dry_run:
            submit_cmd.append("--dry-run")
        subprocess.run(submit_cmd, check=True)


if __name__ == "__main__":
    main()
