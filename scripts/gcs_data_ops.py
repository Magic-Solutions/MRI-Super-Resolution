#!/usr/bin/env python3
"""Utilities to upload run artifacts and processed data from local disk to GCS."""

from __future__ import annotations

import argparse
from pathlib import Path

from google.cloud import storage


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    no_scheme = uri[len("gs://") :]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    blob = "" if len(parts) == 1 else parts[1]
    return bucket, blob


def upload_file(src_file: Path, dst_uri: str, project: str | None = None) -> None:
    client = storage.Client(project=project) if project else storage.Client()
    bucket_name, blob_name = parse_gs_uri(dst_uri)
    if not blob_name:
        raise ValueError(f"Destination file URI must include object path: {dst_uri}")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    print(f"Uploading {src_file} -> {dst_uri}")
    blob.upload_from_filename(str(src_file))


def upload_directory(src_dir: Path, dst_prefix_uri: str, project: str | None = None) -> None:
    client = storage.Client(project=project) if project else storage.Client()
    bucket_name, prefix = parse_gs_uri(dst_prefix_uri)
    prefix = prefix.rstrip("/")
    bucket = client.bucket(bucket_name)

    if not src_dir.exists():
        print(f"Skipping upload; directory does not exist: {src_dir}")
        return

    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_dir).as_posix()
        blob_name = f"{prefix}/{rel}" if prefix else rel
        print(f"Uploading {path} -> gs://{bucket_name}/{blob_name}")
        bucket.blob(blob_name).upload_from_filename(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None, help="Optional GCP project for Storage client")
    sub = parser.add_subparsers(dest="cmd", required=True)

    uf = sub.add_parser("upload-file", help="Upload one file to GCS")
    uf.add_argument("--src-file", type=Path, required=True)
    uf.add_argument("--dst-uri", required=True, help="Destination gs://bucket/path/file.ext")

    ud = sub.add_parser("upload-dir", help="Upload a local directory tree to GCS prefix")
    ud.add_argument("--src-dir", type=Path, required=True)
    ud.add_argument("--dst-prefix", required=True, help="Destination prefix gs://bucket/path")

    args = parser.parse_args()
    if args.cmd == "upload-file":
        upload_file(args.src_file, args.dst_uri, project=args.project)
    elif args.cmd == "upload-dir":
        upload_directory(args.src_dir, args.dst_prefix, project=args.project)
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
