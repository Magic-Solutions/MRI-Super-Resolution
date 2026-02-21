#!/usr/bin/env python3
"""Utilities to move selected recording data and artifacts between local disk and GCS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from google.cloud import storage


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    no_scheme = uri[len("gs://") :]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    blob = "" if len(parts) == 1 else parts[1]
    return bucket, blob


def parse_manifest_line(line: str) -> tuple[str, str]:
    line = line.strip()
    if not line or line.startswith("#"):
        raise ValueError("ignore")
    fields = line.split()
    if len(fields) == 1:
        split, uri = "train", fields[0]
    elif len(fields) == 2:
        split, uri = fields
    else:
        raise ValueError(f"Invalid manifest line: {line}")
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split '{split}' in line: {line}")
    if not uri.endswith(".mkv"):
        raise ValueError(f"Expected .mkv URI in line: {line}")
    return split, uri


def iter_manifest_entries(path: Path) -> Iterable[tuple[str, str]]:
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield parse_manifest_line(stripped)


def download_manifest(manifest_path: Path, dest_raw_dir: Path, project: str | None = None) -> None:
    client = storage.Client(project=project) if project else storage.Client()
    for split, uri in iter_manifest_entries(manifest_path):
        bucket_name, blob_name = parse_gs_uri(uri)
        if not blob_name:
            raise ValueError(f"Manifest URI has empty object path: {uri}")
        filename = blob_name.replace("/", "_")
        local_path = dest_raw_dir / split / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            print(f"SKIP existing {local_path}")
            continue
        print(f"Downloading {uri} -> {local_path}")
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(local_path))


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

    d = sub.add_parser("download-manifest", help="Download all recordings listed in a manifest")
    d.add_argument("--manifest", type=Path, required=True, help="Local manifest path")
    d.add_argument("--dest-raw-dir", type=Path, required=True, help="Destination raw-data root")

    uf = sub.add_parser("upload-file", help="Upload one file to GCS")
    uf.add_argument("--src-file", type=Path, required=True)
    uf.add_argument("--dst-uri", required=True, help="Destination gs://bucket/path/file.ext")

    ud = sub.add_parser("upload-dir", help="Upload a local directory tree to GCS prefix")
    ud.add_argument("--src-dir", type=Path, required=True)
    ud.add_argument("--dst-prefix", required=True, help="Destination prefix gs://bucket/path")

    args = parser.parse_args()
    if args.cmd == "download-manifest":
        download_manifest(args.manifest, args.dest_raw_dir, project=args.project)
    elif args.cmd == "upload-file":
        upload_file(args.src_file, args.dst_uri, project=args.project)
    elif args.cmd == "upload-dir":
        upload_directory(args.src_dir, args.dst_prefix, project=args.project)
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
