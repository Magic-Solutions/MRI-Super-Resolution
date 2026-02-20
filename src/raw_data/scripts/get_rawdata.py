"""Download raw MKV files from GCS into train/test folders.

Reads traintestsplit.txt to determine which bucket objects go into which split.
Each line has the format:

    <split> <bucket_path>

where <bucket_path> is relative to the GCS bucket root.  The file is saved
locally as <folder>_<filename> (slashes replaced with underscores) under
src/raw_data/<split>/.

Usage:
    python get_rawdata.py [--bucket BUCKET] [--split-file SPLIT_FILE]
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent
DEFAULT_SPLIT_FILE = SCRIPT_DIR / "traintestsplit.txt"
DEFAULT_BUCKET = "gs://omgrab-our-exports"


def parse_split_file(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            split, bucket_path = line.split(None, 1)
            if split not in ("train", "test"):
                print(f"WARNING: unknown split '{split}', skipping: {line}")
                continue
            entries.append((split, bucket_path))
    return entries


def local_name(bucket_path: str) -> str:
    """Convert 'om002/cookie_openhand.mkv' -> 'om002_cookie_openhand.mkv'."""
    return bucket_path.replace("/", "_")


def download(bucket: str, bucket_path: str, dest: Path) -> bool:
    src = f"{bucket}/{bucket_path}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  SKIP (already exists): {dest}")
        return True

    print(f"  {src} -> {dest}")
    result = subprocess.run(
        ["gsutil", "-m", "cp", src, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: gsutil failed (exit {result.returncode})")
        if result.stderr:
            print(f"    {result.stderr.strip()}")
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket", default=DEFAULT_BUCKET,
        help=f"GCS bucket URL (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--split-file", type=Path, default=DEFAULT_SPLIT_FILE,
        help=f"Path to train/test split file (default: {DEFAULT_SPLIT_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    entries = parse_split_file(args.split_file)
    if not entries:
        print("No entries found in split file.")
        sys.exit(1)

    print(f"Bucket:     {args.bucket}")
    print(f"Split file: {args.split_file}")
    print(f"Entries:    {len(entries)}\n")

    ok, fail = 0, 0
    for split, bucket_path in entries:
        dest = RAW_DIR / split / local_name(bucket_path)
        print(f"[{split}] {bucket_path}")
        if download(args.bucket, bucket_path, dest):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} downloaded, {fail} failed")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
