#!/usr/bin/env bash
set -euo pipefail

# Expected env vars:
# - OUTPUT_URI: gs://.../runs/<run_id>
# - PREPROCESSED_DATA_URI: gs://... prefix containing already-processed dataset (low_res/full_res)
# Optional:
# - RUN_NAME, TRAIN_3D_ARGS (extra CLI args for train_3d.py)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${OUTPUT_URI:?OUTPUT_URI is required}"
: "${PREPROCESSED_DATA_URI:?PREPROCESSED_DATA_URI is required}"

RUN_NAME="${RUN_NAME:-vertex-3d-$(date +%Y%m%d-%H%M%S)}"
TRAIN_3D_ARGS="${TRAIN_3D_ARGS:-}"
GCP_PROJECT="${GCP_PROJECT:-}"
CKPT_UPLOAD_POLL_SECONDS="${CKPT_UPLOAD_POLL_SECONDS:-30}"

WORK_ROOT="${WORK_ROOT:-/tmp/diamond_vertex}"
PROCESSED_DIR="${WORK_ROOT}/processed_data"
SAVE_DIR="${WORK_ROOT}/checkpoints_3d"

mkdir -p "${WORK_ROOT}" "${SAVE_DIR}"
PROJECT_ARGS=()
if [[ -n "${GCP_PROJECT}" ]]; then
  PROJECT_ARGS=(--project "${GCP_PROJECT}")
fi

if [[ -z "${WANDB_API_KEY:-}" && -n "${WANDB_SECRET_ID:-}" ]]; then
  export WANDB_SECRET_ID
  python - <<'PY'
import os
from google.cloud import secretmanager

project = os.environ.get("GCP_PROJECT")
secret_id = os.environ["WANDB_SECRET_ID"]
if not project:
    raise RuntimeError("GCP_PROJECT is required when resolving WANDB_SECRET_ID")
name = f"projects/{project}/secrets/{secret_id}/versions/latest"
client = secretmanager.SecretManagerServiceClient()
payload = client.access_secret_version(request={"name": name}).payload.data.decode("utf-8").strip()
if not payload:
    raise RuntimeError("Secret payload is empty")
with open("/tmp/wandb_api_key", "w", encoding="utf-8") as f:
    f.write(payload)
print("Loaded WANDB_API_KEY from Secret Manager")
PY
  export WANDB_API_KEY="$(cat /tmp/wandb_api_key)"
fi

echo "Using preprocessed dataset from ${PREPROCESSED_DATA_URI}"
export PREPROCESSED_DATA_URI PROCESSED_DIR
python - <<'PY'
import os
from pathlib import Path
from google.cloud import storage

src = os.environ["PREPROCESSED_DATA_URI"]
dst_root = Path(os.environ["PROCESSED_DIR"])
project = os.environ.get("GCP_PROJECT") or None

if not src.startswith("gs://"):
    raise ValueError(f"Expected gs:// URI, got {src}")
no_scheme = src[len("gs://"):]
bucket_name, _, prefix = no_scheme.partition("/")
prefix = prefix.rstrip("/")
if not bucket_name or not prefix:
    raise ValueError("PREPROCESSED_DATA_URI must include bucket and prefix path")

client = storage.Client(project=project)

downloaded = 0
for blob in client.list_blobs(bucket_name, prefix=prefix + "/"):
    if blob.name.endswith("/"):
        continue
    rel = blob.name[len(prefix) + 1:]
    out_path = dst_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(out_path))
    downloaded += 1
print(f"Downloaded {downloaded} files into {dst_root}")
PY

echo "Starting 3D training run ${RUN_NAME}"
CHECKPOINTS_DST_PREFIX="${OUTPUT_URI}/checkpoints_3d"
LAST_CKPT_SNAPSHOT=""

compute_checkpoint_snapshot() {
  local ckpt_dir="$1"
  python - "$ckpt_dir" <<'PY'
import hashlib
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
if not ckpt_dir.exists():
    print("missing")
    raise SystemExit(0)

entries = []
for p in sorted(ckpt_dir.rglob("*")):
    if p.is_file():
        st = p.stat()
        rel = p.relative_to(ckpt_dir)
        entries.append(f"{rel}:{st.st_size}:{st.st_mtime_ns}")
payload = "\n".join(entries).encode("utf-8")
print(hashlib.sha1(payload).hexdigest())
PY
}

sync_checkpoints_if_changed() {
  local snap
  snap="$(compute_checkpoint_snapshot "${SAVE_DIR}")"
  if [[ "${snap}" != "${LAST_CKPT_SNAPSHOT}" ]]; then
    if [[ "${snap}" != "missing" ]]; then
      echo "Checkpoint change detected; uploading ${SAVE_DIR} -> ${CHECKPOINTS_DST_PREFIX}"
      python scripts/gcs_data_ops.py "${PROJECT_ARGS[@]}" upload-dir --src-dir "${SAVE_DIR}" --dst-prefix "${CHECKPOINTS_DST_PREFIX}"
    fi
    LAST_CKPT_SNAPSHOT="${snap}"
  fi
}

python src/train_3d.py \
  --lr-train "${PROCESSED_DIR}/low_res/train" \
  --hr-train "${PROCESSED_DIR}/full_res/train" \
  --save-dir "${SAVE_DIR}" \
  --wandb-project mri-super-resolution-3d \
  --wandb-name "${RUN_NAME}" \
  ${TRAIN_3D_ARGS} &
TRAIN_PID=$!

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sync_checkpoints_if_changed
  sleep "${CKPT_UPLOAD_POLL_SECONDS}"
done

wait "${TRAIN_PID}"
sync_checkpoints_if_changed

echo "Uploading run outputs to ${OUTPUT_URI}"
python scripts/gcs_data_ops.py "${PROJECT_ARGS[@]}" upload-dir --src-dir "${SAVE_DIR}" --dst-prefix "${CHECKPOINTS_DST_PREFIX}"

echo "3D run complete: ${RUN_NAME}"
