#!/usr/bin/env bash
set -euo pipefail

# Expected env vars:
# - RECORDINGS_MANIFEST_URI: gs://.../recordings_manifest.txt
# - OUTPUT_URI: gs://.../runs/<run_id>
# Optional:
# - RUN_NAME, CONFIG_NAME, USE_DEPTH, CHUNK_SIZE, DEPTH_MIN_MM, DEPTH_MAX_MM, TRAIN_OVERRIDES

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${RECORDINGS_MANIFEST_URI:?RECORDINGS_MANIFEST_URI is required}"
: "${OUTPUT_URI:?OUTPUT_URI is required}"

RUN_NAME="${RUN_NAME:-vertex-run-$(date +%Y%m%d-%H%M%S)}"
CONFIG_NAME="${CONFIG_NAME:-trainer}"
USE_DEPTH="${USE_DEPTH:-false}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
DEPTH_MIN_MM="${DEPTH_MIN_MM:-200}"
DEPTH_MAX_MM="${DEPTH_MAX_MM:-3000}"
TRAIN_OVERRIDES="${TRAIN_OVERRIDES:-}"
GCP_PROJECT="${GCP_PROJECT:-}"
CKPT_UPLOAD_POLL_SECONDS="${CKPT_UPLOAD_POLL_SECONDS:-30}"

WORK_ROOT="${WORK_ROOT:-/tmp/diamond_vertex}"
RAW_DIR="${WORK_ROOT}/raw_data"
PROCESSED_DIR="${WORK_ROOT}/processed_data"
RUN_DIR="${WORK_ROOT}/run"
MANIFEST_LOCAL="${WORK_ROOT}/recordings_manifest.txt"

mkdir -p "${WORK_ROOT}"
export RECORDINGS_MANIFEST_URI MANIFEST_LOCAL GCP_PROJECT
PROJECT_ARGS=()
if [[ -n "${GCP_PROJECT}" ]]; then
  PROJECT_ARGS=(--project "${GCP_PROJECT}")
fi

echo "Downloading manifest ${RECORDINGS_MANIFEST_URI}"
python - <<'PY'
import os
from google.cloud import storage

src = os.environ["RECORDINGS_MANIFEST_URI"]
dst = os.environ["MANIFEST_LOCAL"]
bucket = src[len("gs://"):].split("/", 1)[0]
blob = src[len("gs://")+len(bucket)+1:]
client = storage.Client(project=os.environ.get("GCP_PROJECT") or None)
client.bucket(bucket).blob(blob).download_to_filename(dst)
print(f"Downloaded manifest to {dst}")
PY

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

echo "Downloading recordings listed in manifest"
python scripts/gcs_data_ops.py "${PROJECT_ARGS[@]}" download-manifest --manifest "${MANIFEST_LOCAL}" --dest-raw-dir "${RAW_DIR}"

echo "Preprocessing recordings"
PREPROCESS_ARGS=(
  "${PROCESSED_DIR}"
  --raw-dir "${RAW_DIR}"
  --chunk-size "${CHUNK_SIZE}"
)
if [[ "${USE_DEPTH,,}" == "true" ]]; then
  PREPROCESS_ARGS+=(--use-depth --depth-min-mm "${DEPTH_MIN_MM}" --depth-max-mm "${DEPTH_MAX_MM}")
fi
python src/raw_data/scripts/main.py "${PREPROCESS_ARGS[@]}"

mkdir -p "${RUN_DIR}"
HYDRA_RUN_DIR="${RUN_DIR}/hydra/${RUN_NAME}"
mkdir -p "${HYDRA_RUN_DIR}"
cat > "${RUN_DIR}/run_metadata.json" <<EOF
{
  "run_name": "${RUN_NAME}",
  "recordings_manifest_uri": "${RECORDINGS_MANIFEST_URI}",
  "output_uri": "${OUTPUT_URI}",
  "use_depth": "${USE_DEPTH}",
  "chunk_size": ${CHUNK_SIZE},
  "depth_min_mm": ${DEPTH_MIN_MM},
  "depth_max_mm": ${DEPTH_MAX_MM}
}
EOF

echo "Starting training run ${RUN_NAME}"
CHECKPOINTS_DIR="${HYDRA_RUN_DIR}/checkpoints"
CHECKPOINTS_DST_PREFIX="${OUTPUT_URI}/training_run/hydra/${RUN_NAME}/checkpoints"
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
  snap="$(compute_checkpoint_snapshot "${CHECKPOINTS_DIR}")"
  if [[ "${snap}" != "${LAST_CKPT_SNAPSHOT}" ]]; then
    if [[ "${snap}" != "missing" ]]; then
      echo "Checkpoint change detected; uploading ${CHECKPOINTS_DIR} -> ${CHECKPOINTS_DST_PREFIX}"
      python scripts/gcs_data_ops.py "${PROJECT_ARGS[@]}" upload-dir --src-dir "${CHECKPOINTS_DIR}" --dst-prefix "${CHECKPOINTS_DST_PREFIX}"
    fi
    LAST_CKPT_SNAPSHOT="${snap}"
  fi
}

python src/main.py \
  --config-name "${CONFIG_NAME}" \
  "use_depth=${USE_DEPTH,,}" \
  "env.path_data_low_res=${PROCESSED_DIR}/low_res" \
  "env.path_data_full_res=${PROCESSED_DIR}/full_res" \
  "wandb.name=${RUN_NAME}" \
  "hydra.run.dir=${HYDRA_RUN_DIR}" \
  ${TRAIN_OVERRIDES} &
TRAIN_PID=$!

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sync_checkpoints_if_changed
  sleep "${CKPT_UPLOAD_POLL_SECONDS}"
done

wait "${TRAIN_PID}"
sync_checkpoints_if_changed

echo "Uploading run outputs to ${OUTPUT_URI}"
python scripts/gcs_data_ops.py "${PROJECT_ARGS[@]}" upload-dir --src-dir "${RUN_DIR}" --dst-prefix "${OUTPUT_URI}/training_run"
python scripts/gcs_data_ops.py "${PROJECT_ARGS[@]}" upload-dir --src-dir "${PROCESSED_DIR}" --dst-prefix "${OUTPUT_URI}/processed_data"
python scripts/gcs_data_ops.py "${PROJECT_ARGS[@]}" upload-file --src-file "${MANIFEST_LOCAL}" --dst-uri "${OUTPUT_URI}/recordings_manifest.txt"

echo "Run complete: ${RUN_NAME}"
