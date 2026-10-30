#!/usr/bin/env bash
# Vertex job: train the Res-SRDiff comparator (src/train_ressrdiff.py).
# Res-SRDiff is fetched at the commit pinned in src/ressrdiff_adapter.py rather than vendored.
#
# Env: OUTPUT_URI, PREPROCESSED_DATA_URI (the training/ split), TRAIN_RESSRDIFF_ARGS (extra trainer args).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${OUTPUT_URI:?OUTPUT_URI is required}"
: "${PREPROCESSED_DATA_URI:?PREPROCESSED_DATA_URI is required}"
TRAIN_RESSRDIFF_ARGS="${TRAIN_RESSRDIFF_ARGS:-}"
CKPT_UPLOAD_POLL_SECONDS="${CKPT_UPLOAD_POLL_SECONDS:-60}"
WORK_ROOT="${WORK_ROOT:-/tmp/ressrdiff_vertex}"
PROCESSED_DIR="${WORK_ROOT}/processed_data"
SAVE_DIR="${WORK_ROOT}/checkpoints_ressrdiff"
CODE_DIR="${WORK_ROOT}/Res-SRDiff"
RESSRDIFF_COMMIT="$(python -c 'import sys; sys.path.insert(0, "src"); import ressrdiff_adapter as a; print(a.RESSRDIFF_COMMIT)')"
mkdir -p "${WORK_ROOT}" "${SAVE_DIR}"

# Res-SRDiff's own dependencies, at the versions its environment.yml pins.
pip install --no-cache-dir piq==0.8.0 einops==0.7.0 timm==1.0.14
python -c "import torch, timm, piq, einops; print('torch', torch.__version__, 'timm', timm.__version__)"

# The base image has no git; fetch the pinned commit as a GitHub archive instead.
python - "${RESSRDIFF_COMMIT}" "${CODE_DIR}" <<'PY'
import io, shutil, sys, tarfile, urllib.request
commit, dst = sys.argv[1], sys.argv[2]
data = urllib.request.urlopen(f"https://github.com/mosaf/Res-SRDiff/archive/{commit}.tar.gz", timeout=120).read()
with tarfile.open(fileobj=io.BytesIO(data)) as tar:
    root = tar.getnames()[0].split("/")[0]
    tar.extractall("/tmp/ressrdiff_src")
shutil.move(f"/tmp/ressrdiff_src/{root}", dst)
print(f"Res-SRDiff {commit} extracted to {dst}")
PY

export PREPROCESSED_DATA_URI PROCESSED_DIR
python - <<'PY'
import os
from pathlib import Path
from google.cloud import storage
src = os.environ["PREPROCESSED_DATA_URI"].rstrip("/")
dst = Path(os.environ["PROCESSED_DIR"])
bucket, _, prefix = src[len("gs://"):].partition("/")
n = 0
for blob in storage.Client().list_blobs(bucket, prefix=prefix + "/"):
    if blob.name.endswith("/"):
        continue
    out = dst / blob.name[len(prefix) + 1:]
    out.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(out))
    n += 1
print(f"Downloaded {n} files into {dst}")
PY

upload_checkpoints() {
  python scripts/gcs_data_ops.py upload-dir --src-dir "${SAVE_DIR}" --dst-prefix "${OUTPUT_URI}/checkpoints_ressrdiff" >/dev/null || true
}

# The subshell's pipefail makes the job fail when training fails, not only when tee does.
(
  set -o pipefail
  python -u src/train_ressrdiff.py \
    --code-dir "${CODE_DIR}" \
    --lr-train "${PROCESSED_DIR}/low_res/train" \
    --hr-train "${PROCESSED_DIR}/full_res/train" \
    --lr-test "${PROCESSED_DIR}/low_res/test" \
    --hr-test "${PROCESSED_DIR}/full_res/test" \
    --save-dir "${SAVE_DIR}" \
    ${TRAIN_RESSRDIFF_ARGS} 2>&1 | tee "${SAVE_DIR}/train.log"
) &
TRAIN_PID=$!
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep "${CKPT_UPLOAD_POLL_SECONDS}"
  upload_checkpoints
done
wait "${TRAIN_PID}"
upload_checkpoints
echo "Run complete"
