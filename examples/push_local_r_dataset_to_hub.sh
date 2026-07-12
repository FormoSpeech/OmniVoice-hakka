#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 INPUT_DIR REPO_ID CONFIG_NAME [SPLIT]" >&2
  echo "Example: $0 data/hakka_broadcast_r_upload/hat_asr_sixian_broadcast_clean_r formospeech/hat_asr_sixian_broadcast_clean_r Hakka_Sixian train" >&2
  exit 1
fi

INPUT_DIR="$1"
REPO_ID="$2"
CONFIG_NAME="$3"
SPLIT="${4:-train}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-500MB}"
NUM_SHARDS="${NUM_SHARDS:-}"
NUM_PROC="${NUM_PROC:-}"

ARGS=(
  --input-dir "${INPUT_DIR}"
  --repo-id "${REPO_ID}"
  --config-name "${CONFIG_NAME}"
  --split "${SPLIT}"
  --max-shard-size "${MAX_SHARD_SIZE}"
)

if [[ -n "${NUM_SHARDS}" ]]; then
  ARGS+=(--num-shards "${NUM_SHARDS}")
fi

if [[ -n "${NUM_PROC}" ]]; then
  ARGS+=(--num-proc "${NUM_PROC}")
fi

uv run --with datasets python "${REPO_ROOT}/tools/push_local_r_dataset_to_hub.py" "${ARGS[@]}"

echo "Done. Pushed ${INPUT_DIR} to ${REPO_ID}"
