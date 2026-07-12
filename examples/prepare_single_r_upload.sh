#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 SOURCE_REPO TARGET_NAME DENOISED_AUDIO_DIR OUTPUT_ROOT [CONFIG_NAME] [SPLIT]" >&2
  echo "Example: $0 formospeech/hat_asr_sixian_broadcast_clean hat_asr_sixian_broadcast_clean_r data/formospeech_hakka_sixian_broadcast_denoised/audio data/hakka_broadcast_r_upload Hakka_Sixian train" >&2
  exit 1
fi

SOURCE_REPO="$1"
TARGET_NAME="$2"
DENOISED_AUDIO_DIR="$3"
OUTPUT_ROOT="$4"
CONFIG_NAME="${5:-}"
SPLIT="${6:-train}"

ARGS=(
  --source-repo "${SOURCE_REPO}"
  --target-name "${TARGET_NAME}"
  --split "${SPLIT}"
  --extracted-audio-dir "${DENOISED_AUDIO_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --copy-audio
)

if [[ -n "${CONFIG_NAME}" ]]; then
  ARGS+=(--config-name "${CONFIG_NAME}")
fi

uv run --with datasets python "${REPO_ROOT}/tools/prepare_single_dataset_r_upload.py" "${ARGS[@]}"

echo "Done. Local upload directory written to ${OUTPUT_ROOT}/${TARGET_NAME}"
