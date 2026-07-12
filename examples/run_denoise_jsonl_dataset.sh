#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INPUT_JSONL OUTPUT_DIR" >&2
  echo "Example: $0 data/formospeech_hakka_sixian_broadcast/train.jsonl data/formospeech_hakka_sixian_broadcast_denoised" >&2
  exit 1
fi

INPUT_JSONL="$1"
OUTPUT_DIR="$2"
FEATURE_EXTRACTOR_PATH="${FEATURE_EXTRACTOR_PATH:-${REPO_ROOT}/third_party/sidon-v0.1/feature_extractor_cuda.pt}"
DECODER_PATH="${DECODER_PATH:-${REPO_ROOT}/third_party/sidon-v0.1/decoder_cuda.pt}"

uv run python "${REPO_ROOT}/tools/denoise_jsonl_to_audio_dir.py" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --feature_extractor_path "${FEATURE_EXTRACTOR_PATH}" \
  --decoder_path "${DECODER_PATH}" \
  --target_sample_rate 24000 \
  --batch_duration 200.0 \
  --skip_errors

echo "Done. Denoised output written to ${OUTPUT_DIR}"
