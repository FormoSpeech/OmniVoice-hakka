#!/bin/bash

# Export the Hakka R-mixed corpus:
# - keep original radio / TTS / elearning datasets
# - replace reading / broadcast subsets with uploaded -R variants

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/data/formospeech_hakka_r_mixed"

RAW_DATASETS=(
  "formospeech/hakkaradio_news_clean"
  "formospeech/hat_tts_hailu_clean"
  "formospeech/hat_tts_sixian_clean"
  "formospeech/hakka_elearning_example_clean"
  "formospeech/hat_asr_sixian_reading_clean_r"
  "formospeech/hat_asr_hailu_reading_clean_r"
  "formospeech/hat_asr_nansixian_reading_clean_r"
  "formospeech/hat_asr_sixian_broadcast_clean_r"
)

uv run --with datasets python "${REPO_ROOT}/tools/export_formospeech_hakka_jsonl.py" \
  --output-dir "${OUTPUT_DIR}" \
  --datasets "${RAW_DATASETS[@]}" \
  --train-jsonl train.jsonl \
  --dev-jsonl dev.jsonl

uv run --with datasets python "${REPO_ROOT}/tools/export_formospeech_hakka_jsonl.py" \
  --output-dir "${OUTPUT_DIR}" \
  --datasets "${RAW_DATASETS[@]}" \
  --train-jsonl train_cln.jsonl \
  --dev-jsonl dev_cln.jsonl \
  --text-variant cln \
  --cln-mode restored

echo "Done. Exported r_mixed manifests to ${OUTPUT_DIR}"
