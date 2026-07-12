#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 OUTPUT_DIR DATASET_REPO [DATASET_REPO ...]" >&2
  echo "Example: $0 data/formospeech_hakka_sixian_broadcast formospeech/hat_asr_sixian_broadcast_clean" >&2
  exit 1
fi

OUTPUT_DIR="$1"
shift

uv run --with datasets python "${REPO_ROOT}/tools/export_formospeech_hakka_jsonl.py" \
  --output-dir "${OUTPUT_DIR}" \
  --datasets "$@"

echo "Done. Exported datasets to ${OUTPUT_DIR}"
