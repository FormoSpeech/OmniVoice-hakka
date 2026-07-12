#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Tokenize denoised + cln-restored Hakka reading WebDataset shards.

GPU_IDS="${GPU_IDS:-0,1,2,3}"
MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/data/formospeech_hakka_reading_denoised/data.lst}"
TOKEN_DIR="${TOKEN_DIR:-${REPO_ROOT}/data/formospeech_hakka_reading_denoised/tokens_cln}"
TOKENIZER_PATH="${TOKENIZER_PATH:-eustlb/higgs-audio-v2-tokenizer}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${TOKEN_DIR}/train"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
uv run python -m omnivoice.scripts.extract_audio_tokens \
  --input_manifest "${MANIFEST_PATH}" \
  --tar_output_pattern "${TOKEN_DIR}/train/audios/shard-%06d.tar" \
  --jsonl_output_pattern "${TOKEN_DIR}/train/txts/shard-%06d.jsonl" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --nj_per_gpu 3

echo "Done. Manifest written to ${TOKEN_DIR}/train/data.lst"
