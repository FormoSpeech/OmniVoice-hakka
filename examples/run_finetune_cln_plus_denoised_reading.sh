#!/bin/bash

# Fine-tune OmniVoice on the cleaned Hakka manifests plus denoised+cln reading data.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

stage=1
stop_stage=1

# ====== Modify as needed ======
GPU_IDS="0,1,2,3"
NUM_GPUS=4

MAIN_TOKEN_DIR="${REPO_ROOT}/data/formospeech_hakka/tokens_cln"
READING_MANIFEST="${REPO_ROOT}/data/formospeech_hakka_reading_denoised/data.lst"
READING_TOKEN_DIR="${REPO_ROOT}/data/formospeech_hakka_reading_denoised/tokens_cln"

TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"
TRAIN_CONFIG="${REPO_ROOT}/examples/config/train_config_finetune.json"
DATA_CONFIG="${REPO_ROOT}/examples/config/data_config_finetune_cln_plus_denoised_reading.json"
OUTPUT_DIR="${REPO_ROOT}/exp/omnivoice_finetune_formospeech_hakka_cln_plus_denoised_reading"
# =================================

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Stage 0: Tokenize denoised+cln reading data
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Tokenizing denoised + cln reading manifests"

    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
        uv run python -m omnivoice.scripts.extract_audio_tokens \
        --input_manifest "${READING_MANIFEST}" \
        --tar_output_pattern "${READING_TOKEN_DIR}/train/audios/shard-%06d.tar" \
        --jsonl_output_pattern "${READING_TOKEN_DIR}/train/txts/shard-%06d.jsonl" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --nj_per_gpu 3

    echo "  Done. Manifest written to ${READING_TOKEN_DIR}/train/data.lst"
fi

# Stage 1: Fine-tune
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Fine-tuning on cln manifests + denoised reading data"
    echo "  Main cln tokens: ${MAIN_TOKEN_DIR}"
    echo "  Reading denoised tokens: ${READING_TOKEN_DIR}"

    uv run accelerate launch \
        --gpu_ids "${GPU_IDS}" \
        --num_processes ${NUM_GPUS} \
        -m omnivoice.cli.train \
        --train_config "${TRAIN_CONFIG}" \
        --data_config "${DATA_CONFIG}" \
        --output_dir "${OUTPUT_DIR}"
fi
