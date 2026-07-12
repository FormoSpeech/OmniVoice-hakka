#!/bin/bash

# Fine-tune OmniVoice on the cleaned Hakka manifests generated from
# hanzi_cln/pinyin_cln with local punctuation recovery.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

stage=0
stop_stage=1

# ====== Modify as needed ======
# GPUs to use
GPU_IDS="0,1,2,3"
NUM_GPUS=4

# Path to the cleaned input JSONL files
TRAIN_JSONL="${REPO_ROOT}/data/formospeech_hakka/train_cln.jsonl"

# Path to your dev JSONL file. Set to empty string to skip dev set.
DEV_JSONL="${REPO_ROOT}/data/formospeech_hakka/dev_cln.jsonl"

# Directory to write tokenized WebDataset shards
TOKEN_DIR="${REPO_ROOT}/data/formospeech_hakka/tokens_cln"

# Audio tokenizer model (HuggingFace repo or local path)
TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"

# Training config file
TRAIN_CONFIG="${REPO_ROOT}/examples/config/train_config_finetune.json"

# Data config file
data_config="${REPO_ROOT}/examples/config/data_config_finetune_cln.json"

# Output directory for fine-tuned checkpoints
OUTPUT_DIR="${REPO_ROOT}/exp/omnivoice_finetune_formospeech_hakka_cln"
# =================================

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Stage 0: Tokenize audio into WebDataset shards
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Tokenizing cleaned manifests"

    for split_jsonl_path in ${TRAIN_JSONL} ${DEV_JSONL}; do
        if [ -z "${split_jsonl_path}" ]; then
            continue
        fi

        if [ "${split_jsonl_path}" = "${TRAIN_JSONL}" ]; then
            split="train"
        else
            split="dev"
        fi

        echo "  Tokenizing ${split} from ${split_jsonl_path}"

        CUDA_VISIBLE_DEVICES=${GPU_IDS} \
            uv run python -m omnivoice.scripts.extract_audio_tokens \
            --input_jsonl "${split_jsonl_path}" \
            --tar_output_pattern "${TOKEN_DIR}/${split}/audios/shard-%06d.tar" \
            --jsonl_output_pattern "${TOKEN_DIR}/${split}/txts/shard-%06d.jsonl" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --nj_per_gpu 3 \
            --shuffle True

        echo "  Done. Manifest written to ${TOKEN_DIR}/${split}/data.lst"
    done
fi


# Stage 1: Fine-tune
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Fine-tuning on cleaned manifests"

    uv run accelerate launch \
        --gpu_ids "${GPU_IDS}" \
        --num_processes ${NUM_GPUS} \
        -m omnivoice.cli.train \
        --train_config ${TRAIN_CONFIG} \
        --data_config ${data_config} \
        --output_dir ${OUTPUT_DIR}
fi
