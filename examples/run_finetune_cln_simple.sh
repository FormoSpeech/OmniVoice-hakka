#!/bin/bash

# Fine-tune OmniVoice on the simple cleaned Hakka manifests generated from
# hanzi_cln/pinyin_cln where <UNK>/<spn> samples are filtered and <SIL> is
# replaced by commas.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

stage=0
stop_stage=1

# ====== Modify as needed ======
GPU_IDS="0,1,2,3"
NUM_GPUS=4

TRAIN_JSONL="${REPO_ROOT}/data/formospeech_hakka/train_cln_simple.jsonl"
DEV_JSONL="${REPO_ROOT}/data/formospeech_hakka/dev_cln_simple.jsonl"

TOKEN_DIR="${REPO_ROOT}/data/formospeech_hakka/tokens_cln_simple"
TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"

TRAIN_CONFIG="${REPO_ROOT}/examples/config/train_config_finetune.json"
data_config="${REPO_ROOT}/examples/config/data_config_finetune_cln_simple.json"

OUTPUT_DIR="${REPO_ROOT}/exp/omnivoice_finetune_formospeech_hakka_cln_simple"
# =================================

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Tokenizing simple cleaned manifests"

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

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Fine-tuning on simple cleaned manifests"

    uv run accelerate launch \
        --gpu_ids "${GPU_IDS}" \
        --num_processes ${NUM_GPUS} \
        -m omnivoice.cli.train \
        --train_config ${TRAIN_CONFIG} \
        --data_config ${data_config} \
        --output_dir ${OUTPUT_DIR}
fi
