#!/bin/bash

# Evaluate the r_mixed Hakka checkpoint on the custom FormoSpeech Hakka test list.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

stage=1
stop_stage=4

CHECKPOINT="${REPO_ROOT}/exp/omnivoice_finetune_formospeech_hakka_r_mixed_16000/checkpoint-16000"
TEST_LIST="${REPO_ROOT}/data/formospeech_hakka/dev_test_list.jsonl"
RES_DIR="${REPO_ROOT}/exp/eval_hakka_r_mixed_$(basename "${CHECKPOINT}")"
DOWNLOAD_DIR="${REPO_ROOT}/download"
TTS_EVAL_MODEL_DIR="${DOWNLOAD_DIR}/tts_eval_models"

export LD_LIBRARY_PATH="/home/wayne/ffmpeg-n7.1-latest-linux64-gpl-shared-7.1/lib:${LD_LIBRARY_PATH:-}"
ASR_MODEL="formospeech/whisper-large-v2-taiwanese-hakka-v1"

NJ_PER_GPU=1
INFER_OPTIONS="--preprocess_prompt False \
    --postprocess_output False \
    --batch_duration 600 \
    --audio_chunk_threshold 1000"

CER_BATCH_SIZE=8
CER_NJ_PER_GPU=1
CER_MAX_SECONDS=30
SIM_NJ_PER_GPU=1
UTMOS_NJ_PER_GPU=1

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    mkdir -p "${DOWNLOAD_DIR}" "${TTS_EVAL_MODEL_DIR}"

    uv run python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="k2-fsa/TTS_eval_models",
    local_dir="${TTS_EVAL_MODEL_DIR}",
    local_dir_use_symlinks=False,
)
PY
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    mkdir -p "${RES_DIR}"

    uv run omnivoice-infer-batch \
        --model "${CHECKPOINT}" \
        --test_list "${TEST_LIST}" \
        --res_dir "${RES_DIR}" \
        --nj_per_gpu "${NJ_PER_GPU}" \
        ${INFER_OPTIONS}
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    uv run --extra eval python -m omnivoice.eval.wer.hakka \
        --wav-path "${RES_DIR}" \
        --test-list "${TEST_LIST}" \
        --decode-path "${RES_DIR}.hakka_cer.log" \
        --asr-model "${ASR_MODEL}" \
        --batch-size "${CER_BATCH_SIZE}" \
        --nj-per-gpu "${CER_NJ_PER_GPU}" \
        --max-seconds "${CER_MAX_SECONDS}"
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    uv run --extra eval python -m omnivoice.eval.speaker_similarity.sim \
        --wav-path "${RES_DIR}" \
        --test-list "${TEST_LIST}" \
        --decode-path "${RES_DIR}.sim.log" \
        --model-dir "${TTS_EVAL_MODEL_DIR}" \
        --nj-per-gpu "${SIM_NJ_PER_GPU}"
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    uv run --extra eval python -m omnivoice.eval.mos.utmos \
        --wav-path "${RES_DIR}" \
        --test-list "${TEST_LIST}" \
        --decode-path "${RES_DIR}.utmos.log" \
        --model-dir "${TTS_EVAL_MODEL_DIR}" \
        --nj-per-gpu "${UTMOS_NJ_PER_GPU}"
fi
