#!/bin/bash

# Evaluate the released formospeech/omnivoice-hakka-community-1 checkpoint on
# the FormoSpeech Hakka test list, reporting CER / SIM-o / UTMOS separately
# for the Sixian and Hailu dialect subsets (instead of one number mixing both).
#
# Stage 0: Download evaluation models
# Stage 1: Batch inference on the combined test list (single pass)
# Stage 2-4: CER / SIM-o / UTMOS, run once per dialect subset

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

stage=2
stop_stage=4

# ====== Modify as needed ======
CHECKPOINT="formospeech/omnivoice-hakka-community-1"
EVAL_ROOT="/mnt/md1/user_wayne/omnivoice_eval"
TEST_LIST_COMBINED="${EVAL_ROOT}/data/test_list_combined.jsonl"
DIALECTS=(Hakka_Sixian Hakka_Hailu)
RES_DIR="${EVAL_ROOT}/data/gen_audio"
DOWNLOAD_DIR="${REPO_ROOT}/download"
TTS_EVAL_MODEL_DIR="${DOWNLOAD_DIR}/tts_eval_models"

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
# =================================

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Download evaluation models"

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
    echo "Stage 1: Batch inference on combined Hakka test list"

    mkdir -p "${RES_DIR}"

    uv run omnivoice-infer-batch \
        --model "${CHECKPOINT}" \
        --test_list "${TEST_LIST_COMBINED}" \
        --res_dir "${RES_DIR}" \
        --nj_per_gpu "${NJ_PER_GPU}" \
        ${INFER_OPTIONS}
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Hakka CER evaluation (per dialect)"

    for dialect in "${DIALECTS[@]}"; do
        test_list="${EVAL_ROOT}/data/test_list_${dialect}.jsonl"
        uv run --extra eval python -m omnivoice.eval.wer.hakka \
            --wav-path "${RES_DIR}" \
            --test-list "${test_list}" \
            --decode-path "${RES_DIR}.hakka_cer.${dialect}.log" \
            --asr-model "${ASR_MODEL}" \
            --batch-size "${CER_BATCH_SIZE}" \
            --nj-per-gpu "${CER_NJ_PER_GPU}" \
            --max-seconds "${CER_MAX_SECONDS}"
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Speaker similarity (SIM-o, per dialect)"

    for dialect in "${DIALECTS[@]}"; do
        test_list="${EVAL_ROOT}/data/test_list_${dialect}.jsonl"
        uv run --extra eval python -m omnivoice.eval.speaker_similarity.sim \
            --wav-path "${RES_DIR}" \
            --test-list "${test_list}" \
            --decode-path "${RES_DIR}.sim.${dialect}.log" \
            --model-dir "${TTS_EVAL_MODEL_DIR}" \
            --nj-per-gpu "${SIM_NJ_PER_GPU}"
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: UTMOS (per dialect)"

    for dialect in "${DIALECTS[@]}"; do
        test_list="${EVAL_ROOT}/data/test_list_${dialect}.jsonl"
        uv run --extra eval python -m omnivoice.eval.mos.utmos \
            --wav-path "${RES_DIR}" \
            --test-list "${test_list}" \
            --decode-path "${RES_DIR}.utmos.${dialect}.log" \
            --model-dir "${TTS_EVAL_MODEL_DIR}" \
            --nj-per-gpu "${UTMOS_NJ_PER_GPU}"
    done
fi
