#!/bin/bash

# Evaluate F5-TTS (formospeech/f5-tts-hita-finetune-v1) on the same corrected
# per-dialect FormoSpeech Hakka test list used for
# formospeech/omnivoice-hakka-community-1, reporting CER / SIM-o / UTMOS
# separately for Sixian and Hailu, in both g2p and gt_pinyin text modes.
#
# Stage 1: F5 inference, g2p mode (combined test list, single pass)
# Stage 2: F5 inference, gt_pinyin mode (combined test list, single pass)
# Stage 3-5: CER / SIM-o / UTMOS for g2p, once per dialect
# Stage 6-8: CER / SIM-o / UTMOS for gt_pinyin, once per dialect

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

stage=3
stop_stage=8

# ====== Modify as needed ======
EVAL_ROOT="/mnt/md1/user_wayne/omnivoice_eval"
TEST_LIST_COMBINED="${EVAL_ROOT}/data/test_list_combined.jsonl"
DIALECTS=(Hakka_Sixian Hakka_Hailu)
F5_G2P_RES_DIR="${EVAL_ROOT}/data/gen_audio_f5_g2p"
F5_GT_RES_DIR="${EVAL_ROOT}/data/gen_audio_f5_gt_pinyin"
DOWNLOAD_DIR="${REPO_ROOT}/download"
TTS_EVAL_MODEL_DIR="${DOWNLOAD_DIR}/tts_eval_models"

F5_CKPT_PATH="hf://formospeech/f5-tts-hita-finetune-v1/model_774996.safetensors"
F5_VOCAB_PATH="hf://formospeech/f5-tts-hita-finetune-v1/vocab.txt"
ASR_MODEL="formospeech/whisper-large-v2-taiwanese-hakka-v1"

NJ_PER_GPU=1
NFE_STEP=16
SPEED=1
CROSS_FADE_DURATION=0.15

CER_BATCH_SIZE=8
CER_NJ_PER_GPU=1
CER_MAX_SECONDS=30
SIM_NJ_PER_GPU=1
UTMOS_NJ_PER_GPU=1
# =================================

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: F5 inference, g2p mode"
    mkdir -p "${F5_G2P_RES_DIR}"
    "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/tools/f5_hakka_infer_batch.py" \
        --test-list "${TEST_LIST_COMBINED}" \
        --res-dir "${F5_G2P_RES_DIR}" \
        --mode g2p \
        --ckpt-path "${F5_CKPT_PATH}" \
        --vocab-path "${F5_VOCAB_PATH}" \
        --nfe-step "${NFE_STEP}" \
        --speed "${SPEED}" \
        --cross-fade-duration "${CROSS_FADE_DURATION}" \
        --nj-per-gpu "${NJ_PER_GPU}"
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: F5 inference, gt_pinyin mode"
    mkdir -p "${F5_GT_RES_DIR}"
    "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/tools/f5_hakka_infer_batch.py" \
        --test-list "${TEST_LIST_COMBINED}" \
        --res-dir "${F5_GT_RES_DIR}" \
        --mode gt_pinyin \
        --ckpt-path "${F5_CKPT_PATH}" \
        --vocab-path "${F5_VOCAB_PATH}" \
        --nfe-step "${NFE_STEP}" \
        --speed "${SPEED}" \
        --cross-fade-duration "${CROSS_FADE_DURATION}" \
        --nj-per-gpu "${NJ_PER_GPU}"
fi

score_dialects() {
    local res_dir="$1"
    for dialect in "${DIALECTS[@]}"; do
        local test_list="${EVAL_ROOT}/data/test_list_${dialect}.jsonl"
        uv run --extra eval python -m omnivoice.eval.wer.hakka \
            --wav-path "${res_dir}" \
            --test-list "${test_list}" \
            --decode-path "${res_dir}.hakka_cer.${dialect}.log" \
            --asr-model "${ASR_MODEL}" \
            --batch-size "${CER_BATCH_SIZE}" \
            --nj-per-gpu "${CER_NJ_PER_GPU}" \
            --max-seconds "${CER_MAX_SECONDS}"
    done
    for dialect in "${DIALECTS[@]}"; do
        local test_list="${EVAL_ROOT}/data/test_list_${dialect}.jsonl"
        uv run --extra eval python -m omnivoice.eval.speaker_similarity.sim \
            --wav-path "${res_dir}" \
            --test-list "${test_list}" \
            --decode-path "${res_dir}.sim.${dialect}.log" \
            --model-dir "${TTS_EVAL_MODEL_DIR}" \
            --nj-per-gpu "${SIM_NJ_PER_GPU}"
    done
    for dialect in "${DIALECTS[@]}"; do
        local test_list="${EVAL_ROOT}/data/test_list_${dialect}.jsonl"
        uv run --extra eval python -m omnivoice.eval.mos.utmos \
            --wav-path "${res_dir}" \
            --test-list "${test_list}" \
            --decode-path "${res_dir}.utmos.${dialect}.log" \
            --model-dir "${TTS_EVAL_MODEL_DIR}" \
            --nj-per-gpu "${UTMOS_NJ_PER_GPU}"
    done
}

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 5 ]; then
    echo "Stage 3-5: score g2p"
    score_dialects "${F5_G2P_RES_DIR}"
fi

if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 8 ]; then
    echo "Stage 6-8: score gt_pinyin"
    score_dialects "${F5_GT_RES_DIR}"
fi
