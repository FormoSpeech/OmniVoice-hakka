#!/bin/bash

# Evaluate Taiwanese Hakka F5-TTS on the custom FormoSpeech Hakka test list.
#
# Stage 0: Download evaluation models
# Stage 1: F5 inference with Hanzi -> G2P
# Stage 2: CER / SIM-o / UTMOS for Hanzi -> G2P
# Stage 3: F5 inference with GT pinyin
# Stage 4: CER / SIM-o / UTMOS for GT pinyin

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

stage=0
stop_stage=4

TEST_LIST="${REPO_ROOT}/data/formospeech_hakka/dev_test_list.jsonl"
MANIFEST="${REPO_ROOT}/data/formospeech_hakka/dev.jsonl"
DOWNLOAD_DIR="${REPO_ROOT}/download"
TTS_EVAL_MODEL_DIR="${DOWNLOAD_DIR}/tts_eval_models"

F5_CKPT_PATH="hf://formospeech/f5-tts-hita-finetune-v1/model_774996.safetensors"
F5_VOCAB_PATH="hf://formospeech/f5-tts-hita-finetune-v1/vocab.txt"

F5_G2P_RES_DIR="${REPO_ROOT}/exp/eval_f5_hakka_g2p"
F5_GT_RES_DIR="${REPO_ROOT}/exp/eval_f5_hakka_gt_pinyin"

NJ_PER_GPU=1
NFE_STEP=16
SPEED=1
CROSS_FADE_DURATION=0.15

CER_BATCH_SIZE=8
CER_NJ_PER_GPU=1
CER_MAX_SECONDS=30
SIM_NJ_PER_GPU=1
UTMOS_NJ_PER_GPU=1

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/wayne/ffmpeg-n7.1-latest-linux64-gpl-shared-7.1/lib:${LD_LIBRARY_PATH:-}"

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Install F5 dependencies and download evaluation models"

    uv pip install --python "${REPO_ROOT}/.venv/bin/python" \
        formog2p cached-path soundfile rjieba matplotlib librosa pydub \
        torchdiffeq transformers_stream_generator x_transformers vocos jieba \
        pypinyin ema_pytorch datasets accelerate wandb tomli torchaudio
    uv pip install --python "${REPO_ROOT}/.venv/bin/python" --no-deps f5-tts

    mkdir -p "${DOWNLOAD_DIR}" "${TTS_EVAL_MODEL_DIR}"

    uv run --no-sync python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="k2-fsa/TTS_eval_models",
    local_dir="${TTS_EVAL_MODEL_DIR}",
)
PY
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: F5 inference with Hanzi -> G2P"

    uv run --no-sync python "${REPO_ROOT}/tools/f5_hakka_infer_batch.py" \
        --test-list "${TEST_LIST}" \
        --manifest "${MANIFEST}" \
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
    echo "Stage 2: Evaluate Hanzi -> G2P"

    uv run --extra eval python -m omnivoice.eval.wer.hakka \
        --wav-path "${F5_G2P_RES_DIR}" \
        --test-list "${TEST_LIST}" \
        --decode-path "${F5_G2P_RES_DIR}.hakka_cer.log" \
        --asr-model "formospeech/whisper-large-v2-taiwanese-hakka-v1" \
        --batch-size "${CER_BATCH_SIZE}" \
        --nj-per-gpu "${CER_NJ_PER_GPU}" \
        --max-seconds "${CER_MAX_SECONDS}"

    uv run --extra eval python -m omnivoice.eval.speaker_similarity.sim \
        --wav-path "${F5_G2P_RES_DIR}" \
        --test-list "${TEST_LIST}" \
        --decode-path "${F5_G2P_RES_DIR}.sim.log" \
        --model-dir "${TTS_EVAL_MODEL_DIR}" \
        --nj-per-gpu "${SIM_NJ_PER_GPU}"

    uv run --extra eval python -m omnivoice.eval.mos.utmos \
        --wav-path "${F5_G2P_RES_DIR}" \
        --test-list "${TEST_LIST}" \
        --decode-path "${F5_G2P_RES_DIR}.utmos.log" \
        --model-dir "${TTS_EVAL_MODEL_DIR}" \
        --nj-per-gpu "${UTMOS_NJ_PER_GPU}"
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: F5 inference with GT pinyin"

    uv run --no-sync python "${REPO_ROOT}/tools/f5_hakka_infer_batch.py" \
        --test-list "${TEST_LIST}" \
        --manifest "${MANIFEST}" \
        --res-dir "${F5_GT_RES_DIR}" \
        --mode gt_pinyin \
        --ckpt-path "${F5_CKPT_PATH}" \
        --vocab-path "${F5_VOCAB_PATH}" \
        --nfe-step "${NFE_STEP}" \
        --speed "${SPEED}" \
        --cross-fade-duration "${CROSS_FADE_DURATION}" \
        --nj-per-gpu "${NJ_PER_GPU}"
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Evaluate GT pinyin"

    uv run --extra eval python -m omnivoice.eval.wer.hakka \
        --wav-path "${F5_GT_RES_DIR}" \
        --test-list "${TEST_LIST}" \
        --decode-path "${F5_GT_RES_DIR}.hakka_cer.log" \
        --asr-model "formospeech/whisper-large-v2-taiwanese-hakka-v1" \
        --batch-size "${CER_BATCH_SIZE}" \
        --nj-per-gpu "${CER_NJ_PER_GPU}" \
        --max-seconds "${CER_MAX_SECONDS}"

    uv run --extra eval python -m omnivoice.eval.speaker_similarity.sim \
        --wav-path "${F5_GT_RES_DIR}" \
        --test-list "${TEST_LIST}" \
        --decode-path "${F5_GT_RES_DIR}.sim.log" \
        --model-dir "${TTS_EVAL_MODEL_DIR}" \
        --nj-per-gpu "${SIM_NJ_PER_GPU}"

    uv run --extra eval python -m omnivoice.eval.mos.utmos \
        --wav-path "${F5_GT_RES_DIR}" \
        --test-list "${TEST_LIST}" \
        --decode-path "${F5_GT_RES_DIR}.utmos.log" \
        --model-dir "${TTS_EVAL_MODEL_DIR}" \
        --nj-per-gpu "${UTMOS_NJ_PER_GPU}"
fi
