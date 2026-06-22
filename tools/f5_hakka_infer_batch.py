#!/usr/bin/env python3
"""Batch inference for Taiwanese Hakka F5-TTS evaluation.

Supports two text modes:
    - g2p: convert Hanzi to pinyin with formog2p
    - gt_pinyin: use ground-truth pinyin from the manifest/test list
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
import torch
from cached_path import cached_path
from formog2p.hakka import g2p
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

from f5_tts.infer.utils_infer import (
    chunk_text,
    hop_length,
    infer_batch_process,
    infer_process,
    load_checkpoint,
    load_vocoder,
    mel_spec_type,
    n_fft,
    n_mel_channels,
    ode_method,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    target_sample_rate,
    win_length,
)
from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer
from omnivoice.utils.duration import RuleDurationEstimator
from omnivoice.utils.audio import load_audio

DIALECT_CODE_MAP = {
    "Hakka_Hailu": "hak_hl",
    "Hakka_Sixian": "hak_sx",
}


def normalize_g2p_input(text: str) -> str:
    text = text.replace("酚", "分")
    text = text.replace("COVID-19", "COVID nineteen")
    return text


worker_model = None
worker_vocoder = None
worker_device = None
worker_args = None
worker_duration_estimator = None


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run batch inference for Taiwanese Hakka F5-TTS."
    )
    parser.add_argument("--test-list", type=str, required=True)
    parser.add_argument("--res-dir", type=str, required=True)
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional manifest used to resolve ref_id -> text_pinyin for gt_pinyin mode.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["g2p", "gt_pinyin"],
        required=True,
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="hf://formospeech/f5-tts-hita-finetune-v1/model_774996.safetensors",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="hf://formospeech/f5-tts-hita-finetune-v1/vocab.txt",
    )
    parser.add_argument("--extension", type=str, default="wav")
    parser.add_argument("--nfe-step", type=int, default=32)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--cross-fade-duration", type=float, default=0.15)
    parser.add_argument("--remove-silence", action="store_true")
    parser.add_argument("--nj-per-gpu", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def normalize_pinyin(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[！？!?]", "。", text)
    text = re.sub(r"\s*([，。,\.])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def estimate_chunk_durations(gen_text_batches, total_duration):
    if total_duration is None or len(gen_text_batches) <= 1:
        return [total_duration] * len(gen_text_batches)

    weights = [max(1, len(chunk.encode("utf-8"))) for chunk in gen_text_batches]
    total_weight = sum(weights)
    durations = [total_duration * w / total_weight for w in weights]

    diff = total_duration - sum(durations)
    durations[-1] += diff
    return durations


def combine_waves_with_crossfade(generated_waves, cross_fade_duration, sample_rate):
    if not generated_waves:
        return None
    if len(generated_waves) == 1 or cross_fade_duration <= 0:
        return generated_waves[0] if len(generated_waves) == 1 else __import__("numpy").concatenate(generated_waves)

    import numpy as np

    final_wave = generated_waves[0]
    for next_wave in generated_waves[1:]:
        cross_fade_samples = int(cross_fade_duration * sample_rate)
        cross_fade_samples = min(cross_fade_samples, len(final_wave), len(next_wave))
        if cross_fade_samples <= 0:
            final_wave = np.concatenate([final_wave, next_wave])
            continue

        prev_overlap = final_wave[-cross_fade_samples:]
        next_overlap = next_wave[:cross_fade_samples]
        fade_out = np.linspace(1, 0, cross_fade_samples)
        fade_in = np.linspace(0, 1, cross_fade_samples)
        cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
        final_wave = np.concatenate(
            [
                final_wave[:-cross_fade_samples],
                cross_faded_overlap,
                next_wave[cross_fade_samples:],
            ]
        )
    return final_wave


def to_pinyin(text: str, dialect_code: str) -> str:
    text = normalize_g2p_input(text)
    pattern = r"([a-zA-Z]+(?:\s+[a-zA-Z]+)*)"
    parts = re.split(pattern, text)
    result_parts = []
    for part in parts:
        if not part:
            continue
        if re.match(r"^[a-zA-Z]+(?:\s+[a-zA-Z]+)*$", part):
            result_parts.append(part)
            continue

        result = g2p(part, dialect_code, pronunciation_type="pinyin")
        if result.unknown_words:
            raise ValueError(
                f"Unknown words for dialect {dialect_code}: {','.join(result.unknown_words)}"
            )
        result_parts.append(" ".join(result.pronunciations))
    return normalize_pinyin(" ".join(result_parts))


def detect_dialect_from_id(sample_id: str) -> str:
    parts = sample_id.split("__")
    if len(parts) < 3:
        raise ValueError(f"Unexpected sample id format: {sample_id}")
    dialect_key = parts[2]
    if dialect_key not in DIALECT_CODE_MAP:
        raise ValueError(f"Unsupported dialect in sample id: {sample_id}")
    return DIALECT_CODE_MAP[dialect_key]


def load_jsonl(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_manifest_index(manifest_path: str):
    if not manifest_path:
        return {}
    return {row["id"]: row for row in load_jsonl(manifest_path)}


def load_model(model_cls, model_cfg, ckpt_path, vocab_file, target_device, fp16=False):
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(method=ode_method),
        vocab_char_map=vocab_char_map,
    ).to(target_device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" or not fp16 else None
    return load_checkpoint(
        model, ckpt_path, str(target_device), dtype=dtype, use_ema=False
    )


def worker_init(rank_queue, args_dict):
    global worker_model, worker_vocoder, worker_device, worker_args
    global worker_duration_estimator
    torch.set_num_threads(2)

    rank = rank_queue.get()
    worker_device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    worker_args = args_dict

    ckpt_path = str(cached_path(args_dict["ckpt_path"]))
    vocab_path = str(cached_path(args_dict["vocab_path"]))
    model_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
    )
    worker_model = load_model(DiT, model_cfg, ckpt_path, vocab_path, worker_device)
    worker_vocoder = load_vocoder()
    worker_duration_estimator = RuleDurationEstimator()


def prepare_texts(sample, manifest_index, mode):
    if mode == "g2p":
        ref_dialect = detect_dialect_from_id(sample["ref_id"])
        gen_dialect = detect_dialect_from_id(sample["id"])
        ref_text = to_pinyin(sample["ref_text"], ref_dialect)
        gen_text = to_pinyin(sample["text"], gen_dialect)
        return ref_text, gen_text

    if not sample.get("text_pinyin"):
        raise ValueError(f"Missing text_pinyin for sample {sample['id']}")
    ref_id = sample.get("ref_id")
    if not ref_id:
        raise ValueError(f"Missing ref_id for sample {sample['id']}")
    ref_sample = manifest_index.get(ref_id)
    if ref_sample is None:
        raise ValueError(f"ref_id {ref_id} not found in manifest")
    ref_text_pinyin = ref_sample.get("text_pinyin")
    if not ref_text_pinyin:
        raise ValueError(f"Missing text_pinyin for ref_id {ref_id}")

    ref_text = normalize_pinyin(ref_text_pinyin.lower())
    gen_text = normalize_pinyin(sample["text_pinyin"].lower())
    return ref_text, gen_text


def run_one(sample, manifest_index):
    global worker_model, worker_vocoder, worker_args, worker_duration_estimator
    try:
        raw_ref_audio = load_audio(sample["ref_audio"], target_sample_rate)
        raw_ref_duration = raw_ref_audio.shape[-1] / target_sample_rate

        ref_audio, _ = preprocess_ref_audio_text(
            sample["ref_audio"], sample["ref_text"], show_info=lambda *args, **kwargs: None
        )
        ref_text, gen_text = prepare_texts(sample, manifest_index, worker_args["mode"])

        processed_ref_duration = AudioDecoder(ref_audio).metadata.duration_seconds_from_header
        target_duration = worker_duration_estimator.estimate_duration(
            sample["text"],
            sample["ref_text"],
            raw_ref_duration,
            low_threshold=2.0,
        )
        if worker_args["speed"] > 0 and worker_args["speed"] != 1.0:
            target_duration = target_duration / worker_args["speed"]

        audio, sr = __import__("torchaudio").load(ref_audio)
        max_chars = int(
            len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * worker_args["speed"]
        )
        gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
        chunk_durations = estimate_chunk_durations(gen_text_batches, target_duration)
        chunk_waves = []
        chunk_specs = []
        for gen_text_chunk, chunk_duration in zip(gen_text_batches, chunk_durations):
            result = next(
                infer_batch_process(
                    __import__("torchaudio").load(ref_audio),
                    ref_text,
                    [gen_text_chunk],
                    worker_model,
                    worker_vocoder,
                    progress=None,
                    cross_fade_duration=0.0,
                    nfe_step=worker_args["nfe_step"],
                    speed=worker_args["speed"],
                    fix_duration=chunk_duration + processed_ref_duration,
                    device=worker_device,
                )
            )
            generated_wave, final_sample_rate, generated_spec = result
            chunk_waves.append(generated_wave)
            chunk_specs.append(generated_spec)

        final_wave = combine_waves_with_crossfade(
            chunk_waves, worker_args["cross_fade_duration"], target_sample_rate
        )
        import numpy as np
        combined_spectrogram = np.concatenate(chunk_specs, axis=1) if chunk_specs else None
        final_sample_rate = target_sample_rate

        out_wav = Path(worker_args["res_dir"]) / f"{sample['id']}.{worker_args['extension']}"
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_wav, final_wave, final_sample_rate)

        if worker_args["remove_silence"]:
            remove_silence_for_generated_wav(str(out_wav))

        out_meta = Path(worker_args["res_dir"]) / f"{sample['id']}.json"
        with out_meta.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "id": sample["id"],
                    "mode": worker_args["mode"],
                    "duration_estimation_text": sample["text"],
                    "duration_estimation_ref_text": sample["ref_text"],
                    "ref_text_input": ref_text,
                    "gen_text_input": gen_text,
                    "ref_audio": sample["ref_audio"],
                    "raw_ref_duration": raw_ref_duration,
                    "processed_ref_duration": processed_ref_duration,
                    "target_duration": target_duration,
                    "chunk_durations": chunk_durations,
                    "speed": worker_args["speed"],
                },
                f,
                ensure_ascii=False,
            )

        out_spec = Path(worker_args["res_dir"]) / f"{sample['id']}.png"
        save_spectrogram(combined_spectrogram, str(out_spec))
        return sample["id"], "ok", None
    except Exception as e:
        return sample["id"], "error", f"{e}\n{traceback.format_exc()}"


def main():
    args = get_parser().parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )

    samples = load_jsonl(args.test_list)
    if args.limit > 0:
        samples = samples[: args.limit]
    manifest_index = build_manifest_index(args.manifest)

    os.makedirs(args.res_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPU found. GPU is required."
    total_workers = num_gpus * args.nj_per_gpu
    logging.info(
        "Running F5 batch inference with %s workers on %s GPUs in %s mode.",
        total_workers,
        num_gpus,
        args.mode,
    )

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    rank_queue = manager.Queue()
    for rank in list(range(num_gpus)) * args.nj_per_gpu:
        rank_queue.put(rank)

    args_dict = vars(args).copy()
    failures = []
    with ProcessPoolExecutor(
        max_workers=total_workers,
        initializer=worker_init,
        initargs=(rank_queue, args_dict),
    ) as executor:
        futures = [
            executor.submit(run_one, sample, manifest_index) for sample in samples
        ]
        pbar = tqdm(as_completed(futures), total=len(futures), desc="F5 infer")
        for future in pbar:
            sample_id, status, error = future.result()
            if status != "ok":
                failures.append((sample_id, error))
                logging.error("Failed sample %s: %s", sample_id, error)

    if failures:
        err_path = Path(args.res_dir) / "failures.log"
        with err_path.open("w", encoding="utf-8") as f:
            for sample_id, error in failures:
                f.write(f"[{sample_id}]\n{error}\n\n")
        logging.warning("Finished with %s failures. See %s", len(failures), err_path)
    else:
        logging.info("Finished successfully: %s samples", len(samples))


if __name__ == "__main__":
    main()
