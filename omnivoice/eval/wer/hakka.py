#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  OpenAI adaptation)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute character error rate (CER) for Hakka using a Whisper ASR model.

This script is intended for custom Hakka evaluation with JSONL test lists that
follow the repo's broad "test_list" convention:
    id, text, ref_audio, ref_text, language_id, language_name, ...

Only ``id`` and ``text`` are required for CER evaluation.
"""

import argparse
import logging
import multiprocessing as mp
import os
import string
import traceback
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from zhon.hanzi import punctuation

from omnivoice.eval.utils import load_eval_waveform
from omnivoice.eval.wer.common import process_one
from omnivoice.utils.data_utils import read_test_list

worker_pipe = None
worker_device = None


def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes CER for Hakka with a Whisper ASR model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--wav-path",
        type=str,
        default=None,
        help="Directory containing generated wav files named <id>.wav. "
        "If omitted, the script falls back to each sample's audio_path field.",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        required=True,
        help="JSONL test list. Each line should include at least id and text.",
    )
    parser.add_argument(
        "--decode-path",
        type=str,
        default=None,
        help="Optional output log path for detailed CER results.",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="formospeech/whisper-large-v2-taiwanese-hakka-v1",
        help="Hugging Face repo id or local path of the Hakka ASR model.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="wav",
        help="Generated audio extension.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for the Hugging Face ASR pipeline.",
    )
    parser.add_argument(
        "--nj-per-gpu",
        type=int,
        default=1,
        help="Number of workers per GPU.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help="Truncate audio longer than this many seconds before ASR.",
    )
    return parser


def post_process(text: str) -> str:
    """Normalize Hakka Hanzi text and return space-separated characters."""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()

    punctuation_all = punctuation + string.punctuation
    for ch in punctuation_all:
        text = text.replace(ch, "")

    text = "".join(text.split())
    return " ".join(list(text))


def load_whisper_model(model_name_or_path, device):
    import transformers

    transformers.logging.set_verbosity_error()
    return transformers.pipeline(
        "automatic-speech-recognition",
        model=model_name_or_path,
        dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        device=device,
    )


def process_init(rank_queue, asr_model):
    global worker_pipe, worker_device

    torch.set_num_threads(2)

    try:
        rank = rank_queue.get(timeout=10)
    except Exception:
        raise RuntimeError("Failed to get GPU rank from queue.")

    assert torch.cuda.is_available(), "CUDA is required but not available."
    worker_device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    logging.info(f"Initializing worker on device: {worker_device}")
    worker_pipe = load_whisper_model(asr_model, worker_device)


def run_eval_worker(data_chunk, batch_size, max_seconds):
    global worker_pipe

    if worker_pipe is None:
        logging.error("Worker pipeline is not initialized!")
        return []

    metrics_buffer = []
    try:
        dataset = [
            {
                "array": load_eval_waveform(
                    item["wav_path"],
                    sample_rate=16000,
                    return_numpy=True,
                    max_seconds=max_seconds,
                ),
                "sampling_rate": 16000,
            }
            for item in data_chunk
        ]

        iterator = worker_pipe(dataset, batch_size=batch_size)
        for i, out in enumerate(iterator):
            hypothesis = out["text"].strip()
            ref_item = data_chunk[i]
            truth = ref_item["truth_text"]
            wav_path = ref_item["wav_path"]

            m = process_one(hypothesis, truth, post_process)
            m["wav_path"] = wav_path
            metrics_buffer.append(m)

    except Exception:
        logging.error(f"Worker failed on chunk:\n{traceback.format_exc()}")
        return []

    return metrics_buffer


def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )

    logging.info(f"Calculating CER for {args.wav_path}")
    logging.info(f"ASR model: {args.asr_model}")

    data_list = []
    samples = read_test_list(args.test_list)
    for s in samples:
        if args.wav_path:
            wav_full_path = str(Path(args.wav_path) / f"{s['id']}.{args.extension}")
        else:
            wav_full_path = s.get("audio_path")
        if not os.path.exists(wav_full_path):
            logging.warning(f"File missing: {wav_full_path}")
            continue
        data_list.append({"wav_path": wav_full_path, "truth_text": s["text"]})

    total_files = len(data_list)
    logging.info(f"Total files: {total_files}.")

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPU found. GPU is required."
    total_workers = num_gpus * args.nj_per_gpu

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    rank_queue = manager.Queue()

    for _ in range(args.nj_per_gpu):
        for rank in range(num_gpus):
            rank_queue.put(rank)

    chunk_size = max(1, args.batch_size)
    tasks = [data_list[i : i + chunk_size] for i in range(0, total_files, chunk_size)]

    logging.info(
        f"Split data into {len(tasks)} chunks (size ~{chunk_size}). "
        f"Spawning {total_workers} workers."
    )

    results = []
    with ProcessPoolExecutor(
        max_workers=total_workers,
        initializer=process_init,
        initargs=(rank_queue, args.asr_model),
    ) as executor:
        futures = [
            executor.submit(run_eval_worker, chunk, args.batch_size, args.max_seconds)
            for chunk in tasks
        ]
        with tqdm(total=total_files, desc="Eval Progress", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                chunk_metrics = future.result()
                results.extend(chunk_metrics)
                pbar.update(len(chunk_metrics))

    ers, inses, deles, subses = [], [], [], []
    unit_total = 0

    fout = None
    if args.decode_path:
        os.makedirs(os.path.dirname(args.decode_path), exist_ok=True)
        fout = open(args.decode_path, "w", encoding="utf8")
        logging.info(f"Saving detailed CER results to: {args.decode_path}")
        fout.write(
            "Name\tCER\tTruth\tHypothesis\tInsertions\tDeletions\tSubstitutions\n"
        )

    for res in results:
        ers.append(float(res["wer"]))
        inses.append(float(res["insertions"]))
        deles.append(float(res["deletions"]))
        subses.append(float(res["substitutions"]))
        unit_total += res["word_num"]
        if fout:
            fout.write(
                f"{res['wav_path']}\t{res['wer']}\t{res['truth']}\t"
                f"{res['hypo']}\t{res['insertions']}\t{res['deletions']}\t"
                f"{res['substitutions']}\n"
            )

    cer_avg = round(np.mean(ers) * 100, 2) if ers else float("nan")
    cer_weighted = (
        round((np.sum(subses) + np.sum(deles) + np.sum(inses)) / unit_total * 100, 2)
        if unit_total > 0
        else float("nan")
    )

    print("-" * 50)
    logging.info(f"Processed {len(results)}/{total_files} files.")
    avg_info = f"Hakka CER (Avg of sample CERs): {cer_avg}%"
    weighted_info = f"Hakka CER (Weighted): {cer_weighted}%"
    detail_info = (
        f"Errors: {np.sum(inses)} ins, {np.sum(deles)} del, "
        f"{np.sum(subses)} sub / {unit_total} chars"
    )
    logging.info(avg_info)
    logging.info(weighted_info)
    logging.info(detail_info)
    print("-" * 50)

    if fout:
        fout.write(avg_info + "\n" + weighted_info + "\n" + detail_info + "\n")
        fout.close()


if __name__ == "__main__":
    main()
