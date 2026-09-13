#!/usr/bin/env python3
"""Compute character error rate (CER) for Hakka using a Whisper ASR model
served behind a vLLM OpenAI-compatible `/v1/audio/transcriptions` endpoint.

This is a drop-in alternative to `omnivoice.eval.wer.hakka` for setups where
the Hakka ASR model is served by vLLM (e.g. in a separate Docker container)
instead of loaded in-process via `transformers.pipeline`. Talking to the
server over HTTP means this script needs no GPU/torch access itself, so it
can run in a plain CPU environment with just `requests` installed.
"""

import argparse
import logging
import os
import string
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm
from zhon.hanzi import punctuation

from omnivoice.eval.wer.common import process_one
from omnivoice.utils.data_utils import read_test_list


def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes CER for Hakka using a vLLM-served Whisper ASR model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wav-path", type=str, default=None)
    parser.add_argument("--test-list", type=str, required=True)
    parser.add_argument("--decode-path", type=str, default=None)
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:18000",
        help="Base URL of the vLLM OpenAI-compatible server.",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="formospeech/whisper-large-v2-taiwanese-hakka-v1",
        help="Model name as registered with the vLLM server (usually the HF repo id).",
    )
    parser.add_argument("--extension", type=str, default="wav")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Number of concurrent in-flight HTTP requests to the server.",
    )
    return parser


def post_process(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    punctuation_all = punctuation + string.punctuation
    for ch in punctuation_all:
        text = text.replace(ch, "")
    text = "".join(text.split())
    return " ".join(list(text))


def transcribe_one(session, server_url, asr_model, wav_path):
    with open(wav_path, "rb") as f:
        resp = session.post(
            f"{server_url}/v1/audio/transcriptions",
            files={"file": (os.path.basename(wav_path), f, "audio/wav")},
            data={"model": asr_model},
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()["text"].strip()


def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )

    logging.info(f"Calculating CER for {args.wav_path} via vLLM server {args.server_url}")
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

    session = requests.Session()
    results = []

    def _run(item):
        hypothesis = transcribe_one(session, args.server_url, args.asr_model, item["wav_path"])
        m = process_one(hypothesis, item["truth_text"], post_process)
        m["wav_path"] = item["wav_path"]
        return m

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(_run, item): item for item in data_list}
        with tqdm(total=total_files, desc="Eval Progress", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                item = futures[future]
                try:
                    results.append(future.result())
                except Exception:
                    logging.exception(f"Failed on {item['wav_path']}")
                pbar.update(1)

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
