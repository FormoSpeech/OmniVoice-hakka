#!/usr/bin/env python3
"""Denoise a JSONL dataset with Sidon and write WAV files directly."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from omnivoice.data.batching import StreamLengthGroupDataset
from omnivoice.data.dataset import JsonlDatasetReader
from omnivoice.scripts.denoise_audio import (
    CollateFunction,
    GPUWorkerPool,
    SIDON_INPUT_SAMPLE_RATE,
    SIDON_OUTPUT_SAMPLE_RATE,
    count_lines,
)
from omnivoice.utils.common import str2bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_extractor_path", required=True)
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--target_sample_rate", type=int, default=24000)
    parser.add_argument("--batch_duration", type=float, default=200.0)
    parser.add_argument("--max_sample", type=int, default=32)
    parser.add_argument("--min_length", type=float, default=0.0)
    parser.add_argument("--max_length", type=float, default=80.0)
    parser.add_argument("--nj_per_gpu", type=int, default=1)
    parser.add_argument("--loader_workers", type=int, default=16)
    parser.add_argument("--shuffle", type=str2bool, default=True)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--skip_errors", action="store_true")
    return parser


def write_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    array = waveform.cpu().numpy()
    sf.write(path, array, sample_rate, format="WAV")


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    args = build_parser().parse_args()

    input_jsonl = Path(args.input_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    audio_dir = output_dir / "audio"
    output_jsonl = output_dir / "train.jsonl"
    errors_jsonl = output_dir / "errors.jsonl"
    audio_dir.mkdir(parents=True, exist_ok=True)

    total_samples = count_lines(str(input_jsonl))
    base_dataset = JsonlDatasetReader(
        str(input_jsonl),
        sample_rate=SIDON_INPUT_SAMPLE_RATE,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
    )
    batched_dataset = StreamLengthGroupDataset(
        dataset=base_dataset,
        batch_duration=args.batch_duration,
        max_sample=args.max_sample,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    collate_fn = CollateFunction(
        skip_errors=args.skip_errors,
        sample_rate=SIDON_INPUT_SAMPLE_RATE,
    )
    dataloader = DataLoader(
        dataset=batched_dataset,
        batch_size=None,
        collate_fn=collate_fn,
        num_workers=args.loader_workers,
        prefetch_factor=10 if args.loader_workers > 0 else None,
        pin_memory=True,
        persistent_workers=args.loader_workers > 0,
    )

    num_devices = torch.cuda.device_count()
    num_processes = max(1, num_devices * args.nj_per_gpu) if num_devices else args.nj_per_gpu
    pool_specs = (
        [(None, num_processes)]
        if num_devices == 0
        else [(gpu_id, args.nj_per_gpu) for gpu_id in range(num_devices)]
    )
    logging.info(
        f"GPU count: {num_devices}, Processes per GPU: {args.nj_per_gpu}, "
        f"Total processes: {num_processes}"
    )

    processed_count = 0
    error_count = 0
    failed_ids: list[str] = []
    main_progress = tqdm(total=total_samples, desc="Denoising Audio")

    with output_jsonl.open("w", encoding="utf-8") as train_f, errors_jsonl.open(
        "w", encoding="utf-8"
    ) as err_f:
        pool = GPUWorkerPool(pool_specs, args.feature_extractor_path, args.decoder_path)
        futures = set()
        max_pending = num_processes * 2

        def handle_success(result: dict) -> None:
            nonlocal processed_count
            for key, cleaned, metadata in zip(
                result["keys"], result["results"], result["metadata"]
            ):
                waveform = cleaned
                if args.target_sample_rate != SIDON_OUTPUT_SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(
                        waveform,
                        orig_freq=SIDON_OUTPUT_SAMPLE_RATE,
                        new_freq=args.target_sample_rate,
                    )
                waveform = (waveform / (waveform.abs().max() + 1e-7)) * 0.6
                wav_path = audio_dir / f"{key}.wav"
                write_wav(wav_path, waveform, args.target_sample_rate)

                row = dict(metadata)
                row["audio_path"] = str(wav_path.resolve())
                train_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                processed_count += 1

        def handle_error(result: dict) -> None:
            nonlocal error_count
            error_count += result["size"]
            failed_ids.extend(result["keys"])
            for key in result["keys"]:
                err_f.write(
                    json.dumps({"id": key, "reason": result["error"]}, ensure_ascii=False)
                    + "\n"
                )
            if not args.skip_errors:
                raise RuntimeError(
                    f"Batch starting with {result['keys'][0]} failed: {result['error']}"
                )

        def drain_completed() -> None:
            nonlocal futures
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                futures.discard(f)
                result = f.result()
                main_progress.update(result["size"])
                if result["status"] == "success":
                    handle_success(result)
                else:
                    handle_error(result)
                main_progress.set_postfix(OK=processed_count, Err=error_count)

        try:
            for batch in dataloader:
                if batch.size == 0:
                    continue
                if len(futures) >= max_pending:
                    drain_completed()
                futures.add(pool.submit(batch))

            while futures:
                drain_completed()
        finally:
            pool.shutdown()
            main_progress.close()

    logging.info(
        f"Processing complete. Successful: {processed_count}, Failed: {error_count}, "
        f"Output JSONL: {output_jsonl}"
    )
    if failed_ids:
        logging.warning(f"Failed sample count: {len(failed_ids)}")


if __name__ == "__main__":
    main()
