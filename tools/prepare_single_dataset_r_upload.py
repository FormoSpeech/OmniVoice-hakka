#!/usr/bin/env python3
"""Prepare a local -R dataset folder from extracted denoised audio."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True, help="Source HF dataset repo.")
    parser.add_argument("--target-name", required=True, help="Local target folder name.")
    parser.add_argument("--config-name", default=None, help="Optional HF config name.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--extracted-audio-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--copy-audio",
        action="store_true",
        help="Copy audio into the target folder instead of symlinking it.",
    )
    return parser.parse_args()


def sanitize_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.source_repo, args.config_name, split=args.split)
    skipped_missing_audio = 0
    skipped_zero_duration = 0

    repo_slug = sanitize_fragment(args.source_repo.replace("/", "__"))
    target_dir = args.output_root / args.target_name
    audio_dir = target_dir / "audio"
    meta_path = target_dir / f"{args.split}.jsonl"
    tmp_meta_path = target_dir / f"{args.split}.jsonl.tmp"
    audio_dir.mkdir(parents=True, exist_ok=True)

    with tmp_meta_path.open("w", encoding="utf-8") as fout:
        progress = tqdm(dataset, desc=f"Preparing {args.target_name}", unit="sample")
        for row in progress:
            config_slug = sanitize_fragment(str(row["lang_group_en"]))
            internal_id = (
                f"{repo_slug}__{config_slug}__"
                f"{sanitize_fragment(str(row['id']))}"
            )
            audio_name = f"{row['id']}.wav"
            src_audio = args.extracted_audio_dir / f"{internal_id}.wav"
            if not src_audio.exists():
                audio = row.get("audio")
                duration = row.get("duration")
                if audio is None:
                    skipped_missing_audio += 1
                    continue
                if duration is not None and float(duration) <= 0.0:
                    skipped_zero_duration += 1
                    continue
                raise FileNotFoundError(
                    f"Missing extracted denoised audio for internal id={internal_id}: "
                    f"{src_audio}"
                )
            dst_audio = audio_dir / audio_name
            if not dst_audio.exists():
                if args.copy_audio:
                    shutil.copy2(src_audio, dst_audio)
                else:
                    dst_audio.symlink_to(src_audio.resolve())
            row = {k: v for k, v in row.items() if k != "audio"}
            row["audio"] = f"audio/{audio_name}"
            fout.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    tmp_meta_path.replace(meta_path)
    print(f"Wrote {meta_path}")
    if skipped_missing_audio or skipped_zero_duration:
        print(
            "Skipped samples:",
            {
                "missing_audio": skipped_missing_audio,
                "zero_or_negative_duration": skipped_zero_duration,
            },
        )


if __name__ == "__main__":
    main()
