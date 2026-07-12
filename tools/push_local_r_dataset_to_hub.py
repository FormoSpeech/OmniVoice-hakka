#!/usr/bin/env python3
"""Push a local -R dataset folder to the Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Audio, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-shard-size", default="500MB")
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--num-proc", type=int, default=None)
    return parser.parse_args()


def absolutize_audio(example: dict, input_dir: str) -> dict:
    example["audio"] = str((Path(input_dir) / example["audio"]).resolve())
    return example


def main() -> None:
    args = parse_args()
    manifest = args.input_dir / f"{args.split}.jsonl"
    dataset = load_dataset(
        "json",
        data_files={args.split: str(manifest)},
        split=args.split,
    )
    dataset = dataset.map(
        absolutize_audio,
        fn_kwargs={"input_dir": str(args.input_dir)},
        desc=f"Absolutizing audio paths for {args.input_dir.name}",
    )
    dataset = dataset.cast_column("audio", Audio())
    dataset.push_to_hub(
        repo_id=args.repo_id,
        config_name=args.config_name,
        split=args.split,
        max_shard_size=args.max_shard_size,
        num_shards=args.num_shards,
        num_proc=args.num_proc,
        embed_external_files=True,
        commit_message=f"Upload {args.input_dir.name} {args.split} split",
    )


if __name__ == "__main__":
    main()
