#!/usr/bin/env python3
"""Build a FormoSpeech Hakka test list with speaker-matched references.

Rules:
- target samples come from a source JSONL and must satisfy a target duration range
- reference samples must come from the same speaker and satisfy a ref duration range
- references are assigned in a round-robin manner per speaker to maximize diversity
- a sample will never use itself as its own reference
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/formospeech_hakka/dev.jsonl"),
        help="Source JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/formospeech_hakka/dev_test_list.jsonl"),
        help="Output JSONL file.",
    )
    parser.add_argument("--target-min", type=float, default=3.0)
    parser.add_argument("--target-max", type=float, default=30.0)
    parser.add_argument("--ref-min", type=float, default=3.0)
    parser.add_argument("--ref-max", type=float, default=15.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to shuffle reference pools deterministically.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_targets(rows: list[dict], min_dur: float, max_dur: float) -> list[dict]:
    return [r for r in rows if min_dur <= float(r["duration"]) <= max_dur]


def select_refs(rows: list[dict], min_dur: float, max_dur: float) -> dict[str, list[dict]]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        dur = float(row["duration"])
        if min_dur <= dur <= max_dur:
            by_speaker[row["speaker"]].append(row)
    return by_speaker


def assign_refs(
    targets: list[dict],
    refs_by_speaker: dict[str, list[dict]],
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    rotated_refs: dict[str, list[dict]] = {}
    ref_index: dict[str, int] = {}

    for speaker, refs in refs_by_speaker.items():
        pool = refs.copy()
        rng.shuffle(pool)
        rotated_refs[speaker] = pool
        ref_index[speaker] = 0

    output = []
    ref_usage = Counter()

    for row in targets:
        speaker = row["speaker"]
        pool = rotated_refs.get(speaker, [])
        if not pool:
            raise RuntimeError(f"No reference pool for speaker={speaker}")

        chosen = None
        start_idx = ref_index[speaker]
        pool_len = len(pool)

        for offset in range(pool_len):
            cand = pool[(start_idx + offset) % pool_len]
            if cand["id"] != row["id"]:
                chosen = cand
                ref_index[speaker] = (start_idx + offset + 1) % pool_len
                break

        if chosen is None:
            raise RuntimeError(
                f"Speaker {speaker} only has self-reference candidates for target {row['id']}"
            )

        item = row.copy()
        item["ref_id"] = chosen["id"]
        item["ref_audio"] = chosen["audio_path"]
        item["ref_text"] = chosen["text"]
        item["ref_duration"] = chosen["duration"]
        output.append(item)
        ref_usage[(speaker, chosen["id"])] += 1

    return output


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    targets = select_targets(rows, args.target_min, args.target_max)
    refs_by_speaker = select_refs(rows, args.ref_min, args.ref_max)
    output = assign_refs(targets, refs_by_speaker, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(output)} samples to {args.output}")
    print(
        f"Target duration range: {args.target_min}-{args.target_max}s; "
        f"Ref duration range: {args.ref_min}-{args.ref_max}s"
    )


if __name__ == "__main__":
    main()
