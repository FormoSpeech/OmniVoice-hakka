#!/usr/bin/env python3
"""Build a reproducible per-dialect eval test list for
formospeech/omnivoice-hakka-community-1.

Source: formospeech/hakkaradio_news_clean, `test` split, Hakka_Sixian and
Hakka_Hailu configs (the two dialects the released model card evaluates).

Pinned to commit e9e8762e4200c11781e9108924f832c4d46477aa of that dataset
(the `main` revision as of this writing) so the row selection stays
reproducible even if the dataset is updated later. Ref/target audio is not
redistributed with the test list; re-running this script against the pinned
revision regenerates it identically (this script is itself the artifact that
needs to be published for reproducibility, not the audio).

Text cleaning (per the model card's own "Text Processing" section):
    - rows containing <UNK> or <spn> in hanzi_cln are dropped (bad transcript)
    - <SIL> / <sil> markers are removed
    - remaining words are joined with no inter-word space (Hakka Hanzi
      convention used elsewhere in this repo's manifests)

Test-sample selection (matching this repo's existing convention in
tools/build_formospeech_hakka_test_list.py):
    - target duration in [3, 30]s
    - reference duration in [3, 15]s
    - ref_audio from the same speaker as the target, ref_id != id
    - references assigned round-robin per speaker to maximize diversity

Writes, per dialect, a JSONL test list (id, text, ref_audio, ref_text,
instruct, language_id, language_name, duration) plus the actual reference
wav files, into --out-root. Also writes a combined JSONL for a single
inference pass.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

DATASET_REPO = "formospeech/hakkaradio_news_clean"
DATASET_REVISION = "e9e8762e4200c11781e9108924f832c4d46477aa"

DIALECTS = {
    "Hakka_Sixian": "客語四縣腔",
    "Hakka_Hailu": "客語海陸腔",
}
TARGET_RANGE = (3.0, 30.0)
REF_RANGE = (3.0, 15.0)


def clean_hanzi(text: str) -> str | None:
    if "<UNK>" in text or "<spn>" in text:
        return None
    words = [w for w in text.split() if w not in ("<SIL>", "<sil>")]
    joined = "".join(words)
    return joined or None


def clean_pinyin(text: str) -> str:
    words = [w for w in text.split() if w not in ("<SIL>", "<sil>")]
    return " ".join(words)


def load_rows(config: str) -> list[dict]:
    ds = load_dataset(DATASET_REPO, config, split="test", revision=DATASET_REVISION)
    ds = ds.cast_column("audio", Audio(decode=False))
    rows = []
    for r in ds:
        if r["mismatched_trs"]:
            continue
        cleaned = clean_hanzi(r["hanzi_cln"])
        if cleaned is None:
            continue
        rows.append(
            {
                "id": r["id"],
                "text": cleaned,
                "text_pinyin": clean_pinyin(r["pinyin_cln"]),
                "duration": float(r["duration"]),
                "speaker": r["speaker"],
                "audio": r["audio"],
            }
        )
    return rows


def select_targets(rows: list[dict]) -> list[dict]:
    lo, hi = TARGET_RANGE
    return [r for r in rows if lo <= r["duration"] <= hi]


def select_refs(rows: list[dict]) -> dict[str, list[dict]]:
    lo, hi = REF_RANGE
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if lo <= r["duration"] <= hi:
            by_speaker[r["speaker"]].append(r)
    return by_speaker


def assign_refs(targets: list[dict], refs_by_speaker: dict[str, list[dict]], seed: int) -> list[dict]:
    rng = random.Random(seed)
    rotated: dict[str, list[dict]] = {}
    idx: dict[str, int] = {}
    for speaker, refs in refs_by_speaker.items():
        pool = refs.copy()
        rng.shuffle(pool)
        rotated[speaker] = pool
        idx[speaker] = 0

    output = []
    for row in targets:
        speaker = row["speaker"]
        pool = rotated.get(speaker, [])
        if not pool:
            raise RuntimeError(f"No reference pool for speaker={speaker}")

        chosen = None
        start = idx[speaker]
        n = len(pool)
        for offset in range(n):
            cand = pool[(start + offset) % n]
            if cand["id"] != row["id"]:
                chosen = cand
                idx[speaker] = (start + offset + 1) % n
                break
        if chosen is None:
            raise RuntimeError(f"Speaker {speaker} has only self-reference candidates for {row['id']}")

        item = dict(row)
        item["ref"] = chosen
        output.append(item)
    return output


def save_ref_audio(item: dict, dialect: str, out_root: Path) -> str:
    ref = item["ref"]
    ref_dir = out_root / "ref_audio" / dialect
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path = ref_dir / f"{ref['id']}.wav"
    if not ref_path.exists():
        audio = ref["audio"]
        data, sr = sf.read(io.BytesIO(audio["bytes"]))
        sf.write(ref_path, data, sr)
    return str(ref_path)


def save_target_audio(item: dict, full_id: str, out_root: Path) -> str:
    """Save the target utterance's own (real, human) audio, flat and named by
    the full test-list id, matching the layout `--wav-path` expects for the
    CER/SIM/UTMOS scripts (used for the ground-truth baseline)."""
    baseline_dir = out_root / "baseline_audio"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    out_path = baseline_dir / f"{full_id}.wav"
    if not out_path.exists():
        audio = item["audio"]
        data, sr = sf.read(io.BytesIO(audio["bytes"]))
        sf.write(out_path, data, sr)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("/mnt/md1/user_wayne/omnivoice_eval/data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    combined = []

    for config, instruct in DIALECTS.items():
        print(f"=== {config} ===")
        rows = load_rows(config)
        print(f"  loaded {len(rows)} clean rows")

        targets = select_targets(rows)
        refs_by_speaker = select_refs(rows)
        assigned = assign_refs(targets, refs_by_speaker, args.seed)

        out_rows = []
        for item in assigned:
            full_id = f"{config}__{item['id']}"
            ref_audio_path = save_ref_audio(item, config, args.out_root)
            save_target_audio(item, full_id, args.out_root)
            out_rows.append(
                {
                    "id": full_id,
                    "text": item["text"],
                    "text_pinyin": item["text_pinyin"],
                    "ref_id": f"{config}__{item['ref']['id']}",
                    "ref_audio": ref_audio_path,
                    "ref_text": item["ref"]["text"],
                    "ref_text_pinyin": item["ref"]["text_pinyin"],
                    "instruct": instruct,
                    "language_id": "hak",
                    "language_name": config,
                    "duration": item["duration"],
                }
            )

        out_path = args.out_root / f"test_list_{config}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  wrote {len(out_rows)} samples to {out_path}")

        combined.extend(out_rows)

    combined_path = args.out_root / "test_list_combined.jsonl"
    with combined_path.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(combined)} combined samples to {combined_path}")


if __name__ == "__main__":
    main()
