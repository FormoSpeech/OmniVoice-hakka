#!/usr/bin/env python3
"""Backfill Hakka dialect labels and instruct fields into existing JSONL manifests.

Usage:
    uv run python tools/add_hakka_dialect_instruct.py \
        data/formospeech_hakka/train.jsonl \
        data/formospeech_hakka/dev.jsonl \
        data/formospeech_hakka/dev_test_list.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HAKKA_DIALECT_PATTERNS = [
    ("southsixian", "客語南四縣腔"),
    ("nansixian", "客語南四縣腔"),
    ("sixian", "客語四縣腔"),
    ("hailu", "客語海陸腔"),
    ("dapu", "客語大埔腔"),
    ("raoping", "客語饒平腔"),
    ("zhaoan", "客語詔安腔"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="JSONL manifest paths to rewrite in place.",
    )
    return parser.parse_args()


def resolve_hakka_dialect(row: dict) -> str | None:
    haystacks = [
        str(row.get("dialect") or ""),
        str(row.get("instruct") or ""),
        str(row.get("source_config") or ""),
        str(row.get("source_dataset") or ""),
        str(row.get("id") or ""),
        str(row.get("ref_id") or ""),
    ]
    normalized = " ".join(h.lower() for h in haystacks if h)
    for pattern, dialect in HAKKA_DIALECT_PATTERNS:
        if pattern in normalized:
            return dialect
    return None


def rewrite_jsonl(path: Path) -> tuple[int, int]:
    rows = []
    updated = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dialect = resolve_hakka_dialect(row)
            if row.get("dialect") != dialect or row.get("instruct") != dialect:
                row["dialect"] = dialect
                row["instruct"] = dialect
                updated += 1
            rows.append(row)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)
    return len(rows), updated


def main() -> None:
    args = parse_args()
    for path in args.paths:
        total, updated = rewrite_jsonl(path)
        print(f"{path}: rewrote {total} rows; updated {updated}")


if __name__ == "__main__":
    main()
