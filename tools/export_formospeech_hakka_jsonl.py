#!/usr/bin/env python3
"""Export selected FormoSpeech Hakka datasets into OmniVoice JSONL manifests.

This script downloads Hugging Face datasets, writes the embedded audio bytes to
local files, and produces combined train/dev JSONL files for OmniVoice
fine-tuning.

Usage:
    uv run --with datasets python tools/export_formospeech_hakka_jsonl.py
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Iterable

from datasets import Audio, Dataset, get_dataset_config_names, load_dataset


DEFAULT_DATASETS = [
    "formospeech/hakkaradio_news_clean",
    "formospeech/hat_tts_hailu_clean",
    "formospeech/hat_tts_sixian_clean",
    "formospeech/hakka_elearning_example_clean",
]

PUNCT_NO_SPACE_BEFORE = r",.;:!?%)}\]\u3001\u3002\uff0c\uff1b\uff1a\uff01\uff1f\uff09\u3009\u300b\u300d\u300f\u3011"
PUNCT_NO_SPACE_AFTER = r"({\[\u3008\u300a\u300c\u300e\u3010"
SIL_TAG_RE = re.compile(r"<sil>", re.IGNORECASE)
UNK_TAG_RE = re.compile(r"<unk>", re.IGNORECASE)
SPN_TAG_RE = re.compile(r"<spn>", re.IGNORECASE)
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
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Hugging Face dataset repos to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/formospeech_hakka"),
        help="Directory to write audio files and JSONL manifests.",
    )
    parser.add_argument(
        "--train-jsonl",
        type=str,
        default="train.jsonl",
        help="Train JSONL filename under output-dir.",
    )
    parser.add_argument(
        "--dev-jsonl",
        type=str,
        default="dev.jsonl",
        help="Dev JSONL filename under output-dir.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete output-dir before exporting.",
    )
    parser.add_argument(
        "--text-variant",
        choices=("raw", "cln"),
        default="raw",
        help="Choose whether to export hanzi/pinyin or hanzi_cln/pinyin_cln.",
    )
    parser.add_argument(
        "--cln-mode",
        choices=("restored", "simple"),
        default="restored",
        help="For cln export: 'restored' recovers punctuation from raw text; "
        "'simple' replaces <SIL> with commas.",
    )
    parser.add_argument(
        "--strip-special-tags",
        action="store_true",
        help="Remove tags like <SIL> or <UNK> from the selected text fields.",
    )
    return parser.parse_args()


def cleanup_spaces_around_punctuation(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(rf"\s+([{PUNCT_NO_SPACE_BEFORE}])", r"\1", text)
    text = re.sub(rf"([{PUNCT_NO_SPACE_AFTER}])\s+", r"\1", text)
    return text.strip()


def is_punctuation(char: str) -> bool:
    return bool(char) and unicodedata.category(char).startswith("P")


def contains_unknown_tag(text: str) -> bool:
    return bool(UNK_TAG_RE.search(text) or SPN_TAG_RE.search(text))


def remove_sil_tags(text: str) -> str:
    return re.sub(r"\s+", " ", SIL_TAG_RE.sub(" ", text)).strip()


def replace_sil_tags_with_commas(text: str) -> str:
    return re.sub(r"\s+", " ", SIL_TAG_RE.sub(" ， ", text)).strip()


def is_inline_ascii_word_char(char: str) -> bool:
    return char.isascii() and char.isalnum()


def consume_text_unit(text: str, start: int) -> tuple[str, int]:
    char = text[start]
    if not is_inline_ascii_word_char(char):
        return char, start + 1

    end = start + 1
    while end < len(text):
        curr = text[end]
        if is_inline_ascii_word_char(curr):
            end += 1
            continue
        if (
            curr == "-"
            and end + 1 < len(text)
            and is_inline_ascii_word_char(text[end - 1])
            and is_inline_ascii_word_char(text[end + 1])
        ):
            end += 1
            continue
        break
    return text[start:end], end


def extract_text_units_and_punct(raw_text: str) -> tuple[list[str], list[str]]:
    units: list[str] = []
    punct_after: list[str] = []
    pending_punct: list[str] = []

    idx = 0
    while idx < len(raw_text):
        char = raw_text[idx]
        if char.isspace():
            idx += 1
            continue
        if is_punctuation(char):
            pending_punct.append(char)
            idx += 1
            continue
        if units and pending_punct:
            punct_after[-1] += "".join(pending_punct)
            pending_punct = []
        unit, idx = consume_text_unit(raw_text, idx)
        units.append(unit)
        punct_after.append("")

    if units and pending_punct:
        punct_after[-1] += "".join(pending_punct)

    return units, punct_after


def extract_text_units(text: str) -> list[str]:
    units: list[str] = []
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char.isspace():
            idx += 1
            continue
        if is_punctuation(char):
            idx += 1
            continue
        unit, idx = consume_text_unit(text, idx)
        units.append(unit)
    return units


def tokenize_with_punctuation(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
            continue
        if is_punctuation(char):
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(char)
            continue
        current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def extract_token_units_and_punct(raw_text: str) -> tuple[list[str], list[str]]:
    units: list[str] = []
    punct_after: list[str] = []
    pending_punct: list[str] = []

    for token in tokenize_with_punctuation(raw_text):
        if token and all(is_punctuation(ch) for ch in token):
            pending_punct.append(token)
            continue
        if units and pending_punct:
            punct_after[-1] += "".join(pending_punct)
            pending_punct = []
        units.append(token)
        punct_after.append("")

    if units and pending_punct:
        punct_after[-1] += "".join(pending_punct)

    return units, punct_after


def strip_leading_punctuation(text: str) -> str:
    while text and (text[0].isspace() or is_punctuation(text[0])):
        text = text[1:]
    return text.lstrip()


def normalise_terminal_punctuation(text: str) -> str:
    text = text.rstrip()
    if not text:
        return text

    if text[-1] in ",，":
        text = text[:-1].rstrip() + "。"
    elif not is_punctuation(text[-1]):
        text += "。"

    return text


def normalise_terminal_punctuation_pinyin(text: str) -> str:
    text = text.rstrip()
    if not text:
        return text

    if text[-1] in ",，":
        text = text[:-1].rstrip() + "。"
    elif not is_punctuation(text[-1]):
        text += "。"

    return text


def normalise_pinyin_punctuation(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s+([,.;:!?，。；：！？、])", r"\1", text)
    text = re.sub(r"([,.;:!?，。；：！？、])\s+", r"\1", text)
    return text.strip()


def project_punctuation(
    raw_units: list[str], punct_after: list[str], cln_units: list[str]
) -> list[str]:
    punct_for_cln = [""] * len(cln_units)
    matcher = difflib.SequenceMatcher(a=raw_units, b=cln_units, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for raw_idx, cln_idx in zip(range(i1, i2), range(j1, j2)):
            punct_for_cln[cln_idx] = punct_after[raw_idx]
    return punct_for_cln


def repunctuate_cln_char(raw_text: str, cln_text: str) -> str:
    raw_units, punct_after = extract_text_units_and_punct(raw_text)
    cln_units = extract_text_units(cln_text)
    cln_punct = project_punctuation(raw_units, punct_after, cln_units)
    parts = []
    for unit, punct in zip(cln_units, cln_punct):
        parts.append(unit + punct)
    text = "".join(parts)

    text = cleanup_spaces_around_punctuation(text)
    text = strip_leading_punctuation(text)
    return normalise_terminal_punctuation(text)


def repunctuate_cln_tokens(raw_text: str, cln_text: str) -> str:
    raw_units, punct_after = extract_token_units_and_punct(raw_text)
    cln_units = cln_text.split()
    cln_punct = project_punctuation(raw_units, punct_after, cln_units)
    parts = []
    for unit, punct in zip(cln_units, cln_punct):
        parts.append(unit)
        if punct:
            parts.append(punct)
    text = " ".join(parts)

    text = normalise_pinyin_punctuation(text)
    text = strip_leading_punctuation(text)
    return normalise_terminal_punctuation_pinyin(text)


def choose_text(
    row: dict,
    raw_key: str,
    cln_key: str,
    text_variant: str,
    cln_mode: str,
    strip_special_tags: bool,
) -> str:
    if text_variant == "cln":
        raw_value = row.get(raw_key) or row.get(cln_key) or ""
        cln_value = row.get(cln_key) or row.get(raw_key) or ""
        if not isinstance(raw_value, str):
            raw_value = str(raw_value)
        if not isinstance(cln_value, str):
            cln_value = str(cln_value)
        if contains_unknown_tag(cln_value):
            return ""

        if cln_mode == "simple":
            value = replace_sil_tags_with_commas(cln_value)
            if raw_key == "hanzi":
                value = value.replace(" ", "")
        else:
            cln_value = remove_sil_tags(cln_value)
            if raw_key == "hanzi":
                value = repunctuate_cln_char(raw_value, cln_value)
            else:
                value = repunctuate_cln_tokens(raw_value, cln_value)
        if raw_key == "pinyin":
            value = strip_leading_punctuation(normalise_pinyin_punctuation(value))
            value = normalise_terminal_punctuation_pinyin(value)
        else:
            value = strip_leading_punctuation(cleanup_spaces_around_punctuation(value))
            value = normalise_terminal_punctuation(value)
    else:
        value = row.get(raw_key) or row.get(cln_key) or ""
        if not isinstance(value, str):
            value = str(value)
        if strip_special_tags:
            value = re.sub(r"<[^>]+>", " ", value)
    if raw_key == "pinyin":
        return normalise_pinyin_punctuation(value)
    return cleanup_spaces_around_punctuation(value)


def sanitize_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def iter_rows(dataset: Dataset) -> Iterable[dict]:
    for row in dataset:
        yield row


def resolve_hakka_dialect(dataset_repo: str, config_name: str, row: dict) -> str | None:
    haystacks = [
        config_name,
        dataset_repo,
        str(row.get("dialect") or ""),
        str(row.get("accent") or ""),
        str(row.get("id") or ""),
    ]
    normalized = " ".join(h.lower() for h in haystacks if h)
    for pattern, dialect in HAKKA_DIALECT_PATTERNS:
        if pattern in normalized:
            return dialect
    return None


def export_split(
    split_name: str,
    dataset_repo: str,
    config_name: str,
    dataset: Dataset,
    audio_root: Path,
    fout,
    text_variant: str,
    cln_mode: str,
    strip_special_tags: bool,
) -> int:
    count = 0
    skipped_unknown = 0
    skipped_missing_audio = 0
    repo_slug = dataset_repo.replace("/", "__")
    config_slug = sanitize_fragment(config_name)
    split_audio_dir = audio_root / repo_slug / config_slug / split_name
    split_audio_dir.mkdir(parents=True, exist_ok=True)

    for row in iter_rows(dataset):
        audio = row.get("audio")
        if audio is None:
            skipped_missing_audio += 1
            continue
        audio_bytes = audio.get("bytes")
        if audio_bytes is None:
            skipped_missing_audio += 1
            continue
        rel_name = Path(audio.get("path") or f"{row['id']}.wav").name
        suffix = Path(rel_name).suffix or ".wav"

        sample_id = (
            f"{sanitize_fragment(repo_slug)}__{config_slug}__{sanitize_fragment(row['id'])}"
        )
        audio_path = split_audio_dir / f"{sample_id}{suffix}"
        if not audio_path.exists() or audio_path.stat().st_size != len(audio_bytes):
            audio_path.write_bytes(audio_bytes)

        text = choose_text(
            row,
            raw_key="hanzi",
            cln_key="hanzi_cln",
            text_variant=text_variant,
            cln_mode=cln_mode,
            strip_special_tags=strip_special_tags,
        )
        text_pinyin = choose_text(
            row,
            raw_key="pinyin",
            cln_key="pinyin_cln",
            text_variant=text_variant,
            cln_mode=cln_mode,
            strip_special_tags=strip_special_tags,
        ).upper()
        dialect = resolve_hakka_dialect(dataset_repo, config_name, row)
        if text_variant == "cln" and (not text or not text_pinyin):
            skipped_unknown += 1
            continue
        if not text:
            raise ValueError(
                f"Missing transcript for dataset={dataset_repo} config={config_name} id={row['id']}"
            )

        record = {
            "id": sample_id,
            "audio_path": str(audio_path.resolve()),
            "text": text,
            "text_pinyin": text_pinyin,
            "text_variant": text_variant,
            "cln_mode": cln_mode if text_variant == "cln" else None,
            "special_tags_stripped": strip_special_tags or text_variant == "cln",
            "language_id": "zh",
            "dialect": dialect,
            "instruct": dialect,
            "source_dataset": dataset_repo,
            "source_config": config_name,
            "source_split": split_name,
            "speaker": row.get("speaker"),
            "duration": row.get("duration"),
            "mandarin": row.get("mandarin"),
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

    if skipped_unknown:
        print(
            f"Skipped {skipped_unknown} samples with <UNK>/<spn> in "
            f"{dataset_repo} [{config_name}] {split_name}"
        )
    if skipped_missing_audio:
        print(
            f"Skipped {skipped_missing_audio} samples with missing audio in "
            f"{dataset_repo} [{config_name}] {split_name}"
        )

    return count


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    audio_root = output_dir / "audio"
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = output_dir / args.train_jsonl
    dev_jsonl = output_dir / args.dev_jsonl
    summary = []
    failures = []

    with train_jsonl.open("w", encoding="utf-8") as train_f, dev_jsonl.open(
        "w", encoding="utf-8"
    ) as dev_f:
        for dataset_repo in args.datasets:
            for config_name in get_dataset_config_names(dataset_repo):
                try:
                    ds_dict = load_dataset(dataset_repo, config_name)
                    ds_dict = ds_dict.cast_column("audio", Audio(decode=False))

                    if "train" in ds_dict:
                        count = export_split(
                            "train",
                            dataset_repo,
                            config_name,
                            ds_dict["train"],
                            audio_root,
                            train_f,
                            args.text_variant,
                            args.cln_mode,
                            args.strip_special_tags,
                        )
                        summary.append((dataset_repo, config_name, "train", count))
                    if "test" in ds_dict:
                        count = export_split(
                            "test",
                            dataset_repo,
                            config_name,
                            ds_dict["test"],
                            audio_root,
                            dev_f,
                            args.text_variant,
                            args.cln_mode,
                            args.strip_special_tags,
                        )
                        summary.append((dataset_repo, config_name, "test", count))
                except Exception as exc:  # pragma: no cover - operational path
                    failures.append((dataset_repo, config_name, str(exc)))
                    print(
                        f"Skipping {dataset_repo} [{config_name}] due to error: {exc}"
                    )

    print(f"Wrote train JSONL: {train_jsonl}")
    print(f"Wrote dev JSONL:   {dev_jsonl}")
    for dataset_repo, config_name, split_name, count in summary:
        print(
            f"- {dataset_repo} [{config_name}] {split_name}: {count} samples"
        )
    if failures:
        print("Failures:")
        for dataset_repo, config_name, message in failures:
            print(f"- {dataset_repo} [{config_name}]: {message}")


if __name__ == "__main__":
    main()
