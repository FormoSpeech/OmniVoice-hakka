---
base_model: k2-fsa/OmniVoice
language:
- zh
- hak
library_name: omnivoice
pipeline_tag: text-to-speech
license: apache-2.0
tags:
- omnivoice
- text-to-speech
- voice-cloning
- voice-design
- taiwanese-hakka
- hakka
- multilingual
---

# OmniVoice Taiwanese Hakka

`formospeech/omnivoice-taiwanese-hakka` is a Taiwanese Hakka fine-tuned checkpoint based on [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice).

## Summary

- Base model: `k2-fsa/OmniVoice`
- Model type: `OmniVoice`
- Primary use case: Taiwanese Hakka text-to-speech
- Released checkpoint: trained for `16000` steps
- Supported generation modes:
  - zero-shot voice cloning with `ref_audio` + `ref_text`
  - voice design with `instruct`

## Usage

To get started, install the Hakka-enabled OmniVoice fork:

- `txya900619/OmniVoice-hakka`

This fork is required because the upstream `k2-fsa/OmniVoice` runtime does not support Hakka dialect labels such as `客語四縣腔` in `instruct`.

For this Hakka checkpoint, the target dialect must be specified through `instruct` if you want to control the generated Hakka dialect. This differs from the base OmniVoice usage because Hakka dialect prompting is part of the intended inference setup for this release.

### Installation

**Step 1**: Install PyTorch

<details>
<summary>NVIDIA GPU</summary>

```bash
# Install pytorch with your CUDA version, e.g.
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

> See [PyTorch official site](https://pytorch.org/get-started/locally/) for other versions installation.

</details>

<details>
<summary>Apple Silicon</summary>

```bash
pip install torch==2.8.0 torchaudio==2.8.0
```

</details>

**Step 2**: Install OmniVoice

```bash
pip install git+https://github.com/txya900619/OmniVoice-hakka.git
```

### Python API

```python
import soundfile as sf
import torch
from omnivoice import OmniVoice

model = OmniVoice.from_pretrained(
    "formospeech/omnivoice-taiwanese-hakka",
    device_map="cuda:0",
    dtype=torch.float16,
)

audio = model.generate(
    text="客語語音合成測試。",
    ref_audio="ref.wav",
    ref_text="這係參考語音。",
    instruct="客語四縣腔",
)

sf.write("out.wav", audio[0], 24000)
```

## Supported Hakka Dialect Labels

Supported Hakka dialect labels in this project are:

- `客語四縣腔`
- `客語海陸腔`
- `客語大埔腔`
- `客語饒平腔`
- `客語詔安腔`
- `客語南四縣腔`

## Training Data

Training uses a mixed FormoSpeech Hakka setup.

Original datasets:

- `formospeech/hakkaradio_news_clean`
- `formospeech/hat_tts_hailu_clean`
- `formospeech/hat_tts_sixian_clean`
- `formospeech/hakka_elearning_example_clean`

Denoised `-R` datasets:

- `formospeech/hat_asr_sixian_reading_clean_r`
- `formospeech/hat_asr_hailu_reading_clean_r`
- `formospeech/hat_asr_nansixian_reading_clean_r`
- `formospeech/hat_asr_sixian_broadcast_clean_r`

## Text Processing

Training uses cleaned `hanzi_cln` / `pinyin_cln` text with a restored-punctuation pipeline.

The restored pipeline applies the following steps:

1. Samples containing `<UNK>` or `<spn>` are filtered out.
2. `<SIL>` / `<sil>` markers are removed from the cleaned text.
3. The raw text is split into text units and the punctuation attached to each unit is recorded.
   - For Hanzi, the units are individual Chinese characters plus inline ASCII word spans.
   - For pinyin, the units are whitespace-delimited pinyin tokens.
4. The cleaned text is converted into the same kind of units after removing silence markers.
5. The raw-unit sequence and cleaned-unit sequence are aligned with partial matching.
6. Punctuation is restored for the parts of the cleaned text that still align with the raw text.
   - Small insertions, deletions, or replacements do not prevent punctuation recovery for surrounding matched regions.
   - Punctuation is copied back only for aligned regions and is not forced into mismatched regions.
7. After local punctuation transfer, the text is normalized:
   - leading punctuation is removed
   - missing or weak sentence-final punctuation is normalized
   - in `text_pinyin`, full-width punctuation is kept and extra spaces around punctuation are removed

In practice, this means the pipeline is conservative for local mismatches while still recovering punctuation around them. Short or fragmentary utterances can still be normalized into sentence-final `。`.

## Training Config

This release was trained with:

- cleaned `cln restored` manifests
- `instruct_ratio = 1.0`
- `only_instruct_ratio = 0.3`
- `use_pinyin_ratio = 0.3`
- total training steps: `16000`

## Evaluation

Evaluation uses a custom Taiwanese Hakka test list built from the `test` split of:

- `formospeech/hakkaradio_news_clean`
  - `Hakka_Hailu`
  - `Hakka_Sixian`

Construction rules:

- target utterances are selected with duration between `3` and `30` seconds
- reference utterances are selected with duration between `3` and `15` seconds
- each `ref_audio` comes from the same speaker as the target
- `ref_id != id`
- references are assigned to increase diversity per speaker

Evaluation metrics and models:

- CER
  - computed with the Taiwanese Hakka ASR model `formospeech/whisper-large-v2-taiwanese-hakka-v1`

- SIM-o
  - computed with a WavLM-based speaker verification model distributed through `k2-fsa/TTS_eval_models`

- UTMOS
  - computed with the UTMOS predictor distributed through `k2-fsa/TTS_eval_models`

Released checkpoint result:

| Metric | Score |
| --- | ---: |
| Hakka CER (Avg of sample CERs) | 2.61% |
| Hakka CER (Weighted) | 2.29% |
| SIM-o | 0.801 |
| UTMOS | 3.65 |

Comparison against the current F5-TTS baseline `formospeech/f5-tts-hita-finetune-v1` on the same Hakka evaluation setup:

- `g2p`: inference converts Hanzi to pinyin with `formog2p.hakka.g2p`
- `gt_pinyin`: inference uses the existing pinyin annotations from the manifest / test list

| Model | CER avg | CER weighted | SIM-o | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| OmniVoice Taiwanese Hakka | 2.61% | 2.29% | 0.801 | 3.65 |
| F5-TTS g2p | 6.74% | 6.43% | 0.794 | 3.29 |
| F5-TTS gt_pinyin | 6.20% | 5.87% | 0.795 | 3.26 |

## Notes

This release is optimized for Taiwanese Hakka TTS and keeps the general OmniVoice interface from the base model.

The original OmniVoice features for Voice Design and Fine-grained Control may be weaker in this checkpoint. The finetuning data used for this release does not provide the full annotation coverage needed to preserve those capabilities. This is especially true for Voice Design, because finetuning actively uses `instruct`, so the Hakka-dialect prompting setup can interfere more strongly with the broader attribute-control behavior from the base model.

Evaluation numbers reported here come from the project-side Hakka benchmark setup, not from a standardized public benchmark leaderboard.
