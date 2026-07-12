# Hakka Finetune and Evaluation Notes

This document records the current FormoSpeech Hakka fine-tuning and evaluation setup in this repository.

## Data

Source datasets:

- `formospeech/hakkaradio_news_clean`
- `formospeech/hat_tts_hailu_clean`
- `formospeech/hat_tts_sixian_clean`
- `formospeech/hakka_elearning_example_clean`

Exported manifests:

- train: [data/formospeech_hakka/train.jsonl](/home/wayne/OmniVoice/data/formospeech_hakka/train.jsonl)
- dev: [data/formospeech_hakka/dev.jsonl](/home/wayne/OmniVoice/data/formospeech_hakka/dev.jsonl)

Current field mapping:

- `text = hanzi`
- `text_pinyin = pinyin.upper()`
- `language_id = "zh"`
- `dialect` is inferred from `source_config`
- `instruct = dialect`

Current counts:

- `train`: `165202`
- `dev`: `4837`

## Finetune Setup

Finetune entrypoint:

- [examples/run_finetune.sh](/home/wayne/OmniVoice/examples/run_finetune.sh)

Related config:

- [examples/config/data_config_finetune.json](/home/wayne/OmniVoice/examples/config/data_config_finetune.json)
- [examples/config/train_config_finetune.json](/home/wayne/OmniVoice/examples/config/train_config_finetune.json)

Current defaults in `run_finetune.sh`:

- `TRAIN_JSONL=/home/wayne/OmniVoice/data/formospeech_hakka/train.jsonl`
- `DEV_JSONL=/home/wayne/OmniVoice/data/formospeech_hakka/dev.jsonl`
- `TOKEN_DIR=/home/wayne/OmniVoice/data/formospeech_hakka/tokens`
- `OUTPUT_DIR=/home/wayne/OmniVoice/exp/omnivoice_finetune_formospeech_hakka`

## Test List

Canonical custom evaluation list:

- [data/formospeech_hakka/dev_test_list.jsonl](/home/wayne/OmniVoice/data/formospeech_hakka/dev_test_list.jsonl)

Selection rules:

- target duration: `3` to `30` seconds
- ref duration: `3` to `15` seconds
- `ref_audio` is chosen from the same speaker
- `ref_id != id`
- refs are assigned to be as diverse as possible per speaker

Current size:

- `3935` samples

Test-list fields in use:

- `id`
- `audio_path`
- `text`
- `text_pinyin`
- `speaker`
- `duration`
- `ref_id`
- `ref_audio`
- `ref_text`
- `ref_duration`
- `language_id`
- `language_name`
- `dialect`
- `instruct`

## Dialect Control

The Hakka export pipeline now derives a dialect label from the dataset config
and writes it into both `dialect` and `instruct`. Current mappings:

- `Hakka_Sixian` -> `客語四縣腔`
- `Hakka_Hailu` -> `客語海陸腔`
- `Hakka_Dapu` -> `客語大埔腔`
- `Hakka_Raoping` -> `客語饒平腔`
- `Hakka_Zhaoan` -> `客語詔安腔`

To rebuild manifests from source datasets:

```bash
uv run --with datasets python tools/export_formospeech_hakka_jsonl.py
uv run python tools/build_formospeech_hakka_test_list.py
```

To backfill existing local manifests in place:

```bash
uv run python tools/add_hakka_dialect_instruct.py \
  data/formospeech_hakka/train.jsonl \
  data/formospeech_hakka/dev.jsonl \
  data/formospeech_hakka/dev_test_list.jsonl
```

## Evaluation Setup

Custom Hakka evaluation script:

- [examples/run_eval_hakka.sh](/home/wayne/OmniVoice/examples/run_eval_hakka.sh)

Custom CER evaluator:

- [omnivoice/eval/wer/hakka.py](/home/wayne/OmniVoice/omnivoice/eval/wer/hakka.py)

Hakka ASR model for CER:

- `formospeech/whisper-large-v2-taiwanese-hakka-v1`

Metrics:

- `CER` against `text`
- `SIM-o`
- `UTMOS`

`run_eval_hakka.sh` stages:

- `0`: download `k2-fsa/TTS_eval_models`
- `1`: batch inference
- `2`: Hakka CER
- `3`: SIM-o
- `4`: UTMOS

Run:

```bash
bash examples/run_eval_hakka.sh
```

If `torchcodec` is used, export:

```bash
export LD_LIBRARY_PATH="/home/wayne/ffmpeg-n7.1-latest-linux64-gpl-shared-7.1/lib:$LD_LIBRARY_PATH"
```

## Baseline

Baseline means evaluating the original target audio from `audio_path` in the test list.

Baseline logs:

- [exp/dev_test_list_baseline.hakka_cer.log](/home/wayne/OmniVoice/exp/dev_test_list_baseline.hakka_cer.log)
- [exp/dev_test_list_baseline.sim.log](/home/wayne/OmniVoice/exp/dev_test_list_baseline.sim.log)
- [exp/dev_test_list_baseline.utmos.log](/home/wayne/OmniVoice/exp/dev_test_list_baseline.utmos.log)

Baseline results:

| Model | CER avg | CER weighted | SIM-o | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| Original target audio | 3.83% | 3.52% | 0.780 | 3.50 |

## Finetuned Checkpoint Result

Checkpoint:

- `/home/wayne/OmniVoice/exp/omnivoice_finetune_formospeech_hakka/checkpoint-5000`

Reported results:

| Model | CER avg | CER weighted | SIM-o | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-5000 | 3.18% | 2.84% | 0.770 | 3.70 |

Comparison vs baseline:

- `CER weighted`: `3.52% -> 2.84%` (`-0.68`)
- `SIM-o`: `0.780 -> 0.770` (`-0.010`)
- `UTMOS`: `3.50 -> 3.70` (`+0.20`)

Current interpretation:

- text accuracy improved
- predicted naturalness improved
- speaker similarity dropped slightly

## Finetuned Checkpoint Result With Cleaned Training Text

Checkpoint:

- `/home/wayne/OmniVoice/exp/omnivoice_finetune_formospeech_hakka_cln/checkpoint-5000`

Condition:

- training used cleaned `hanzi_cln/pinyin_cln`
- `<UNK>/<spn>` samples were filtered out
- `<SIL>` was removed and punctuation was restored from the raw text
- training also used `only_instruct_ratio=0.3`
- training also used `use_pinyin_ratio=0.3`
- evaluation used the same test list as the raw-manifest run

Reported results:

| Model | CER avg | CER weighted | SIM-o | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-5000 (cln train, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 3.37% | 2.89% | 0.785 | 3.71 |

Comparison vs baseline:

- `CER weighted`: `3.52% -> 2.89%` (`-0.63`)
- `SIM-o`: `0.780 -> 0.785` (`+0.005`)
- `UTMOS`: `3.50 -> 3.71` (`+0.21`)

Comparison vs raw-manifest checkpoint:

- `CER weighted`: `2.84% -> 2.89%` (`+0.05`)
- `SIM-o`: `0.770 -> 0.785` (`+0.015`)
- `UTMOS`: `3.70 -> 3.71` (`+0.01`)

Current interpretation:

- content accuracy stayed close to the raw-manifest checkpoint
- speaker similarity improved noticeably over the raw-manifest checkpoint
- predicted naturalness improved slightly over the raw-manifest checkpoint

## Finetuned Checkpoint Result With Dialect Instruct

Condition:

- test list includes dialect labels in `instruct`
- training used `only_instruct_ratio=0.1`

Reported results:

| Model | CER avg | CER weighted | SIM-o | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-5000 + dialect instruct (`only_instruct_ratio=0.1`) | 4.40% | 3.76% | 0.784 | 3.73 |

Comparison vs baseline:

- `CER weighted`: `3.52% -> 3.76%` (`+0.24`)
- `SIM-o`: `0.780 -> 0.784` (`+0.004`)
- `UTMOS`: `3.50 -> 3.73` (`+0.23`)

Current interpretation:

- speaker similarity improved slightly
- predicted naturalness improved
- text accuracy degraded relative to both baseline and the no-instruct checkpoint result

## Finetuned Checkpoint Result With Dialect Instruct (`only_instruct_ratio=0.3`)

Condition:

- test list includes dialect labels in `instruct`
- training used `only_instruct_ratio=0.3`

Reported results:

| Model | CER avg | CER weighted | SIM-o | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-5000 + dialect instruct (`only_instruct_ratio=0.3`) | 3.97% | 3.35% | 0.779 | 3.70 |

Comparison vs baseline:

- `CER weighted`: `3.52% -> 3.35%` (`-0.17`)
- `SIM-o`: `0.780 -> 0.779` (`-0.001`)
- `UTMOS`: `3.50 -> 3.70` (`+0.20`)

Comparison vs dialect instruct (`only_instruct_ratio=0.1`):

- `CER weighted`: `3.76% -> 3.35%` (`-0.41`)
- `SIM-o`: `0.784 -> 0.779` (`-0.005`)
- `UTMOS`: `3.73 -> 3.70` (`-0.03`)

Current interpretation:

- text accuracy improved relative to the `only_instruct_ratio=0.1` setting
- speaker similarity returned close to baseline
- predicted naturalness stayed improved over baseline

## Finetuned Checkpoint Result With Dialect Instruct (`only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`)

Condition:

- test list includes dialect labels in `instruct`
- training used `only_instruct_ratio=0.3`
- training used `use_pinyin_ratio=0.3`

Reported results:

| Model | CER avg | CER weighted | SIM-o | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-5000 + dialect instruct (`only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 3.57% | 3.04% | 0.780 | 3.69 |

Comparison vs baseline:

- `CER weighted`: `3.52% -> 3.04%` (`-0.48`)
- `SIM-o`: `0.780 -> 0.780` (`+0.000`)
- `UTMOS`: `3.50 -> 3.69` (`+0.19`)

Comparison vs dialect instruct (`only_instruct_ratio=0.3`):

- `CER weighted`: `3.35% -> 3.04%` (`-0.31`)
- `SIM-o`: `0.779 -> 0.780` (`+0.001`)
- `UTMOS`: `3.70 -> 3.69` (`-0.01`)

Current interpretation:

- pinyin mixing further improved text accuracy under the dialect-instruct setup
- speaker similarity returned exactly to baseline
- predicted naturalness stayed above baseline with only a negligible drop vs the no-pinyin `0.3` setup

## Finetuned Checkpoint Result With Simple Cleaned Training Text

Checkpoint:

- `/home/wayne/OmniVoice/exp/omnivoice_finetune_formospeech_hakka_cln_simple/checkpoint-5000`

Condition:

- training used cleaned `hanzi_cln/pinyin_cln`
- `<UNK>/<spn>` samples were filtered out
- `<SIL>` was replaced with `，`
- leading punctuation was stripped
- trailing `，` or missing terminal punctuation was normalized to `。`
- Hanzi spaces in `cln simple` were removed
- training also used `only_instruct_ratio=0.3`
- training also used `use_pinyin_ratio=0.3`
- evaluation used the same test list as the raw-manifest run

Reported results:

| Model | CER avg | CER weighted | SIM-o | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-5000 (cln simple train, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 4.14% | 3.46% | 0.786 | 3.71 |

Comparison vs baseline:

- `CER weighted`: `3.52% -> 3.46%` (`-0.06`)
- `SIM-o`: `0.780 -> 0.786` (`+0.006`)
- `UTMOS`: `3.50 -> 3.71` (`+0.21`)

Comparison vs raw-manifest checkpoint:

- `CER weighted`: `2.84% -> 3.46%` (`+0.62`)
- `SIM-o`: `0.770 -> 0.786` (`+0.016`)
- `UTMOS`: `3.70 -> 3.71` (`+0.01`)

Comparison vs cleaned-text checkpoint with restored punctuation:

- `CER weighted`: `2.89% -> 3.46%` (`+0.57`)
- `SIM-o`: `0.785 -> 0.786` (`+0.001`)
- `UTMOS`: `3.71 -> 3.71` (`+0.00`)

Current interpretation:

- simple cleaned text preserved the SIM-o gain
- UTMOS stayed at the same level as the restored-punctuation cleaned-text run
- CER degraded substantially relative to both the raw-manifest and restored-punctuation cleaned-text runs

## Finetuned Checkpoint Result With Cleaned Training Text Plus Denoised Reading Data

Checkpoint:

- `/home/wayne/OmniVoice/exp/omnivoice_finetune_formospeech_hakka_cln_plus_denoised_reading/checkpoint-5000`

Condition:

- training used cleaned `hanzi_cln/pinyin_cln`
- the main Hakka corpus used the same `cln restored` setup as the cleaned-text run
- the three reading datasets were denoised with Sidon and their shard metadata was rewritten to the same `cln restored` format
- `<UNK>/<spn>` samples were filtered out from the denoised reading manifests as well
- training also used `only_instruct_ratio=0.3`
- training also used `use_pinyin_ratio=0.3`
- evaluation used the same test list as the raw-manifest run

Reported results:

| Model | CER avg | CER weighted | SIM-o | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-5000 (cln train + denoised reading, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 3.15% | 2.79% | 0.790 | 3.70 |

Comparison vs baseline:

- `CER weighted`: `3.52% -> 2.79%` (`-0.73`)
- `SIM-o`: `0.780 -> 0.790` (`+0.010`)
- `UTMOS`: `3.50 -> 3.70` (`+0.20`)

Comparison vs raw-manifest checkpoint:

- `CER weighted`: `2.84% -> 2.79%` (`-0.05`)
- `SIM-o`: `0.770 -> 0.790` (`+0.020`)
- `UTMOS`: `3.70 -> 3.70` (`+0.00`)

Comparison vs cleaned-text checkpoint:

- `CER weighted`: `2.89% -> 2.79%` (`-0.10`)
- `SIM-o`: `0.785 -> 0.790` (`+0.005`)
- `UTMOS`: `3.71 -> 3.70` (`-0.01`)

Current interpretation:

- this is the strongest OmniVoice result so far on both CER and SIM-o
- adding denoised reading data improved content accuracy relative to both the raw-manifest and cleaned-text checkpoints
- UTMOS stayed effectively flat relative to the strongest previous OmniVoice runs

## Finetuned Checkpoint Result With R-Mixed Training Data

Checkpoints:

- `/home/wayne/OmniVoice/exp/omnivoice_finetune_formospeech_hakka_r_mixed/checkpoint-5000`
- `/home/wayne/OmniVoice/exp/omnivoice_finetune_formospeech_hakka_r_mixed/checkpoint-9500`
- `/home/wayne/OmniVoice/exp/omnivoice_finetune_formospeech_hakka_r_mixed_16000/checkpoint-16000`

Condition:

- training used cleaned `hanzi_cln/pinyin_cln`
- radio / TTS / elearning datasets stayed on the original source audio
- reading and Sixian broadcast datasets were replaced by uploaded `-R` variants
- training used `instruct_ratio=1.0`
- training used `only_instruct_ratio=0.3`
- training used `use_pinyin_ratio=0.3`
- evaluation used the same test list as the other OmniVoice runs

Reported results:

| Model | CER avg | CER weighted | SIM-o | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-5000 (r_mixed train, `instruct_ratio=1.0`, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 3.20% | 2.84% | 0.804 | 3.69 |
| checkpoint-9500 (r_mixed train, `instruct_ratio=1.0`, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 2.79% | 2.43% | 0.800 | 3.66 |
| checkpoint-16000 (r_mixed train, `instruct_ratio=1.0`, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 2.61% | 2.29% | 0.801 | 3.65 |

Comparison vs baseline (`checkpoint-16000`):

- `CER weighted`: `3.52% -> 2.29%` (`-1.23`)
- `SIM-o`: `0.780 -> 0.801` (`+0.021`)
- `UTMOS`: `3.50 -> 3.65` (`+0.15`)

Comparison vs raw-manifest checkpoint (`checkpoint-16000`):

- `CER weighted`: `2.84% -> 2.29%` (`-0.55`)
- `SIM-o`: `0.770 -> 0.801` (`+0.031`)
- `UTMOS`: `3.70 -> 3.65` (`-0.05`)

Comparison vs cln + denoised reading checkpoint (`checkpoint-16000`):

- `CER weighted`: `2.79% -> 2.29%` (`-0.50`)
- `SIM-o`: `0.790 -> 0.801` (`+0.011`)
- `UTMOS`: `3.70 -> 3.65` (`-0.05`)

Current interpretation:

- `r_mixed checkpoint-16000` is now the strongest OmniVoice variant so far on both `CER` and `SIM-o`
- `r_mixed checkpoint-9500` already beats the earlier `cln + denoised reading` result on CER while keeping `SIM-o` high
- later `r_mixed` checkpoints improve CER substantially, with a mild UTMOS tradeoff

## Finetuned Checkpoint Result With R-Mixed Training Data Without Broadcast -R

Checkpoints:

- `/home/wayne/OmniVoice/exp/omnivoice_finetune_formospeech_hakka_r_mixed_no_broadcast/checkpoint-5000`
- `/home/wayne/OmniVoice/exp/omnivoice_finetune_formospeech_hakka_r_mixed_no_broadcast/checkpoint-9000`
- `/home/wayne/OmniVoice/exp/omnivoice_finetune_formospeech_hakka_r_mixed_no_broadcast/checkpoint-16000`

Condition:

- training used cleaned `hanzi_cln/pinyin_cln`
- radio / TTS / elearning datasets stayed on the original source audio
- reading datasets were replaced by uploaded `-R` variants
- `formospeech/hat_asr_sixian_broadcast_clean_r` was excluded
- training used `instruct_ratio=1.0`
- training used `only_instruct_ratio=0.3`
- training used `use_pinyin_ratio=0.3`
- evaluation used the same test list as the other OmniVoice runs

Reported results:

| Model | CER avg | CER weighted | SIM-o | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-5000 (r_mixed_no_broadcast train, `instruct_ratio=1.0`, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 3.09% | 2.76% | 0.792 | 3.67 |
| checkpoint-9000 (r_mixed_no_broadcast train, `instruct_ratio=1.0`, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 2.71% | 2.37% | 0.787 | 3.67 |
| checkpoint-16000 (r_mixed_no_broadcast train, `instruct_ratio=1.0`, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 2.59% | 2.29% | 0.786 | 3.67 |

Comparison vs r_mixed (`checkpoint-16000`):

- `CER weighted`: `2.29% -> 2.29%` (`+0.00`)
- `SIM-o`: `0.801 -> 0.786` (`-0.015`)
- `UTMOS`: `3.65 -> 3.67` (`+0.02`)

Comparison vs cln + denoised reading checkpoint (`checkpoint-16000`):

- `CER weighted`: `2.79% -> 2.29%` (`-0.50`)
- `SIM-o`: `0.790 -> 0.786` (`-0.004`)
- `UTMOS`: `3.70 -> 3.67` (`-0.03`)

Current interpretation:

- removing broadcast `-R` keeps the CER gain almost unchanged at `checkpoint-16000`
- the main regression is `SIM-o`, which drops noticeably relative to `r_mixed`
- this suggests the broadcast `-R` subset is contributing more to speaker similarity than to CER

## F5-TTS Result

Evaluation script:

- [examples/run_eval_f5_hakka.sh](/home/wayne/OmniVoice/examples/run_eval_f5_hakka.sh)

Inference wrapper:

- [tools/f5_hakka_infer_batch.py](/home/wayne/OmniVoice/tools/f5_hakka_infer_batch.py)

Current reported results:

| Model | CER avg | CER weighted | SIM-o | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| F5-TTS g2p | 6.74% | 6.43% | 0.794 | 3.29 |
| F5-TTS gt_pinyin | 6.20% | 5.87% | 0.795 | 3.26 |

Comparison across systems:

| Model | CER avg | CER weighted | SIM-o | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| Original target audio | 3.83% | 3.52% | 0.780 | 3.50 |
| OmniVoice checkpoint-5000 | 3.18% | 2.84% | 0.770 | 3.70 |
| OmniVoice checkpoint-5000 (cln train, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 3.37% | 2.89% | 0.785 | 3.71 |
| OmniVoice checkpoint-5000 (cln train + denoised reading, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 3.15% | 2.79% | 0.790 | 3.70 |
| OmniVoice checkpoint-5000 (r_mixed train, `instruct_ratio=1.0`, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 3.20% | 2.84% | 0.804 | 3.69 |
| OmniVoice checkpoint-9500 (r_mixed train, `instruct_ratio=1.0`, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 2.79% | 2.43% | 0.800 | 3.66 |
| OmniVoice checkpoint-16000 (r_mixed train, `instruct_ratio=1.0`, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 2.61% | 2.29% | 0.801 | 3.65 |
| OmniVoice checkpoint-5000 (r_mixed_no_broadcast train, `instruct_ratio=1.0`, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 3.09% | 2.76% | 0.792 | 3.67 |
| OmniVoice checkpoint-9000 (r_mixed_no_broadcast train, `instruct_ratio=1.0`, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 2.71% | 2.37% | 0.787 | 3.67 |
| OmniVoice checkpoint-16000 (r_mixed_no_broadcast train, `instruct_ratio=1.0`, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 2.59% | 2.29% | 0.786 | 3.67 |
| OmniVoice checkpoint-5000 (cln simple train, `only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 4.14% | 3.46% | 0.786 | 3.71 |
| OmniVoice + dialect instruct (`only_instruct_ratio=0.1`) | 4.40% | 3.76% | 0.784 | 3.73 |
| OmniVoice + dialect instruct (`only_instruct_ratio=0.3`) | 3.97% | 3.35% | 0.779 | 3.70 |
| OmniVoice + dialect instruct (`only_instruct_ratio=0.3`, `use_pinyin_ratio=0.3`) | 3.57% | 3.04% | 0.780 | 3.69 |
| F5-TTS g2p | 6.74% | 6.43% | 0.794 | 3.29 |
| F5-TTS gt_pinyin | 6.20% | 5.87% | 0.795 | 3.26 |

Current interpretation:

- OmniVoice is best on CER and UTMOS
- `r_mixed checkpoint-16000` is the best OmniVoice variant so far on both CER and SIM-o
- `r_mixed_no_broadcast checkpoint-16000` matches the best CER but gives up SIM-o
- the cleaned-text checkpoints remain strong SIM-o variants even without denoised reading
- neither cleaned-text result is an isolated cln-only ablation; both still include dialect instruct and pinyin mixing
- the `cln + denoised reading` result also includes dialect instruct and pinyin mixing
- the `r_mixed` result also includes dialect instruct and pinyin mixing
- the `r_mixed_no_broadcast` result also includes dialect instruct and pinyin mixing
- restored punctuation is clearly better than the simple `<SIL> -> ，` rule on CER
- within the dialect-instruct runs, `only_instruct_ratio=0.3 + use_pinyin_ratio=0.3` is the best CER tradeoff
- `r_mixed checkpoint-16000` is best on SIM-o among the current systems in this table
- F5-TTS `gt_pinyin` is slightly better than `g2p` on CER
- The gap between F5 `g2p` and `gt_pinyin` shows G2P contributes some error
- Even with GT pinyin, F5-TTS is still substantially worse than OmniVoice on content accuracy

## Notes

- For this Hakka setup, `CER` is treated as the primary metric.
- `SIM-o` and `UTMOS` are still useful, but should be interpreted as auxiliary metrics.
- A reasonable next step is to evaluate more checkpoints, such as `checkpoint-3000`, `checkpoint-7000`, and the latest checkpoint, using the same test list.
