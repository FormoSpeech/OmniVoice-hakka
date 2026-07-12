# OmniVoice Working Notes

## Environment

- Use `uv` for Python commands and dependency execution.
- If using `torchcodec`, export:
  `LD_LIBRARY_PATH="/home/wayne/ffmpeg-n7.1-latest-linux64-gpl-shared-7.1/lib:$LD_LIBRARY_PATH"`

## Hakka Instruct Labels

- Supported Hakka dialect instruct labels are canonical only:
  - `客語四縣腔`
  - `客語海陸腔`
  - `客語大埔腔`
  - `客語饒平腔`
  - `客語詔安腔`
  - `客語南四縣腔`
- Old aliases like `海陸腔` or `南四縣腔` without the `客語` prefix are intentionally unsupported.
- `instruct` can be used together with `ref_audio` / `ref_text`.

## Hakka Data Export

- Main exporter: [tools/export_formospeech_hakka_jsonl.py](/home/wayne/OmniVoice/tools/export_formospeech_hakka_jsonl.py)
- The exporter supports:
  - `--text-variant raw|cln`
  - `--cln-mode restored|simple`
  - `--strip-special-tags`
- Dialect and `instruct` are inferred from `source_config` / dataset names.
- Current mappings include:
  - `Hakka_Sixian -> 客語四縣腔`
  - `Hakka_Hailu -> 客語海陸腔`
  - `Hakka_Dapu -> 客語大埔腔`
  - `Hakka_Raoping -> 客語饒平腔`
  - `Hakka_Zhaoan -> 客語詔安腔`
  - `Hakka_NanSixian -> 客語南四縣腔`

## Cleaned Text Variants

- `cln restored`
  - filters `<UNK>` / `<spn>`
  - removes `<SIL>` / `<sil>`
  - restores punctuation from raw text with partial alignment
  - keeps full-width punctuation in `text_pinyin`
  - removes spaces around punctuation in `text_pinyin`
- `cln simple`
  - filters `<UNK>` / `<spn>`
  - replaces `<SIL>` / `<sil>` with `，`
  - strips leading punctuation
  - if trailing punctuation is `，` or missing, normalizes to `。`
  - removes Hanzi spaces in `text`
  - keeps token spaces in `text_pinyin`

## Hakka Manifests And Scripts

- Standard manifests:
  - `data/formospeech_hakka/train.jsonl`
  - `data/formospeech_hakka/dev.jsonl`
  - `data/formospeech_hakka/dev_test_list.jsonl`
- Cleaned manifests:
  - `data/formospeech_hakka/train_cln.jsonl`
  - `data/formospeech_hakka/dev_cln.jsonl`
  - `data/formospeech_hakka/train_cln_simple.jsonl`
  - `data/formospeech_hakka/dev_cln_simple.jsonl`
- Related scripts/configs:
  - `examples/run_finetune.sh`
  - `examples/run_finetune_cln.sh`
  - `examples/run_finetune_cln_simple.sh`
  - `examples/run_finetune_cln_plus_denoised_reading.sh`
  - `examples/run_eval_hakka.sh`
  - `examples/run_eval_hakka_cln.sh`
  - `examples/run_eval_hakka_cln_simple.sh`
  - `examples/run_eval_hakka_cln_plus_denoised_reading.sh`
  - `examples/config/data_config_finetune_cln.json`
  - `examples/config/data_config_finetune_cln_simple.json`
  - `examples/config/data_config_finetune_cln_plus_denoised_reading.json`

## Hakka Finetune Defaults

- [examples/config/train_config_finetune.json](/home/wayne/OmniVoice/examples/config/train_config_finetune.json) uses:
  - `instruct_ratio = 1.0`
  - `only_instruct_ratio = 0.3`
  - `use_pinyin_ratio = 0.3`
- `cln` experiments that use this config are not pure text-only ablations.

## Hakka Evaluation Notes

- Main summary doc: [docs/hakka_finetune_eval.md](/home/wayne/OmniVoice/docs/hakka_finetune_eval.md)
- The doc includes comparisons across:
  - raw train
  - `cln restored`
  - `cln simple`
  - `cln restored + denoised reading`
  - dialect instruct settings with different `only_instruct_ratio`
  - dialect instruct settings with and without `use_pinyin_ratio`

## Reading Dataset Export For Sidon

- Exported reading-only manifests:
  - `data/formospeech_hakka_reading/train.jsonl`
  - `data/formospeech_hakka_reading/dev.jsonl`
- Included datasets:
  - `formospeech/hat_asr_sixian_reading_clean`
  - `formospeech/hat_asr_hailu_reading_clean`
  - `formospeech/hat_asr_nansixian_reading_clean`
- Exported counts:
  - `247118` train samples
  - `0` dev samples

## Sidon Denoise

- Sidon checkpoints downloaded to:
  - `third_party/sidon-v0.1/feature_extractor_cuda.pt`
  - `third_party/sidon-v0.1/decoder_cuda.pt`
- Generic denoise pipeline scripts:
  - `examples/run_export_formospeech_hakka_dataset.sh`
  - `examples/run_denoise_jsonl_dataset.sh`
  - `examples/prepare_single_r_upload.sh`
  - `examples/push_local_r_dataset_to_hub.sh`
- Direct denoise helper:
  - `tools/denoise_jsonl_to_audio_dir.py`
- Current direct denoise output layout:
  - `OUTPUT_DIR/audio/<sample_id>.wav`
  - `OUTPUT_DIR/train.jsonl`
  - `OUTPUT_DIR/errors.jsonl`
- `denoise_audio.py` serializes FLAC with `soundfile` instead of `torchaudio.save(..., BytesIO, format="flac")`.
- This avoids the `Couldn't allocate AVFormatContext` failure when writing denoised shards to in-memory buffers.

## Denoised Reading + CLN Pipeline

- Reading `cln restored` manifest:
  - `data/formospeech_hakka_reading/train_cln.jsonl`
- The repacked denoised dataset:
  - rewrites shard metadata to `text_variant = "cln"`
  - sets `cln_mode = "restored"`
  - keeps `special_tags_stripped = true`
  - drops samples filtered by the same `<UNK>/<spn>` logic as the main `cln` exporter
  - rewrites `data.lst` counts and durations to match the filtered dataset
- Current repacked denoised-reading sample count:
  - `243045`
- Tokenization script:
  - `examples/run_tokenize_hakka_reading_denoised_cln.sh`
- Data config for denoised + cln reading only:
  - `examples/config/data_config_hakka_reading_denoised_cln.json`
- Mixed training data config:
  - `examples/config/data_config_finetune_cln_plus_denoised_reading.json`

## Sidon Sanity Check Artifacts

- Single-example spec comparison:
  - `exp/sidon_spec_examples`
- Multi-sample sanity-check bundle:
  - `exp/sidon_sanity_check`
- The sanity-check bundle contains:
  - 9 samples total
  - 3 from Sixian
  - 3 from Hailu
  - 3 from NanSixian
- Each sample directory contains:
  - `resample24k.wav`
  - `sidon24k.wav`
  - `compare.spec.png`
- Summary files:
  - `exp/sidon_sanity_check/summary.json`
  - `exp/sidon_sanity_check/report.json`
  - `exp/sidon_sanity_check/report.md`

## Sidon Sanity-Check Findings

- Denoised outputs are being written successfully.
- In the checked samples:
  - original audio is `16 kHz`
  - denoised output is `24 kHz`
  - `dur_diff_ms = 0.0` for all 9 summarized samples
- Energy is not uniformly reduced:
  - `rms_ratio` range in current 9-sample summary: `0.4497` to `1.5427`
  - mean `rms_ratio`: `1.0442`
- Interpretation:
  - no obvious length mismatch problem in the summarized samples
  - no current write failure problem in the denoise output path
  - listening checks are still necessary because Sidon can either attenuate or amplify energy depending on the sample

## Current Best Hakka Result

- Checkpoint:
  - `exp/omnivoice_finetune_formospeech_hakka_r_mixed/checkpoint-16000`
- Condition:
  - main corpus uses `cln restored`
  - radio / TTS / elearning datasets stay on original source audio
  - reading and Sixian broadcast datasets use uploaded `-R` variants
  - training config uses `instruct_ratio = 1.0`
  - training config uses `only_instruct_ratio = 0.3`
  - training config uses `use_pinyin_ratio = 0.3`
- Reported metrics:
  - `Hakka CER (Avg of sample CERs): 2.61%`
  - `Hakka CER (Weighted): 2.29%`
  - `SIM-o score: 0.801`
  - `UTMOS score: 3.65`

## R-Mixed Hakka Result

- Checkpoint:
  - `exp/omnivoice_finetune_formospeech_hakka_r_mixed/checkpoint-5000`
  - `exp/omnivoice_finetune_formospeech_hakka_r_mixed/checkpoint-9500`
  - `exp/omnivoice_finetune_formospeech_hakka_r_mixed/checkpoint-16000`
- Condition:
  - main corpus uses `cln restored`
  - radio / TTS / elearning datasets stay on original source audio
  - reading and Sixian broadcast datasets use uploaded `-R` variants
  - training config uses `instruct_ratio = 1.0`
  - training config uses `only_instruct_ratio = 0.3`
  - training config uses `use_pinyin_ratio = 0.3`
- Reported metrics:
  - `checkpoint-5000`
    - `Hakka CER (Avg of sample CERs): 3.2%`
    - `Hakka CER (Weighted): 2.84%`
    - `SIM-o score: 0.804`
    - `UTMOS score: 3.69`
  - `checkpoint-9500`
    - `Hakka CER (Avg of sample CERs): 2.79%`
    - `Hakka CER (Weighted): 2.43%`
    - `SIM-o score: 0.800`
    - `UTMOS score: 3.66`
  - `checkpoint-16000`
    - `Hakka CER (Avg of sample CERs): 2.61%`
    - `Hakka CER (Weighted): 2.29%`
    - `SIM-o score: 0.801`
    - `UTMOS score: 3.65`
- Interpretation:
  - `checkpoint-16000` is the strongest OmniVoice Hakka result so far on CER
  - `checkpoint-5000` remains the strongest `SIM-o` point within the current `r_mixed` runs

## R-Mixed No-Broadcast Hakka Result

- Checkpoint:
  - `exp/omnivoice_finetune_formospeech_hakka_r_mixed_no_broadcast/checkpoint-5000`
  - `exp/omnivoice_finetune_formospeech_hakka_r_mixed_no_broadcast/checkpoint-9000`
  - `exp/omnivoice_finetune_formospeech_hakka_r_mixed_no_broadcast/checkpoint-16000`
- Condition:
  - main corpus uses `cln restored`
  - radio / TTS / elearning datasets stay on original source audio
  - reading datasets use uploaded `-R` variants
  - `formospeech/hat_asr_sixian_broadcast_clean_r` is excluded
  - training config uses `instruct_ratio = 1.0`
  - training config uses `only_instruct_ratio = 0.3`
  - training config uses `use_pinyin_ratio = 0.3`
- Reported metrics:
  - `checkpoint-5000`
    - `Hakka CER (Avg of sample CERs): 3.09%`
    - `Hakka CER (Weighted): 2.76%`
    - `SIM-o score: 0.792`
    - `UTMOS score: 3.67`
  - `checkpoint-9000`
    - `Hakka CER (Avg of sample CERs): 2.71%`
    - `Hakka CER (Weighted): 2.37%`
    - `SIM-o score: 0.787`
    - `UTMOS score: 3.67`
  - `checkpoint-16000`
    - `Hakka CER (Avg of sample CERs): 2.59%`
    - `Hakka CER (Weighted): 2.29%`
    - `SIM-o score: 0.786`
    - `UTMOS score: 3.67`
- Interpretation:
  - `checkpoint-16000` matches the best current CER
  - removing broadcast `-R` mainly hurts `SIM-o`
