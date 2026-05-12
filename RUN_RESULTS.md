# Colab Run Results

This file records the successful Colab run for the speech alignment and codec tokenization pipeline.

## Project Claim Covered

Speech Alignment and Tokenization Pipeline for TTS

- Built a preprocessing pipeline that converts raw audio-transcript pairs into word-aligned speech segments using Wav2Vec2 CTC emissions and forced alignment.
- Tokenized aligned speech clips with an EnCodec-style RVQ codec to produce discrete audio tokens, waveform reconstructions, and metadata for downstream TTS/ASR training.

No model training was performed in this run. Wav2Vec2 and EnCodec were used as pretrained components. The implemented work is the data preparation and orchestration pipeline around alignment, segmentation, tokenization, reconstruction, and metadata generation.

## Main Portfolio Run

| Field | Value |
| --- | --- |
| Run name | `portfolio_t4_20260511_185544` |
| Runtime | Colab T4 |
| Notebook | `notebooks/voice_ai_colab.ipynb` |
| Dataset | `openslr/librispeech_asr` |
| Dataset config | `clean` |
| Dataset split | `validation` |
| Dataset loading | Streaming |
| Wav2Vec2 model | `facebook/wav2vec2-base-960h` |
| Codec model | `encodec_24khz` |
| EnCodec bandwidth | `6.0 kbps` |
| Max runtime budget | `5.0 hours` |
| Actual elapsed runtime | `0.0467 hours` |
| Completed | `true` |

## Main Results

| Metric | Value |
| --- | ---: |
| Utterances requested | 300 |
| Utterances seen | 300 |
| Utterances aligned | 300 |
| Utterances failed | 0 |
| Segments extracted | 514 |
| Segments tokenized | 514 |
| Reconstructions saved | 514 |
| Total segment duration | 1643.62 sec |
| Total segment duration | 27.39 min |
| Average segment duration | 3.1977 sec |
| Average alignment confidence | 0.831545 |
| Minimum segment duration | 0.5006 sec |
| Maximum segment duration | 5.6526 sec |
| Stopped due to time budget | false |
| Stopped due to disk budget | false |
| Local disk free at summary | 69.104 GB |

## Output Artifacts

The run wrote artifacts under:

```text
/content/drive/MyDrive/audioai/alignment_codec_runs/portfolio_t4_20260511_185544
```

Expected artifact groups:

- `alignments/*.json`: per-utterance forced alignment outputs with token spans, word spans, timing, confidence, and emission metadata.
- `segments/wav/*.wav`: aligned speech clips cut from the source audio.
- `segments/meta/*.json`: per-segment metadata with transcript text, source row, timing, duration, padding, and confidence.
- `segments/metadata.jsonl`: combined segment manifest.
- `codec/tokens/*.pt`: EnCodec RVQ token tensors and token metadata.
- `codec/reconstructions/*.wav`: decoded waveform reconstructions from the codec tokens.
- `codec/meta/*.json`: per-segment codec metadata.
- `codec/metadata.jsonl`: combined codec manifest.
- `reports/summary.json`: run-level metrics.
- `reports/samples/`: sample plots and audio examples.

## Smoke Run

The smoke run completed before the full portfolio run:

| Field | Value |
| --- | --- |
| Run name | `smoke_t4_20260511_184332` |
| Runtime | Colab T4 |
| Dataset split | `validation` |
| Dataset loading | Streaming |
| Completed | `true` |
| Actual elapsed runtime | `0.0096 hours` |

Smoke metrics:

| Metric | Value |
| --- | ---: |
| Utterances requested | 10 |
| Utterances seen | 10 |
| Utterances aligned | 10 |
| Utterances failed | 0 |
| Segments extracted | 17 |
| Segments tokenized | 17 |
| Reconstructions saved | 17 |
| Average alignment confidence | 0.856774 |
| Stopped due to time budget | false |
| Stopped due to disk budget | false |

## Notes

- The first failed Colab attempt used non-streaming dataset loading and spent time materializing large LibriSpeech splits before reaching alignment. That run did not validate the pipeline because it produced zero alignments and zero tokenized segments.
- The successful notebook uses streaming dataset loading and Drive-backed Hugging Face/Torch caches.
- Hugging Face authentication warnings were informational only. The public dataset and model were accessible without `HF_TOKEN`.
- The Wav2Vec2 `masked_spec_embed` missing-key warning is expected for inference use here and did not affect forced alignment.
