# Hanasu

VITS2-based multilingual TTS model.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- CUDA GPU for training

## Setup

```console
uv sync
```

## Dataset Format

LJSpeech-style directory with pre-phonemized text. You are responsible for phonemizing and preprocessing your text before training.

```console
my_dataset/
  metadata.csv    # id|phonemes (pipe-separated, one line per utterance)
  wav/
    0.wav
    1.wav
    ...
```

Example `metadata.csv`:
```console
0|ʃalˈom, ʔanˈi medabˈeʁ ʔivʁˈit.
1|hɛlˈoʊ wˈɜːld.
```

## Data Preparation

```console
# 1. Normalize audio (skip if already 48kHz mono 16-bit PCM)
uv run scripts/prepare_audio.py --input raw_dataset/wav --output raw_dataset/wav_48k

# 2. Build filelists with train/val split
uv run scripts/prepare_data.py dataset/ --dataset_dir raw_dataset --wav_dir wav_48k

# 3. Pre-compute mel spectrograms
uv run scripts/prepare_mels.py dataset/ --filelist dataset/train.txt --config src/config.json
uv run scripts/prepare_mels.py dataset/ --filelist dataset/val.txt --config src/config.json
```

## Training

```console
uv run scripts/train.py -c src/config.json -m my_model
```

Checkpoints (`G_*.pth`, `D_*.pth`, `DUR_*.pth`) are saved every `eval_interval` steps (default: 1000) to `logs/my_model/`. TensorBoard logs go to the same directory. Old checkpoints are auto-cleaned based on `keep_checkpoints` in config.

## First Run Checklist

Don't walk away immediately. Verify things are healthy first:

1. Run and make sure it doesn't crash in the first epoch (bad symbols, shape mismatches, OOM)
2. Check TensorBoard after ~200 steps — losses should be going down, not NaN
3. Check mel/attention plots — attention should start forming a diagonal pattern
4. If healthy after 500-1000 steps, safe to leave overnight

Consider bumping `batch_size` (default: 2) to 16-32 depending on your GPU VRAM for faster convergence.

## Inference

```console
uv run scripts/infer.py -c src/config.json -m my_model
```

Generates `output.wav` from hardcoded phonemes using the latest checkpoint.
