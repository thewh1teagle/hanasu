# Data Pipeline

3 scripts, run in order. Each step's output feeds the next.

## 1. Normalize Audio (`scripts/prepare_audio.py`)

Resamples to 48kHz, mono, 16-bit PCM WAV. Skip if audio is already in this format.

```console
uv run scripts/prepare_audio.py --input raw_dataset/wav --output raw_dataset/wav_48k
```

**Input**: directory of wav files (any format)
**Output**: directory of normalized wav files

## 2. Build Filelists (`scripts/prepare_data.py`)

Reads `metadata.csv` (`id|phonemes`), resolves audio paths, splits into train/val.

```console
uv run scripts/prepare_data.py dataset/ --dataset_dir raw_dataset --wav_dir wav_48k
```

**Input**: `metadata.csv` + wav directory
**Output**: `dataset/train.txt`, `dataset/val.txt` (pipe-separated: `audio_path|speaker_id|phonemes`)

Filters entries with unknown symbols by default. Reproducible split via `--seed`.

## 3. Pre-compute Mels (`scripts/prepare_mels.py`)

Computes mel spectrograms in parallel, writes jsonl manifest for training.

```console
uv run scripts/prepare_mels.py dataset/ --filelist dataset/train.txt --config src/config.json
uv run scripts/prepare_mels.py dataset/ --filelist dataset/val.txt --config src/config.json
```

**Input**: filelist + model config
**Output**:
```console
dataset/
  train.jsonl       # {"audio": "/abs/path.wav", "mel": "mels/0.mel.pt", "speaker_id": 0, "phonemes": "..."}
  val.jsonl
  mels/
    0.mel.pt
    1.mel.pt
    ...
```

Training reads the jsonl files directly (`training_files` / `validation_files` in `src/config.json`).

## Format Requirements

- **Audio**: 48kHz, mono, 16-bit PCM WAV
- **Metadata**: `id|phonemes` — pipe-separated, already phonemized (IPA)
- **Phonemes**: must only contain symbols from the symbol table in `src/text.py`
