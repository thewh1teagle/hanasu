"""
Pre-compute mel spectrograms for training.

Reads a filelist (from prepare_data.py) and a model config, computes mel
spectrograms in parallel, and writes a jsonl manifest.

Step 3 of 3 in the data preparation pipeline:

    # 1. Normalize audio (skip if already 48kHz mono 16-bit)
    uv run scripts/prepare_audio.py --input raw_dataset/wav --output raw_dataset/wav_48k

    # 2. Build filelists with train/val split
    uv run scripts/prepare_data.py dataset/ --dataset_dir raw_dataset --wav_dir wav_48k

    # 3. Pre-compute mel spectrograms
    uv run scripts/prepare_mels.py dataset/ --filelist dataset/train.txt --config src/config.json
    uv run scripts/prepare_mels.py dataset/ --filelist dataset/val.txt --config src/config.json
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import torch
from scipy.io.wavfile import read
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.mel_processing import mel_spectrogram_torch


def compute_mel(args):
    idx, audio_path, mel_dir, mel_params, overwrite = args
    mel_path = os.path.join(mel_dir, f"{idx}.mel.pt")

    if not overwrite and os.path.exists(mel_path):
        return mel_path

    sampling_rate, data = read(audio_path)
    if sampling_rate != mel_params["sampling_rate"]:
        raise ValueError(f"{audio_path}: {sampling_rate} SR doesn't match target {mel_params['sampling_rate']}")

    audio = torch.FloatTensor(data.astype("float32"))
    audio_norm = (audio / mel_params["max_wav_value"]).unsqueeze(0)

    mel = mel_spectrogram_torch(
        audio_norm,
        mel_params["filter_length"],
        mel_params["n_mel_channels"],
        mel_params["sampling_rate"],
        mel_params["hop_length"],
        mel_params["win_length"],
        mel_params["mel_fmin"],
        mel_params["mel_fmax"],
        center=False,
    )
    mel = mel.squeeze(0)
    torch.save(mel, mel_path)
    return mel_path


def main():
    parser = argparse.ArgumentParser(description="Pre-compute mel spectrograms")
    parser.add_argument("output_dir", type=str, help="Output directory for mels and jsonl")
    parser.add_argument("--filelist", type=str, required=True, help="Input filelist (from prepare_data.py)")
    parser.add_argument("--config", type=str, default="src/config.json", help="Model config with mel params")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing mels")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    mel_params = {
        "sampling_rate": config["data"]["sampling_rate"],
        "filter_length": config["data"]["filter_length"],
        "hop_length": config["data"]["hop_length"],
        "win_length": config["data"]["win_length"],
        "n_mel_channels": config["data"]["n_mel_channels"],
        "mel_fmin": config["data"]["mel_fmin"],
        "mel_fmax": config["data"]["mel_fmax"],
        "max_wav_value": config["data"]["max_wav_value"],
    }

    # Read filelist
    entries = []
    with open(args.filelist, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            audio_path, speaker_id, phonemes = parts
            entries.append((audio_path, int(speaker_id), phonemes))

    # Create output dirs
    mel_dir = os.path.join(args.output_dir, "mels")
    os.makedirs(mel_dir, exist_ok=True)

    # Build work items
    work = [
        (i, entry[0], mel_dir, mel_params, args.overwrite)
        for i, entry in enumerate(entries)
    ]

    # Compute mels in parallel
    workers = args.workers or os.cpu_count()
    with Pool(workers) as pool:
        mel_paths = list(tqdm(
            pool.imap(compute_mel, work),
            total=len(work),
            desc="Computing mels",
        ))

    # Write jsonl
    filelist_stem = Path(args.filelist).stem
    jsonl_path = os.path.join(args.output_dir, f"{filelist_stem}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(entries):
            audio_path, speaker_id, phonemes = entry
            record = {
                "audio": os.path.abspath(audio_path),
                "mel": os.path.relpath(mel_paths[i], args.output_dir),
                "speaker_id": speaker_id,
                "phonemes": phonemes,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} mels to {mel_dir}/")
    print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    main()
