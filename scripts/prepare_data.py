"""
Build train/val filelists from an LJSpeech-style dataset.

Reads metadata.csv (id|phonemes, already phonemized) and produces
train/val filelists in the format: audio_path|speaker_id|phonemes

Step 2 of 3 in the data preparation pipeline:

    # 1. Normalize audio (skip if already 48kHz mono 16-bit)
    uv run scripts/prepare_audio.py --input raw_dataset/wav --output raw_dataset/wav_48k

    # 2. Build filelists with train/val split
    uv run scripts/prepare_data.py dataset/ --dataset_dir raw_dataset --wav_dir wav_48k

    # 3. Pre-compute mel spectrograms
    uv run scripts/prepare_mels.py dataset/ --filelist dataset/train.txt --config src/config.json
    uv run scripts/prepare_mels.py dataset/ --filelist dataset/val.txt --config src/config.json
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.text import _symbol_to_id


def main():
    parser = argparse.ArgumentParser(description="Prepare training filelist from LJSpeech-style dataset")
    parser.add_argument("output_dir", type=str, help="Output directory (e.g. dataset/)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory containing metadata.csv and wav/")
    parser.add_argument("--val_count", type=int, default=20, help="Number of entries to hold out for validation")
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker ID for single-speaker dataset")
    parser.add_argument("--wav_dir", type=str, default="wav", help="Subdirectory containing wav files")
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Metadata filename")
    parser.add_argument("--no_filter_symbols", action="store_true", help="Disable filtering entries with unknown symbols")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible train/val split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.txt")
    val_path = os.path.join(args.output_dir, "val.txt")

    metadata_path = os.path.join(args.dataset_dir, args.metadata)
    wav_dir = os.path.join(args.dataset_dir, args.wav_dir)

    entries = []
    skipped_missing, skipped_symbols = 0, 0

    with open(metadata_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            utterance_id, phonemes = parts

            wav_path = os.path.join(wav_dir, f"{utterance_id}.wav")
            if not os.path.isfile(wav_path):
                skipped_missing += 1
                continue

            if not args.no_filter_symbols:
                unknown = [c for c in phonemes if c not in _symbol_to_id]
                if unknown:
                    skipped_symbols += 1
                    print(f"Skipping {utterance_id}: unknown symbols {set(unknown)}")
                    continue

            entries.append(f"{wav_path}|{args.speaker_id}|{phonemes}")

    # Reproducible train/val split
    random.Random(args.seed).shuffle(entries)
    val_count = min(args.val_count, len(entries))
    val_entries = entries[:val_count]
    train_entries = entries[val_count:]

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in train_entries)

    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in val_entries)

    print(f"Train: {len(train_entries)} entries -> {train_path}")
    print(f"Val:   {len(val_entries)} entries -> {val_path}")
    if skipped_missing:
        print(f"Skipped {skipped_missing} entries (missing wav)")
    if skipped_symbols:
        print(f"Skipped {skipped_symbols} entries (unknown symbols)")


if __name__ == "__main__":
    main()
