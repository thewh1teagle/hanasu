"""
Normalize audio files for training: resample to 48kHz, mono, 16-bit PCM WAV.

Step 1 of 3 in the data preparation pipeline:

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
from multiprocessing import Pool

import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm

try:
    import librosa
except ImportError:
    print("librosa is required: uv add librosa")
    raise


def process_file(args):
    input_path, output_path, target_sr, overwrite = args

    if not overwrite and os.path.exists(output_path):
        return "skipped"

    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
    audio_16bit = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, target_sr, audio_16bit)
    return "done"


def main():
    parser = argparse.ArgumentParser(description="Normalize audio to 48kHz mono 16-bit PCM WAV")
    parser.add_argument("--input", type=str, required=True, help="Input directory with wav files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for normalized wavs")
    parser.add_argument("--sample_rate", type=int, default=48000, help="Target sample rate (default: 48000)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess existing files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    wav_files = [f for f in os.listdir(args.input) if f.endswith(".wav")]
    if not wav_files:
        print(f"No wav files found in {args.input}")
        return

    work = [
        (
            os.path.join(args.input, f),
            os.path.join(args.output, f),
            args.sample_rate,
            args.overwrite,
        )
        for f in wav_files
    ]

    workers = args.workers or os.cpu_count()
    with Pool(workers) as pool:
        results = list(tqdm(
            pool.imap(process_file, work),
            total=len(work),
            desc="Normalizing audio",
        ))

    done = results.count("done")
    skipped = results.count("skipped")
    print(f"Processed {done} files, skipped {skipped} (already exist)")


if __name__ == "__main__":
    main()
