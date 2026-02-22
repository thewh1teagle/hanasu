"""Minimal inference script.

Usage:
    uv run scripts/infer.py -c src/config.json -m my_model
"""

import torch
from . import utils
from .models import SynthesizerTrn, inference
from .text import symbols

PHONEMES = "ʃalˈom, ʔanˈi medabˈeʁ ʔivʁˈit."

def main():
    hps = utils.get_hparams()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net_g = SynthesizerTrn(
        len(symbols),
        128,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    net_g.eval()

    ckpt = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
    if ckpt is None:
        print(f"No checkpoint found in {hps.model_dir}")
        return
    utils.load_checkpoint(ckpt, net_g, None)
    print(f"Loaded {ckpt}")

    inference(model=net_g, text=PHONEMES, device=device, output_file="output.wav")
    print("Saved output.wav")

if __name__ == "__main__":
    main()
