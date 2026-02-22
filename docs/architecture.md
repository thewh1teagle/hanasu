# Architecture

VITS2-based TTS. Single forward pass from phonemes to waveform.

## Model (`src/models.py` — `SynthesizerTrn`)

- **TextEncoder**: phoneme token IDs → prior stats `(m_p, logs_p)`
- **PosteriorEncoder**: mel → latent posterior `(z, m_q, logs_q)` (training only)
- **Flow**: maps between posterior and prior latent space `(z <-> z_p)`, uses transformer flows (VITS2)
- **Duration Predictor**: stochastic, estimates per-token durations. Regularized by `DurationDiscriminatorV2` (VITS2)
- **MAS**: monotonic alignment search aligns tokens to mel frames during training. Uses `monotonic-alignment-search` package
- **Generator**: HiFi-GAN decoder, latent → waveform

## Training (`src/train.py`)

- Dataset loader reads jsonl manifest (from `prepare_mels.py`), loads pre-computed mels + raw audio
- Phonemes → token IDs via `cleaned_text_to_sequence`
- Losses: adversarial (generator + discriminator), feature matching, mel L1, KL divergence, duration (+ duration adversarial)
- DDP + mixed precision

## Inference (`src/models.py` — `inference()`, `src/infer.py`)

- Input: phoneme string (already phonemized)
- Phonemes → TextEncoder → predict durations → expand to frame-level → sample latent → invert flow → decode waveform
- Controls: `noise_scale`, `length_scale`, `noise_scale_w`, `temperature`, `duration_blur_sigma`
- Per-token duration multipliers and noise profiles for prosody control (`build_duration_multipliers_from_ids`, `build_noise_profile_from_ids`)

## Key Files

- `src/text.py`: symbol table, phoneme↔ID conversion
- `src/models.py`: model architecture, inference
- `src/train.py`: dataset loader, training loop, losses
- `src/mel_processing.py`: mel spectrogram computation
- `src/utils.py`: config, checkpoints, logging
- `src/infer.py`: minimal inference script

## Speaker Configuration

- `n_speakers: 0` — single speaker, speaker conditioning auto-disabled
- `n_speakers: 1` — single speaker but with speaker embedding (allows future multi-speaker fine-tuning)
- `n_speakers: N` — multi-speaker with N embeddings

`use_spk_conditioned_encoder` and `gin_channels` are ignored when `n_speakers: 0`.

## Notes for Agents

- Main class: `SynthesizerTrn` in `src/models.py`
- Prosody controls are in `SynthesizerTrn.infer()` kwargs
- If changing losses, update both discriminator and generator sections in `train_and_evaluate()`
