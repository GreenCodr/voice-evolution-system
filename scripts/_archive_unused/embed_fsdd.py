# scripts/embed_fsdd.py

import argparse
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FSDD_AUDIO_DIR = PROJECT_ROOT / "datasets/fsdd/recordings"
FSDD_EMB_DIR = PROJECT_ROOT / "datasets/fsdd/embeddings"

MODEL_NAME = "facebook/wav2vec2-base-960h"
TARGET_SR = 16000


def load_audio(path, target_sr=TARGET_SR):
    audio, sr = sf.read(path)

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample (FIXED)
    if sr != target_sr:
        audio = librosa.resample(
            y=audio.astype("float32"),
            orig_sr=sr,
            target_sr=target_sr
        )
        sr = target_sr

    return audio, sr


def main(device="cpu"):
    FSDD_EMB_DIR.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(FSDD_AUDIO_DIR.glob("*.wav"))
    print(f"Found {len(wav_files)} FSDD files")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    for wav in wav_files:
        try:
            audio, sr = load_audio(wav)

            inputs = processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                outputs = model(inputs.input_values.to(device))
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            emb = emb / np.linalg.norm(emb)

            out_path = FSDD_EMB_DIR / f"{wav.stem}_wav2vec_emb.npy"
            np.save(out_path, emb)

        except Exception as e:
            print(f"❌ Failed on {wav.name}: {e}")

    print("✅ FSDD embeddings created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.device)