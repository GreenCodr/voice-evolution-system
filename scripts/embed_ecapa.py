import torch
import torchaudio
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from pathlib import Path
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

def load_audio(path, target_sr=16000):
    audio, sr = torchaudio.load(path)

    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    return audio

def extract_embedding(audio_path: Path):
    signal = load_audio(audio_path)
    with torch.no_grad():
        emb = classifier.encode_batch(signal)
    emb = emb.squeeze().cpu().numpy()
    emb = emb / np.linalg.norm(emb)
    return emb

def main(args):
    audio_path = Path(args.audio)
    out_path = Path(args.out)

    emb = extract_embedding(audio_path)
    np.save(out_path, emb)

    print(f"✅ ECAPA embedding saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    main(args)