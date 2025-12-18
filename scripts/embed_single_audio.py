import argparse
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base-960h"

def load_audio(path, target_sr=16000):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio.astype("float32"), sr, target_sr)
        sr = target_sr

    return audio, sr

def main(audio_path, out_path, device):
    audio_path = Path(audio_path)
    out_path = Path(out_path)

    print("Using device:", device)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    audio, sr = load_audio(audio_path, processor.feature_extractor.sampling_rate)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)

    emb = torch.nn.functional.normalize(emb, dim=0)
    np.save(out_path, emb.cpu().numpy())

    print("✅ Reference embedding saved to:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    main(args.audio, args.out, args.device)