# src/embeddings_wav2vec.py
import os
import numpy as np
import torch
from pathlib import Path
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2Processor

def load_audio(path, target_sr=16000):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr

def compute_wav2vec_embedding(wav_path, model_name="facebook/wav2vec2-base-960h", device="cpu"):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    audio, sr = load_audio(wav_path, target_sr=processor.feature_extractor.sampling_rate)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=False, return_dict=True)
        hidden = outputs.last_hidden_state.squeeze(0)  # (T, H)
        emb = hidden.mean(dim=0).cpu().numpy()
    return emb

if __name__ == "__main__":
    inp = "preprocessed/test_tone_preproc.wav"
    out_dir = "embeddings"
    os.makedirs(out_dir, exist_ok=True)
    device = "mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else "cpu"
    print("Computing Wav2Vec2 embedding for:", inp, "on device:", device)
    emb = compute_wav2vec_embedding(inp, device=device)
    print("Embedding shape:", emb.shape)
    print("First 8 values:", emb[:8].tolist())
    out_path = os.path.join(out_dir, Path(inp).stem + "_wav2vec_emb.npy")
    np.save(out_path, emb)
    print("Saved embedding to:", out_path)