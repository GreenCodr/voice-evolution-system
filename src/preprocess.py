import os
import numpy as np
import soundfile as sf
import librosa
import webrtcvad
from pathlib import Path

def read_wav(path, sr=16000):
    audio, orig_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr, sr)
    return audio.astype("float32"), sr

def rms_normalize(audio, target_rms=0.1):
    rms = np.sqrt(np.mean(audio**2) + 1e-8)
    return np.clip(audio * (target_rms / rms), -1.0, 1.0)

def run_vad(audio, sr=16000):
    vad = webrtcvad.Vad(2)
    frame_len = int(0.03 * sr)  # 30 ms
    pcm = (audio * 32768).astype("int16")
    frames = []

    for i in range(0, len(pcm) - frame_len, frame_len):
        frame = pcm[i:i+frame_len]
        if vad.is_speech(frame.tobytes(), sr):
            frames.append(frame.astype("float32") / 32768.0)

    if len(frames) == 0:
        return audio[: int(sr * 0.5)]
    
    return np.concatenate(frames)

def preprocess(input_path, out_path):
    audio, sr = read_wav(input_path, 16000)
    voiced = run_vad(audio, sr)
    voiced = rms_normalize(voiced)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, voiced, sr)

    print("Saved preprocessed:", out_path)

if __name__ == "__main__":
    inp = "notebooks/data/test_tone.wav"
    out = "preprocessed/test_tone_preproc.wav"
    preprocess(inp, out)