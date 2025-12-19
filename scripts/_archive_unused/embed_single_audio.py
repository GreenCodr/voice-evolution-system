# scripts/embed_single_audio.py

import tempfile
import subprocess
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from pathlib import Path

# Load model once
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()


def _convert_to_wav(input_path: str) -> str:
    """
    Convert any audio (mp3/wav/etc) to clean 16kHz mono wav
    """
    tmp_wav = Path(tempfile.gettempdir()) / "tmp_clean_audio.wav"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-err_detect", "ignore_err",
            "-i", input_path,
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(tmp_wav),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    return str(tmp_wav)


def extract_embedding(audio_path: str) -> np.ndarray:
    """
    Robust embedding extraction (frontend-safe)
    """

    # ✅ Always convert first
    clean_wav = _convert_to_wav(audio_path)

    # Now soundfile is safe
    audio, sr = sf.read(clean_wav)

    if sr != 16000:
        raise RuntimeError("Sample rate mismatch after conversion")

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state.mean(dim=1)

    emb = hidden.squeeze().cpu().numpy().astype("float32")
    emb = emb / np.linalg.norm(emb)

    return emb