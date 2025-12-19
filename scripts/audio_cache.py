# scripts/audio_cache.py

import hashlib
from pathlib import Path
import shutil
import soundfile as sf
from TTS.api import TTS

# ------------------ CONSTANTS ------------------

CACHE_DIR = Path("cache/audio")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24000


# ------------------ CACHE KEY ------------------

def make_cache_key(text: str, speaker_wav: str) -> str:
    speaker_wav = str(Path(speaker_wav).resolve())
    key = f"{MODEL_NAME}|{text}|{speaker_wav}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ------------------ MAIN API ------------------

def get_cached_audio(
    text: str,
    speaker_wav: str,
    speaker_embedding=None,   # kept for future use
) -> Path:
    """
    Unified audio cache + synthesis entry.
    Returns path to cached WAV.
    """

    cache_key = make_cache_key(text, speaker_wav)
    cached_file = CACHE_DIR / f"{cache_key}.wav"

    if cached_file.exists():
        print("⚡ Cache hit — serving cached audio")
        return cached_file

    print("🔊 Cache miss — synthesizing")

    tts = TTS(model_name=MODEL_NAME)

    wav = tts.tts(
        text=text,
        speaker_wav=speaker_wav,
        language="en"
    )

    sf.write(cached_file, wav, SAMPLE_RATE)
    print("🧠 Audio cached")

    return cached_file