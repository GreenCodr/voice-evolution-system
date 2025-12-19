# scripts/synthesize_from_embedding.py

import argparse
import hashlib
import shutil
from pathlib import Path

import soundfile as sf
from TTS.api import TTS

from scripts.rate_limiter import check_rate_limit
from scripts.structured_logger import log_event
from scripts.audio_cache import get_cached_audio

# ------------------ CONSTANTS ------------------

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24000


# ================== PROGRAMMATIC API ==================

def synthesize_from_embedding(
    text: str,
    out_path: str,
    speaker_embedding,
    reference_wav: str,
):
    """
    Programmatic API used by backend / playback engine
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cached_file = get_cached_audio(
        text=text,
        speaker_wav=reference_wav,
        speaker_embedding=speaker_embedding,
    )

    shutil.copyfile(cached_file, out_path)
    return out_path


# ================== INTERNAL SYNTHESIS ==================

def _synthesize_and_cache(text: str, speaker_wav: str, cache_path: Path):
    """
    Internal XTTS synthesis (single place)
    """

    print("🔊 Loading XTTS model...")
    tts = TTS(model_name=MODEL_NAME)

    print("🎙️ Synthesizing voice...")
    wav = tts.tts(
        text=text,
        speaker_wav=speaker_wav,
        language="en"
    )

    sf.write(cache_path, wav, SAMPLE_RATE)
    return wav


# ================== CACHE HELPERS ==================

CACHE_DIR = Path("cache/audio")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def make_cache_key(text: str, speaker_wav: str) -> str:
    speaker_wav = str(Path(speaker_wav).resolve())
    key = f"{MODEL_NAME}|{text}|{speaker_wav}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ================== CLI ENTRY ==================

def main(text: str, out_path: str, speaker_wav: str):
    user_id = "user_001"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -------- Rate limiting --------
    rate = check_rate_limit(user_id)
    if not rate["allowed"]:
        log_event("PLAYBACK_BLOCKED", {
            "user_id": user_id,
            "reason": "rate_limit",
            "reset_in_sec": rate["reset_in_sec"]
        })
        print("❌ Rate limit exceeded")
        print(f"⏳ Try again in {rate['reset_in_sec']} seconds")
        return

    # -------- Cache lookup --------
    cache_key = make_cache_key(text, speaker_wav)
    cached_file = CACHE_DIR / f"{cache_key}.wav"

    if cached_file.exists():
        log_event("CACHE_HIT", {
            "user_id": user_id,
            "output": str(out_path),
        })
        shutil.copyfile(cached_file, out_path)
        print("⚡ Cache hit — serving cached audio")
        print(f"✅ Output (cached): {out_path}")
        return

    log_event("CACHE_MISS", {
        "user_id": user_id,
        "output": str(out_path),
    })

    # -------- Synthesis --------
    wav = _synthesize_and_cache(text, speaker_wav, cached_file)

    sf.write(out_path, wav, SAMPLE_RATE)

    log_event("AUDIO_GENERATED", {
        "user_id": user_id,
        "output": str(out_path),
        "model": MODEL_NAME
    })

    print(f"✅ Audio generated: {out_path}")
    print("🧠 Cached for future requests")


# ================== ENTRY POINT ==================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("out", help="Output WAV file")
    parser.add_argument("--speaker_wav", required=True, help="Reference speaker audio")

    args = parser.parse_args()
    main(args.text, args.out, args.speaker_wav)