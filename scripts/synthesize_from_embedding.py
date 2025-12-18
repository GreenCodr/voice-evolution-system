# scripts/synthesize_from_embedding.py

import argparse
import hashlib
from pathlib import Path
import shutil
import soundfile as sf
from TTS.api import TTS

from rate_limiter import check_rate_limit
from structured_logger import log_event
from voice_label import write_voice_metadata

# ------------------ CACHE SETUP ------------------

CACHE_DIR = Path("cache/audio")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# ------------------ HELPERS ------------------

def make_cache_key(text: str, speaker_wav: str) -> str:
    speaker_wav = str(Path(speaker_wav).resolve())
    key = f"{MODEL_NAME}|{text}|{speaker_wav}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

# ------------------ MAIN ------------------

def main(text: str, out_path: str, speaker_wav: str):
    user_id = "user_001"
    out_path = Path(out_path)

    # -------- A3.3.1: Rate limiting --------
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

    # -------- A3.3.2: Cache lookup --------
    cache_key = make_cache_key(text, speaker_wav)
    cached_file = CACHE_DIR / f"{cache_key}.wav"

    if cached_file.exists():
        log_event("CACHE_HIT", {
            "user_id": user_id,
            "output": str(out_path),
        })

        shutil.copyfile(cached_file, out_path)

        write_voice_metadata(
            audio_path=out_path,
            voice_type="RECORDED",
            note="Served from cache"
        )

        print("⚡ Cache hit — serving cached audio")
        print(f"✅ Output (cached): {out_path}")
        return

    log_event("CACHE_MISS", {
        "user_id": user_id,
        "output": str(out_path),
    })

    # -------- Voice synthesis --------
    print("🔊 Loading XTTS model...")
    tts = TTS(model_name=MODEL_NAME)

    print("🎙️ Synthesizing voice...")
    wav = tts.tts(
        text=text,
        speaker_wav=speaker_wav,
        language="en"
    )

    # Save to cache + output
    sf.write(cached_file, wav, 24000)
    sf.write(out_path, wav, 24000)

    log_event("AUDIO_GENERATED", {
        "user_id": user_id,
        "output": str(out_path),
        "model": MODEL_NAME
    })

    write_voice_metadata(
        audio_path=out_path,
        voice_type="RECORDED",
        note="Fresh synthesis"
    )

    print(f"✅ Audio generated: {out_path}")
    print("🧠 Cached for future requests")

# ------------------ ENTRY ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("out", help="Output WAV file")
    parser.add_argument("--speaker_wav", required=True, help="Reference speaker audio")

    args = parser.parse_args()
    main(args.text, args.out, args.speaker_wav)