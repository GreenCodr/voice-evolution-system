# scripts/audio_cache.py
from pathlib import Path
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "cache" / "audio"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def cache_key(version_id, target_age):
    raw = f"{version_id}_{target_age}".encode()
    return hashlib.md5(raw).hexdigest()

def cache_audio(version_id, target_age, audio_path):
    key = cache_key(version_id, target_age)
    cached = CACHE_DIR / f"{key}.wav"

    if cached.exists():
        return cached, True  # cache hit

    cached.write_bytes(Path(audio_path).read_bytes())
    return cached, False