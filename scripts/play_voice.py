# scripts/play_voice.py

from scripts.voice_timeline import build_voice_timeline
from scripts.age_selector import select_voice_by_age
from scripts.playback_engine import decide_playback_mode
from scripts.age_embedding import get_embedding_for_age
from scripts.slerp import slerp
from scripts.audio_cache import cache_audio
from scripts.rate_limiter import RateLimiter
from scripts.confidence_explainer import explain_confidence
import numpy as np


rate_limiter = RateLimiter(max_calls=5, window_sec=60)


def play_voice(user_id: str, target_age: int, confidence_hint: float = 0.8):
    # 1️⃣ Rate limit
    if not rate_limiter.allow(user_id):
        return {"error": "Rate limit exceeded"}

    # 2️⃣ Timeline
    timeline = build_voice_timeline()

    # 3️⃣ Select age
    selection = select_voice_by_age(timeline, target_age)

    # 4️⃣ Decide playback mode
    decision = decide_playback_mode(selection, confidence_hint)

    # 5️⃣ Load embedding(s)
    emb, meta = get_embedding_for_age(selection)

    # 6️⃣ Interpolate if needed
    if decision["playback_type"] == "INTERPOLATED":
        e0 = emb
        e1 = emb  # placeholder (future neighbor embedding)
        emb = slerp(e0, e1, 0.5)

    # 7️⃣ Cache audio
    cached_audio, cache_hit = cache_audio(
        version_id=selection["version"]["version_id"],
        target_age=target_age,
        audio_path=selection["version"]["audio_file"]
    )

    # 8️⃣ Explain confidence
    explanation = explain_confidence(decision["confidence"])

    # 9️⃣ Final response
    return {
        "audio_path": cached_audio,
        "cache_hit": cache_hit,
        "confidence": decision["confidence"],
        "confidence_level": explanation["level"],
        "explanation": explanation["message"]
    }