# scripts/playback_decision.py

def decide_playback_mode(selection, confidence):
    """
    Returns playback metadata for UI / API
    """

    mode = selection["mode"]

    if mode == "EXACT":
        return {
            "playback_type": "REAL",
            "confidence": round(confidence, 3),
            "explanation": "Exact age recording found"
        }

    if mode in ("PAST_ONLY", "FUTURE_ONLY"):
        return {
            "playback_type": "REAL",
            "confidence": round(confidence * 0.8, 3),
            "explanation": "Closest available age used"
        }

    if mode == "INTERPOLATE":
        return {
            "playback_type": "PREDICTED",
            "confidence": round(confidence * 0.65, 3),
            "explanation": "Voice interpolated using SLERP"
        }

    return {
        "playback_type": "UNKNOWN",
        "confidence": 0.0,
        "explanation": "Insufficient data"
    }