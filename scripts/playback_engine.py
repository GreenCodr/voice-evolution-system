# scripts/playback_engine.py

def decide_playback_mode(selection, confidence, min_confidence=0.6):
    """
    Decide whether to play REAL voice or PREDICTED (interpolated) voice
    """

    if selection["mode"] == "EXACT":
        return {
            "playback_type": "REAL",
            "confidence": round(confidence, 3),
            "explanation": "Exact age match"
        }

    if selection["mode"] in ("PAST_ONLY", "FUTURE_ONLY"):
        if confidence >= min_confidence:
            return {
                "playback_type": "REAL",
                "confidence": round(confidence * 0.8, 3),
                "explanation": "Closest available age used"
            }
        else:
            return {
                "playback_type": "UNAVAILABLE",
                "confidence": round(confidence, 3),
                "explanation": "Insufficient confidence"
            }

    if selection["mode"] == "INTERPOLATE":
        if confidence >= min_confidence:
            return {
                "playback_type": "PREDICTED",
                "confidence": round(confidence * 0.9, 3),
                "explanation": "Interpolated using SLERP"
            }
        else:
            return {
                "playback_type": "UNAVAILABLE",
                "confidence": round(confidence, 3),
                "explanation": "Prediction confidence too low"
            }

    return {
        "playback_type": "UNAVAILABLE",
        "confidence": 0.0,
        "explanation": "Unknown state"
    }