# scripts/confidence_explainer.py

def explain_confidence(confidence: float) -> dict:
    if confidence >= 0.85:
        return {
            "level": "HIGH",
            "message": "This playback is based on a real recorded voice sample."
        }

    if confidence >= 0.65:
        return {
            "level": "MEDIUM",
            "message": "This voice is interpolated using nearby recordings."
        }

    return {
        "level": "LOW",
        "message": "This voice is predicted and may be less accurate."
    }