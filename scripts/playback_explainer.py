# scripts/playback_explainer.py

def explain_playback(decision: dict) -> dict:
    mode = decision.get("mode")

    if mode == "RECORDED":
        return {
            "label": "RECORDED",
            "icon": "✅",
            "message": "This is a recorded voice from the selected time."
        }

    if mode == "INTERPOLATED":
        return {
            "label": "INTERPOLATED",
            "icon": "🟡",
            "message": "This voice is interpolated between two recorded versions."
        }

    if mode == "PREDICTED":
        return {
            "label": "PREDICTED",
            "icon": "⚠️",
            "message": "This voice is predicted and may not exactly match reality."
        }

    return {
        "label": "UNKNOWN",
        "icon": "❓",
        "message": "Voice type could not be determined."
    }