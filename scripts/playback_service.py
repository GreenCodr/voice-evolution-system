# scripts/playback_service.py

from pathlib import Path

from scripts.hybrid_playback_decider import decide_playback_mode
from scripts.synthesize_from_embedding import synthesize_from_embedding

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def play_voice(user_id: str, target_age: int, text: str) -> dict:
    """
    Single backend entry point for frontend playback
    """

    decision = decide_playback_mode(user_id, target_age)

    mode = decision.get("mode")

    # ---------------- RECORDED ----------------
    if mode == "RECORDED":
        version = decision.get("version") or decision.get("base_version")

        if not version:
            return {
                "mode": "ERROR",
                "reason": "Recorded version missing"
            }

        return {
            "mode": "RECORDED",
            "audio_path": version["audio_path"],
            "reason": "Using real recorded voice"
        }

    # ---------------- AGED ----------------
    if mode == "AGED":
        base_version = decision.get("base_version")

        if not base_version:
            return {
                "mode": "ERROR",
                "reason": "Base version missing for aging"
            }

        out_path = OUTPUT_DIR / f"{user_id}_aged_{target_age}.wav"

        synthesize_from_embedding(
            text=text,
            out_path=str(out_path),
            speaker_embedding=decision["embedding"],
            reference_wav=base_version["audio_path"],
        )

        return {
            "mode": "AGED",
            "audio_path": str(out_path),
            "alpha": decision.get("alpha"),
            "relation": decision.get("relation"),
            "reason": "Age-evolved voice generated"
        }

    # ---------------- NONE / FALLBACK ----------------
    return {
        "mode": "NONE",
        "reason": decision.get("reason", "No voice available")
    }