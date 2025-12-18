# scripts/hybrid_playback_decider.py

from datetime import datetime
from typing import Dict, List
import json
from pathlib import Path

USERS_DIR = Path("users")

# ------------------ CONFIG ------------------

MAX_DIRECT_GAP_YEARS = 1.0     # close enough → recorded
MAX_INTERP_GAP_YEARS = 6.0     # mid-range → interpolation

# ------------------ HELPERS ------------------

def load_user(user_id: str) -> Dict:
    path = USERS_DIR / f"{user_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"User not found: {user_id}")
    return json.loads(path.read_text())


def years_between(a: int, b: int) -> float:
    return abs(a - b)


# ------------------ MAIN LOGIC ------------------

def decide_playback_mode(user_id: str, target_age: int) -> Dict:
    user = load_user(user_id)
    versions = user.get("voice_versions", [])

    if not versions:
        return {
            "mode": "NONE",
            "reason": "No voice data available"
        }

    # Keep only versions with age info
    aged = [
        v for v in versions
        if v.get("age_at_recording") is not None
    ]

    if not aged:
        return {
            "mode": "PREDICTED",
            "reason": "No age-tagged versions"
        }

    # Sort by age
    aged.sort(key=lambda v: v["age_at_recording"])

    # Find nearest versions
    below = [v for v in aged if v["age_at_recording"] <= target_age]
    above = [v for v in aged if v["age_at_recording"] >= target_age]

    nearest = min(
        aged,
        key=lambda v: years_between(v["age_at_recording"], target_age)
    )

    gap = years_between(nearest["age_at_recording"], target_age)

    # -------- Decision Tree --------

    # 1️⃣ Direct recorded
    if gap <= MAX_DIRECT_GAP_YEARS:
        return {
            "mode": "RECORDED",
            "version": nearest,
            "reason": "Exact or very close age match"
        }

    # 2️⃣ Interpolation possible
    if below and above:
        lower = below[-1]
        upper = above[0]

        span = years_between(
            lower["age_at_recording"],
            upper["age_at_recording"]
        )

        if span <= MAX_INTERP_GAP_YEARS:
            return {
                "mode": "INTERPOLATED",
                "lower": lower,
                "upper": upper,
                "reason": "Interpolating between two nearby versions"
            }

    # 3️⃣ Fallback to prediction
    return {
        "mode": "PREDICTED",
        "nearest": nearest,
        "reason": "Outside known voice range"
    }