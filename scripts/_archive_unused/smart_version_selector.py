# scripts/smart_version_selector.py

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

USERS_DIR = Path("users")

# ------------------ CONFIG ------------------

MIN_CONFIDENCE = 0.80      # ignore weak versions
CONF_WEIGHT = 0.6          # confidence importance
TIME_WEIGHT = 0.4          # age proximity importance

# ------------------ HELPERS ------------------

def load_user(user_id: str) -> Dict:
    path = USERS_DIR / f"{user_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"User not found: {user_id}")
    return json.loads(path.read_text())


def score_version(v: Dict, target_age: int) -> float:
    """
    Higher score = better choice
    """
    confidence = v.get("confidence", 0.0)
    age = v.get("age_at_recording")

    if age is None:
        age_penalty = 1.0
    else:
        age_penalty = 1.0 / (1.0 + abs(age - target_age))

    score = (
        CONF_WEIGHT * confidence +
        TIME_WEIGHT * age_penalty
    )
    return round(score, 4)


# ------------------ MAIN LOGIC ------------------

def select_best_version(user_id: str, target_age: int) -> Dict:
    user = load_user(user_id)
    versions = user.get("voice_versions", [])

    if not versions:
        raise ValueError("No voice versions available")

    # 1️⃣ Filter low-confidence versions
    strong_versions = [
        v for v in versions
        if v.get("confidence", 0) >= MIN_CONFIDENCE
    ]

    if not strong_versions:
        raise ValueError("No high-confidence versions available")

    # 2️⃣ Score each version
    scored = []
    for v in strong_versions:
        s = score_version(v, target_age)
        scored.append((s, v))

    # 3️⃣ Pick best
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_version = scored[0]

    return {
        "selected_version": best_version,
        "selection_score": best_score,
        "strategy": "confidence-weighted"
    }