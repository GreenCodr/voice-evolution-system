# scripts/hybrid_playback_decider.py

import sys
from pathlib import Path
import numpy as np

# ------------------ PATH FIX ------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------ IMPORTS ------------------
from scripts.user_registry import UserRegistry
from scripts.smart_version_selector import select_best_version
from scripts.age_selector import classify_age_relation

# ------------------ CONSTANTS ------------------
AGE_DELTAS_PATH = PROJECT_ROOT / "embeddings" / "age_deltas.npy"
EMB_DIR = PROJECT_ROOT / "versions" / "embeddings"


def decide_playback_mode(user_id: str, target_age: int) -> dict:
    """
    Phase-2 playback decision logic
    """

    user = UserRegistry(user_id)
    versions = user.get_versions()

    if not versions:
        return {"mode": "NONE", "reason": "no_voice_versions"}

    # 1️⃣ Select best recorded version
    selection = select_best_version(
        versions=versions,
        target_age=target_age
    )

    # ---- RECORDED PATH ----
    if selection["mode"] == "RECORDED":
        return {
            "mode": "RECORDED",
            "version": selection["version"],
            "reason": "real_voice_close_to_target",
            "age_gap": selection.get("age_gap"),
        }

    # ---- AGED PATH ----
    base_version = user.get_latest_version()
    if not base_version or not base_version.get("embedding_path"):
        return {"mode": "NONE", "reason": "no_embedding_available"}

    base_age = base_version.get("age_at_recording")

    relation = classify_age_relation(base_age, target_age)
    if relation == "same":
        return {
            "mode": "RECORDED",
            "version": base_version,
            "reason": "same_age_requested",
        }

    # Load base embedding
    base_emb = np.load(PROJECT_ROOT / base_version["embedding_path"])
    base_emb /= np.linalg.norm(base_emb)

    # ✅ Load age deltas (FIXED)
    age_deltas = np.load(AGE_DELTAS_PATH, allow_pickle=True).item()

    delta_key = (
        "children_to_adult"
        if relation == "future"
        else "adult_to_children"
    )

    if delta_key not in age_deltas:
        return {"mode": "NONE", "reason": f"missing_delta:{delta_key}"}

    delta = age_deltas[delta_key]

    years = abs((base_age or target_age) - target_age)
    alpha = min(years / 40.0, 1.0)

    aged_emb = base_emb + alpha * delta
    aged_emb /= np.linalg.norm(aged_emb)

    return {
    "mode": "AGED",
    "embedding": aged_emb,
    "base_version": base_version,   # ✅ REQUIRED
    "target_age": target_age,
    "alpha": round(alpha, 2),
    "relation": relation,
    "reason": "age_delta_applied"
}