# scripts/smart_version_selector.py

from typing import List, Dict


def select_best_version(*, versions: List[Dict], target_age: int) -> Dict:
    """
    Phase-2 smart selector:
    - Prefer real recorded voices close to target age
    - Fallback to generated voice if none exist

    Parameters
    ----------
    versions : list of voice version dicts
    target_age : int

    Returns
    -------
    dict with playback decision
    """

    if not versions:
        return {
            "mode": "GENERATED",
            "reason": "no_recorded_versions"
        }

    best = None
    best_gap = float("inf")

    for v in versions:
        age = v.get("age_at_recording")

        # Skip versions without age info
        if age is None:
            continue

        gap = abs(age - target_age)

        if gap < best_gap:
            best_gap = gap
            best = v

    # ✅ Rule: within 5 years → use recorded voice
    if best and best_gap <= 5:
        return {
            "mode": "RECORDED",
            "version": best,
            "age_gap": best_gap
        }

    # ❌ Otherwise → generated voice
    return {
        "mode": "GENERATED",
        "reason": "no_close_real_voice",
        "closest_gap": best_gap if best else None
    }