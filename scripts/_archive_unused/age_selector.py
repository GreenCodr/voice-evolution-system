# scripts/age_selector.py
from typing import List, Dict, Optional


def select_voice_by_age(
    timeline: List[Dict],
    target_age: int
) -> Dict:
    """
    Select best voice version for requested age.

    Strategy:
    1. Exact age match → return
    2. Nearest past age
    3. Nearest future age
    4. If two sides exist → return both (for interpolation)
    """

    valid = [v for v in timeline if v["age"] is not None]

    if not valid:
        return {
            "mode": "NO_DATA",
            "reason": "No age-labelled voice data available"
        }

    # exact match
    for v in valid:
        if v["age"] == target_age:
            return {
                "mode": "EXACT",
                "version": v
            }

    # split past / future
    past = [v for v in valid if v["age"] < target_age]
    future = [v for v in valid if v["age"] > target_age]

    past.sort(key=lambda x: x["age"], reverse=True)
    future.sort(key=lambda x: x["age"])

    if past and future:
        return {
            "mode": "INTERPOLATE",
            "left": past[0],
            "right": future[0],
            "alpha": (target_age - past[0]["age"]) /
                     (future[0]["age"] - past[0]["age"])
        }

    if past:
        return {
            "mode": "PAST_ONLY",
            "version": past[0]
        }

    if future:
        return {
            "mode": "FUTURE_ONLY",
            "version": future[0]
        }

    return {
        "mode": "NO_MATCH"
    }