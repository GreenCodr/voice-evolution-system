# scripts/age_selector.py

def classify_age_relation(current_age: int, target_age: int) -> str:
    """
    Classify age relationship for playback logic
    """

    if current_age is None:
        return "UNKNOWN"

    if target_age < current_age:
        return "PAST"
    elif target_age > current_age:
        return "FUTURE"
    else:
        return "PRESENT"