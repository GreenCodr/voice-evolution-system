# scripts/confidence_engine.py

def compute_confidence(
    duration_s: float,
    snr_db: float | None,
    speaker_similarity: float,
    device_match: float,
    history_count: int,
    min_duration: float = 10.0,
    min_snr: float = 20.0,
):
    
    """
    Returns confidence score in [0, 1]
    """

    score = 0.0

    # 1. Duration (25%)
    dur_score = min(duration_s / min_duration, 1.0)
    score += 0.25 * dur_score

    # 2. SNR (20%)
    if snr_db is None:
        snr_score = 0.0
    else:
        snr_score = min(snr_db / min_snr, 1.0)
    score += 0.20 * snr_score

    # 3. Speaker similarity (30%)
    speaker_score = max(0.0, min(speaker_similarity, 1.0))
    score += 0.30 * speaker_score

    # 4. Device match (15%)
    score += 0.15 * device_match

    # 5. Temporal consistency (10%)
    # (Phase-1 placeholder: assume stable timeline)
    # 5. Temporal consistency (10%)
    if history_count >= 3:
        temporal_score = 1.0
    elif history_count == 2:
        temporal_score = 0.6
    elif history_count == 1:
        temporal_score = 0.3
    else:
        temporal_score = 0.0

    score += 0.10 * temporal_score

    return round(score, 3)