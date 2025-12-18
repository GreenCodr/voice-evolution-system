# scripts/device_fingerprint.py
import soundfile as sf
from pathlib import Path

def extract_device_fingerprint(audio_path: str) -> dict:
    audio_path = Path(audio_path)
    info = sf.info(audio_path)

    duration_bucket = round(info.duration, 1)

    fingerprint = {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,   # PCM_16, PCM_24, FLOAT
        "duration_bucket": duration_bucket
    }

    return fingerprint


def device_match_score(fp_new: dict, fp_ref: dict) -> float:
    """
    Returns score between 0–1
    """
    score = 0
    total = 4

    score += int(fp_new["sample_rate"] == fp_ref["sample_rate"])
    score += int(fp_new["channels"] == fp_ref["channels"])
    score += int(fp_new["subtype"] == fp_ref["subtype"])
    score += int(abs(fp_new["duration_bucket"] - fp_ref["duration_bucket"]) < 0.5)

    return score / total