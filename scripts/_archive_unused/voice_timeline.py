# scripts/voice_timeline.py
import csv
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_CSV = PROJECT_ROOT / "versions" / "versions.csv"


def compute_age(dob_iso, rec_iso):
    try:
        dob = datetime.fromisoformat(dob_iso).date()
        rec = datetime.fromisoformat(rec_iso.replace("Z", "")).date()
        age = rec.year - dob.year
        if (rec.month, rec.day) < (dob.month, dob.day):
            age -= 1
        return age
    except Exception:
        return None


def build_voice_timeline():
    if not VERSIONS_CSV.exists():
        raise FileNotFoundError("versions.csv not found")

    timeline = []

    with open(VERSIONS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            age = r.get("age_at_recording")

            if not age and r.get("dob") and r.get("timestamp_utc"):
                age = compute_age(r["dob"], r["timestamp_utc"])

            timeline.append({
                "version_id": r["version_id"],
                "timestamp": r["timestamp_utc"],
                "age": int(age) if age else None,
                "emb_file": r["emb_file"],
                "audio_file": r["audio_file"],
                "confidence": float(r.get("confidence", 0.0))
            })

    # sort by time
    timeline.sort(key=lambda x: x["timestamp"])

    return timeline


if __name__ == "__main__":
    timeline = build_voice_timeline()

    print("\nVOICE TIMELINE")
    print("-" * 40)
    for t in timeline:
        print(
            f"Age: {t['age']}, "
            f"Time: {t['timestamp']}, "
            f"Confidence: {t['confidence']:.2f}"
        )