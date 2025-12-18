from datetime import datetime, date
import csv
from pathlib import Path
import time

from config_loader import CONFIG
from structured_logger import log_event

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_FILE = PROJECT_ROOT / "versions" / "versions.csv"

# ------------------ CONFIG-DRIVEN THRESHOLDS ------------------

SIM_REJECT_HARD = CONFIG["speaker_verification"]["similarity_reject_hard"]
SIM_NO_CHANGE = CONFIG["speaker_verification"]["similarity_no_change"]

CONF_REJECT = CONFIG["confidence"]["reject_below"]
CONF_CREATE = CONFIG["confidence"]["create_above"]

DEVICE_MATCH_MIN = CONFIG["device"]["min_match_score"]

MIN_DAYS_BETWEEN_VERSIONS = CONFIG.get("versioning", {}).get(
    "min_days_between_versions", 30
)

# ------------------------------------------------------------


def calculate_age(dob: str, recording_date: date) -> int | None:
    if not dob:
        return None

    birth = datetime.strptime(dob, "%Y-%m-%d").date()
    age = recording_date.year - birth.year
    if (recording_date.month, recording_date.day) < (birth.month, birth.day):
        age -= 1
    return age


def days_since_last_version() -> int | None:
    if not VERSIONS_FILE.exists():
        return None

    try:
        with open(VERSIONS_FILE, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if not rows:
                return None

            last = rows[-1]
            last_date = datetime.fromisoformat(last["recorded_utc"]).date()
            return (datetime.utcnow().date() - last_date).days
    except Exception:
        return None


def decide_voice_version(
    similarity: float,
    confidence: float,
    speaker_ok: bool,
    device_match: float,
    embedding_path: str,
    audio_path: str | None,
    user_dob: str | None,
):
    """
    Production-grade voice version decision with:
    - config-driven thresholds
    - structured logging
    - age tagging
    - time-gap protection
    """

    recording_date = datetime.utcnow().date()
    age_at_recording = calculate_age(user_dob, recording_date)

    # ---------- HARD REJECTS ----------

    if not speaker_ok:
        log_event("VERSION_REJECTED", {
            "reason": "speaker_verification_failed",
            "similarity": similarity,
            "confidence": confidence
        })
        return _reject("Speaker verification failed", similarity, confidence)

    if similarity < SIM_REJECT_HARD:
        log_event("VERSION_REJECTED", {
            "reason": "low_similarity",
            "similarity": similarity,
            "confidence": confidence
        })
        return _reject("Similarity below threshold", similarity, confidence)

    if confidence < CONF_REJECT:
        log_event("VERSION_REJECTED", {
            "reason": "low_confidence",
            "similarity": similarity,
            "confidence": confidence
        })
        return _reject("Low confidence", similarity, confidence)

    # ---------- SAME VOICE ----------

    if similarity >= SIM_NO_CHANGE:
        log_event("NO_NEW_VERSION", {
            "reason": "voice_stable",
            "similarity": similarity,
            "confidence": confidence,
            "age_at_recording": age_at_recording
        })
        return {
            "action": "NO_NEW_VERSION",
            "reason": "Voice stable",
            "confidence": round(confidence, 3),
            "similarity": round(similarity, 4),
            "age_at_recording": age_at_recording,
        }

    # ---------- TIME GAP CHECK ----------

    gap_days = days_since_last_version()
    if gap_days is not None and gap_days < MIN_DAYS_BETWEEN_VERSIONS:
        log_event("VERSION_REJECTED", {
            "reason": "min_days_not_elapsed",
            "gap_days": gap_days,
            "similarity": similarity,
            "confidence": confidence
        })
        return {
            "action": "REJECT",
            "reason": f"Only {gap_days} days since last version",
            "confidence": round(confidence, 3),
            "similarity": round(similarity, 4),
            "age_at_recording": age_at_recording,
        }

    # ---------- VOICE EVOLUTION ----------

    if confidence >= CONF_CREATE and device_match >= DEVICE_MATCH_MIN:
        record = create_version_record(
            embedding_path,
            audio_path,
            confidence,
            similarity,
            age_at_recording,
            recording_date
        )
        write_version(record)

        log_event("VERSION_CREATED", {
            "version_id": record["version_id"],
            "similarity": similarity,
            "confidence": confidence,
            "age_at_recording": age_at_recording,
            "embedding_path": embedding_path
        })

        return {
            "action": "CREATE_VERSION",
            "record": record
        }

    # ---------- FALLBACK ----------

    log_event("VERSION_REJECTED", {
        "reason": "gray_zone",
        "similarity": similarity,
        "confidence": confidence
    })

    return _reject("Gray zone, insufficient confidence", similarity, confidence)


# ------------------ HELPERS ------------------

def _reject(reason: str, similarity: float, confidence: float):
    return {
        "action": "REJECT",
        "reason": reason,
        "confidence": round(confidence, 3),
        "similarity": round(similarity, 4),
    }


def create_version_record(
    embedding_path: str,
    audio_path: str | None,
    confidence: float,
    similarity: float,
    age_at_recording: int | None,
    recording_date: date,
):
    return {
        "version_id": int(time.time()),
        "recorded_utc": recording_date.isoformat(),
        "age_at_recording": age_at_recording,
        "embedding_path": embedding_path,
        "audio_path": audio_path,
        "confidence": round(confidence, 3),
        "similarity": round(similarity, 4),
        "notes": "auto-created"
    }


def write_version(record: dict):
    VERSIONS_FILE.parent.mkdir(exist_ok=True)
    file_exists = VERSIONS_FILE.exists()

    with open(VERSIONS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)