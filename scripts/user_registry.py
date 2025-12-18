# scripts/user_registry.py

import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional

# ------------------ PATH ------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USERS_DIR = PROJECT_ROOT / "users"
USERS_DIR.mkdir(exist_ok=True)

# ------------------ USER REGISTRY ------------------

class UserRegistry:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.user_file = USERS_DIR / f"{user_id}.json"

        if self.user_file.exists():
            self._load()
        else:
            self.data = {
                "user_id": user_id,
                "date_of_birth": None,
                "created_utc": datetime.utcnow().isoformat() + "Z",
                "registered_devices": [],
                "voice_versions": []
            }
            self._save()

    # ------------------ CORE ------------------

    def _load(self):
        with open(self.user_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def _save(self):
        with open(self.user_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    # ------------------ DOB ------------------

    def set_date_of_birth(self, dob: str):
        """
        dob format: YYYY-MM-DD
        """
        self.data["date_of_birth"] = dob
        self._save()

    def calculate_age(self, recording_date: Optional[str] = None) -> Optional[int]:
        if not self.data["date_of_birth"]:
            return None

        dob = datetime.strptime(self.data["date_of_birth"], "%Y-%m-%d").date()

        if recording_date:
            rec = datetime.fromisoformat(recording_date.replace("Z", "")).date()
        else:
            rec = date.today()

        age = rec.year - dob.year
        if (rec.month, rec.day) < (dob.month, dob.day):
            age -= 1
        return age

    # ------------------ DEVICES ------------------

    def register_device(self, device_id: str, fingerprint: dict):
        for d in self.data["registered_devices"]:
            if d["device_id"] == device_id:
                return  # already registered

        self.data["registered_devices"].append({
            "device_id": device_id,
            "fingerprint": fingerprint,
            "first_seen_utc": datetime.utcnow().isoformat() + "Z"
        })
        self._save()

    # ------------------ VOICE VERSIONS ------------------

    def add_voice_version(
        self,
        version_id: str,
        embedding_path: str,
        audio_path: str,
        confidence: float,
        voice_type: str = "RECORDED",
        recorded_utc: Optional[str] = None
    ):
        if not recorded_utc:
            recorded_utc = datetime.utcnow().isoformat() + "Z"

        age = self.calculate_age(recorded_utc)

        self.data["voice_versions"].append({
            "version_id": version_id,
            "recorded_utc": recorded_utc,
            "age_at_recording": age,
            "embedding_path": embedding_path,
            "audio_path": audio_path,
            "confidence": round(confidence, 3),
            "type": voice_type
        })

        self._save()

    # ------------------ READ HELPERS ------------------

    def get_versions(self):
        return self.data["voice_versions"]

    def get_latest_version(self):
        if not self.data["voice_versions"]:
            return None
        return sorted(
            self.data["voice_versions"],
            key=lambda v: v["recorded_utc"]
        )[-1]