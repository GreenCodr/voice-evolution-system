# core/user_store.py

import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USERS_DIR = PROJECT_ROOT / "users"
USERS_DIR.mkdir(exist_ok=True)


class UserStore:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.path = USERS_DIR / f"{user_id}.json"

        if self.path.exists():
            self._load()
        else:
            self.data = {
                "user_id": user_id,
                "date_of_birth": None,
                "created_utc": self._now(),
                "voice_versions": []
            }
            self._save()

    # ---------- internal ----------

    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _load(self):
        self.data = json.loads(self.path.read_text())

    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

    # ---------- DOB ----------

    def set_dob(self, dob: str):
        """
        dob format: YYYY-MM-DD
        """
        self.data["date_of_birth"] = dob
        self._save()

    def calculate_age(self, at_time: Optional[str] = None) -> Optional[int]:
        if not self.data["date_of_birth"]:
            return None

        birth = datetime.strptime(
            self.data["date_of_birth"], "%Y-%m-%d"
        ).date()

        ref = (
            datetime.fromisoformat(at_time.replace("Z", "")).date()
            if at_time else date.today()
        )

        age = ref.year - birth.year
        if (ref.month, ref.day) < (birth.month, birth.day):
            age -= 1
        return age

    # ---------- Voice Versions ----------

    def add_voice_version(
        self,
        audio_path: str,
        confidence: float,
        voice_type: str = "RECORDED"
    ):
        recorded_utc = self._now()
        age = self.calculate_age(recorded_utc)

        self.data["voice_versions"].append({
            "version_id": f"v{len(self.data['voice_versions']) + 1}",
            "recorded_utc": recorded_utc,
            "age_at_recording": age,
            "audio_path": audio_path,
            "confidence": round(confidence, 3),
            "type": voice_type
        })

        self._save()

    # ---------- Read ----------

    def versions(self) -> List[Dict[str, Any]]:
        return self.data["voice_versions"]

    def latest_version(self) -> Optional[Dict[str, Any]]:
        if not self.data["voice_versions"]:
            return None
        return self.data["voice_versions"][-1]