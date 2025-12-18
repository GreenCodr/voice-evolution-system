# scripts/migrate_versions_to_user.py

import sys
import csv
from pathlib import Path

# ------------------ PATH SETUP ------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------ IMPORTS ------------------

from scripts.user_registry import UserRegistry

# ------------------ CONSTANTS ------------------

VERSIONS_CSV = PROJECT_ROOT / "versions" / "versions.csv"
USER_ID = "user_001"   # single-user system for now

# ------------------ MAIN ------------------

def main():
    if not VERSIONS_CSV.exists():
        print("❌ versions.csv not found")
        return

    user = UserRegistry(USER_ID)

    with open(VERSIONS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user.add_voice_version(
                version_id=row.get("version_id"),
                embedding_path=row.get("embedding_path") or row.get("emb_file"),
                audio_path=row.get("audio_path") or row.get("audio_file"),
                confidence=float(row.get("confidence", 0)),
                voice_type="RECORDED",
                recorded_utc=row.get("recorded_utc") or row.get("timestamp_utc"),
            )

    print(f"✅ Migrated versions into users/{USER_ID}.json")

# ------------------ ENTRY ------------------

if __name__ == "__main__":
    main()