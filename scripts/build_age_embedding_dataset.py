# scripts/build_age_embedding_dataset.py

import json
import csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USERS_DIR = PROJECT_ROOT / "users"
OUT_FILE = PROJECT_ROOT / "learning" / "age_embedding_dataset.csv"

OUT_FILE.parent.mkdir(exist_ok=True)

FIELDS = ["user_id", "version_id", "age_at_recording", "embedding_path"]

def main():
    rows = []

    for user_file in USERS_DIR.glob("*.json"):
        with open(user_file, encoding="utf-8") as f:
            user = json.load(f)

        user_id = user["user_id"]

        for v in user.get("voice_versions", []):
            if v.get("age_at_recording") is None:
                continue

            rows.append({
                "user_id": user_id,
                "version_id": v["version_id"],
                "age_at_recording": v["age_at_recording"],
                "embedding_path": v["embedding_path"]
            })

    if not rows:
        print("❌ No age-tagged data found")
        return

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Age-embedding dataset created")
    print(f"📄 Saved to: {OUT_FILE}")
    print(f"🔢 Samples: {len(rows)}")


if __name__ == "__main__":
    main()