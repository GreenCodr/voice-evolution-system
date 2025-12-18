# scripts/list_versions_by_time.py

import csv
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_FILE = PROJECT_ROOT / "versions" / "versions.csv"

if not VERSIONS_FILE.exists():
    print("❌ versions.csv not found")
    exit(1)

rows = []

with open(VERSIONS_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        ts = r.get("timestamp_utc") or r.get("recorded_utc")
        if not ts:
            continue
        try:
            r["_parsed_time"] = datetime.fromisoformat(ts.replace("Z", ""))
            rows.append(r)
        except Exception:
            continue

if not rows:
    print("❌ No valid timestamps found")
    exit(0)

rows.sort(key=lambda x: x["_parsed_time"])

print("\n📜 Voice Versions (Chronological Order)\n")

for i, r in enumerate(rows):
    print(
        f"{i+1:02d}. "
        f"time={r['_parsed_time']} | "
        f"age={r.get('age_at_recording') or 'UNKNOWN'} | "
        f"confidence={r.get('confidence')}"
    )