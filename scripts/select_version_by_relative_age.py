# scripts/select_version_by_relative_age.py

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = PROJECT_ROOT / "versions" / "versions_with_relative_age.csv"

if not CSV_FILE.exists():
    print("❌ versions_with_relative_age.csv not found")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage:")
    print("  python select_version_by_relative_age.py <index|fraction>")
    sys.exit(1)

query = sys.argv[1]

rows = []
with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

max_index = len(rows) - 1

# ---- Determine target index ----
if "." in query:
    # fraction mode (0.0 – 1.0)
    frac = float(query)
    frac = min(max(frac, 0.0), 1.0)
    target_index = round(frac * max_index)
else:
    # direct index
    target_index = int(query)
    target_index = min(max(target_index, 0), max_index)

# ---- Find closest ----
best = min(
    rows,
    key=lambda r: abs(int(r["relative_age_index"]) - target_index)
)

print("\n🎧 Selected Voice Version")
print("-------------------------")
print("Relative age index :", best["relative_age_index"])
print("Recorded time     :", best.get("recorded_utc") or best.get("timestamp_utc"))
print("Embedding file    :", best.get("embedding_path") or best.get("emb_file"))
print("Audio file        :", best.get("audio_path") or best.get("audio_file"))
print("Confidence        :", best["confidence"])