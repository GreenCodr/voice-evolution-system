# scripts/assign_relative_age.py

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_FILE = PROJECT_ROOT / "versions" / "versions.csv"
OUT_FILE = PROJECT_ROOT / "versions" / "versions_with_relative_age.csv"

rows = []

with open(VERSIONS_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

if len(rows) < 2:
    print("❌ Need at least 2 versions for relative age mapping")
    exit(0)

# Assign relative ages
for i, r in enumerate(rows):
    r["relative_age_index"] = i  # 0 = earliest

# Save new CSV
with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Relative age assigned (0 → {len(rows)-1})")
print(f"📄 Saved to: {OUT_FILE}")