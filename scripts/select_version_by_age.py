import csv
import sys
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_FILE = PROJECT_ROOT / "versions" / "versions.csv"


def select_version_by_age(target_age: int):
    if not VERSIONS_FILE.exists():
        print("❌ No versions found")
        return

    versions = []

    with open(VERSIONS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("age_at_recording"):
                versions.append({
                    "version_id": row["version_id"],
                    "age": int(row["age_at_recording"]),
                    "embedding": row["embedding_path"],
                })

    if not versions:
        print("❌ No age-tagged versions available")
        return

    versions.sort(key=lambda x: x["age"])

    # ---- Exact match ----
    for v in versions:
        if v["age"] == target_age:
            print("✅ EXACT MATCH FOUND")
            print(v)
            return

    # ---- Nearest versions ----
    lower = None
    upper = None

    for v in versions:
        if v["age"] < target_age:
            lower = v
        elif v["age"] > target_age and upper is None:
            upper = v

    print("🟡 NO EXACT MATCH — USING NEAREST VERSIONS")
    print("Lower:", lower)
    print("Upper:", upper)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python select_version_by_age.py <target_age>")
        sys.exit(1)

    target_age = int(sys.argv[1])
    select_version_by_age(target_age)