# scripts/build_speaker_map.py

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFESTS = [
    "data/librispeech_manifest_small.csv",
]

OUTPUT_PATH = PROJECT_ROOT / "data" / "speaker_map.json"


def main():
    speaker_ids = set()

    for m in MANIFESTS:
        path = PROJECT_ROOT / m
        if not path.exists():
            continue

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                spk = row.get("speaker_id")
                if spk is not None and spk != "":
                    speaker_ids.add(str(spk))

    speaker_ids = sorted(speaker_ids)

    speaker_map = {
        spk: idx for idx, spk in enumerate(speaker_ids)
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(speaker_map, f, indent=2)

    print("✅ Speaker map built")
    print("Total speakers:", len(speaker_map))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()