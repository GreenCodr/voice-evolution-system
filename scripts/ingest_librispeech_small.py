# scripts/ingest_librispeech_small.py
import csv
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets" / "librispeech" / "LibriSpeech" / "test-clean"
OUT_CSV = PROJECT_ROOT / "data" / "librispeech_manifest_small.csv"

MAX_SPEAKERS = 3
MAX_FILES_PER_SPEAKER = 8

def main():
    if not DATA_ROOT.exists():
        print("❌ Dataset path not found:", DATA_ROOT)
        return

    rows = []
    speakers = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    random.shuffle(speakers)
    speakers = speakers[:MAX_SPEAKERS]

    for spk in speakers:
        flacs = list(spk.rglob("*.flac"))
        random.shuffle(flacs)
        flacs = flacs[:MAX_FILES_PER_SPEAKER]

        for f in flacs:
            rows.append({
                "speaker_id": spk.name,
                "file_path": str(f.relative_to(PROJECT_ROOT)),
                "dataset": "librispeech",
                "recording_date": "",
                "dob": ""
            })

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["speaker_id", "file_path", "dataset", "recording_date", "dob"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Small test manifest written: {OUT_CSV}")
    print(f"Speakers: {len(set(r['speaker_id'] for r in rows))} | Files: {len(rows)}")

if __name__ == "__main__":
    main()