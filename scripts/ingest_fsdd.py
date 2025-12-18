import csv
from pathlib import Path

def main():
    base = Path("datasets/fsdd/recordings")
    if not base.exists():
        print("ERROR: datasets/fsdd/recordings not found.")
        return

    wavs = sorted(base.glob("*.wav"))
    print(f"Found {len(wavs)} audio files in FSDD")

    out_path = Path("data/fsdd_manifest.csv")
    out_path.parent.mkdir(exist_ok=True, parents=True)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_path", "speaker", "digit", "take"])

        for w in wavs:
            # filename format: DIGIT_SPEAKER_INDEX.wav
            name = w.stem  # e.g., "7_jackson_32"
            parts = name.split("_")

            if len(parts) != 3:
                continue

            digit, speaker, take = parts

            writer.writerow([str(w), speaker, digit, take])

    print(f"Manifest saved to: {out_path}")

if __name__ == "__main__":
    main()