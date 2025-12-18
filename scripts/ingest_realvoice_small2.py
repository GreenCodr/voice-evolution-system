import csv
from datasets import load_dataset
from pathlib import Path

OUT = Path("data/realvoice_manifest_small2.csv")

def main():
    print("Downloading small real-voice dataset (non-gated)...")

    ds = load_dataset(
        "Shahules/Arabic-accented-english-speech",
        split="train"
    )

    samples = []
    for i, ex in enumerate(ds):
        if i >= 25:
            break

        audio = ex["audio"]
        speaker = ex.get("speaker_id", "unknown")

        samples.append({
            "file_path": audio["path"],
            "speaker_id": speaker,
            "age": ""   # no age → synthetic ages later
        })

    OUT.parent.mkdir(exist_ok=True, parents=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_path", "speaker_id", "age"])
        w.writeheader()
        w.writerows(samples)

    print(f"\nManifest saved to: {OUT}")
    print("Total samples extracted:", len(samples))

if __name__ == "__main__":
    main()