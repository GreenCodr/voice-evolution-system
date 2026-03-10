from pathlib import Path
from collections import defaultdict

DATASET = Path("training/dataset/final_wav")

counts = defaultdict(int)

for f in DATASET.glob("*.wav"):

    name = f.name.lower()

    if "twenties" in name:
        counts["20-30"] += 1

    elif "thirties" in name:
        counts["30-40"] += 1

    elif "fourties" in name:
        counts["40-50"] += 1

    elif "fifties" in name:
        counts["50-60"] += 1

    elif "sixties" in name or "seventies" in name:
        counts["60+"] += 1

    elif "-" in name:  # children dataset
        parts = name.split("-")
        if parts[0].isdigit():
            counts["child"] += 1

print("\nDataset Age Distribution\n")

total = 0

for k, v in counts.items():
    print(k, ":", v)
    total += v

print("\nTotal:", total)