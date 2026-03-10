import numpy as np
import csv
from pathlib import Path
from itertools import combinations

EMB_DIR = Path("training/dataset/embeddings")
OUTPUT_FILE = Path("training/dataset/age_pairs.csv")

def parse_name(file):
    name = file.stem
    parts = name.split("_")
    speaker = parts[0]
    age = int(parts[1])
    return speaker, age

def main():

    rows = []

    files = list(EMB_DIR.glob("*.npy"))

    speaker_map = {}

    for f in files:
        speaker, age = parse_name(f)
        speaker_map.setdefault(speaker, []).append((age, f))

    for speaker, items in speaker_map.items():

        items.sort()

        for (age1, f1), (age2, f2) in combinations(items, 2):

            rows.append([
                f1,
                age1,
                age2,
                f2
            ])

            rows.append([
                f2,
                age2,
                age1,
                f1
            ])

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "input_embedding",
            "age_input",
            "age_target",
            "target_embedding"
        ])

        writer.writerows(rows)

    print(f"Age training pairs created: {len(rows)}")

if __name__ == "__main__":
    main()