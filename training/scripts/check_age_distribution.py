import os
from collections import defaultdict
from pathlib import Path

DATASET = Path("training/dataset/age_groups")

counts = defaultdict(int)

for group in DATASET.iterdir():

    if group.is_dir():

        files = list(group.glob("*.wav"))
        counts[group.name] = len(files)

print("\nAge distribution:\n")

total = 0

for age, count in sorted(counts.items()):
    print(f"{age} : {count}")
    total += count

print("\nTotal:", total)