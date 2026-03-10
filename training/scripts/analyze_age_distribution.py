from pathlib import Path
import pandas as pd
from collections import Counter

dataset = Path("training/dataset/all_wav")

counter = Counter()

for f in dataset.glob("*.wav"):
    name = f.name.lower()

    if "twenties" in name:
        counter["20-30"] += 1
    elif "thirties" in name:
        counter["30-40"] += 1
    elif "fourties" in name:
        counter["40-50"] += 1
    elif "fifties" in name:
        counter["50-60"] += 1
    elif "sixties" in name:
        counter["60+"] += 1
    elif "seventies" in name:
        counter["70+"] += 1
    elif "teens" in name:
        counter["15-20"] += 1
    else:
        counter["child"] += 1

print("\nAge distribution:\n")
for k,v in counter.items():
    print(k, ":", v)