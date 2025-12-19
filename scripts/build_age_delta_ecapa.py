
import numpy as np
import pandas as pd
from pathlib import Path
from embed_ecapa import extract_embedding

META = Path("datasets/common_voice/age_audio/all_age_metadata.csv")
OUT = Path("embeddings/age_deltas.npy")

df = pd.read_csv(META)

groups = {}
for age in df["age_group"].unique():
    embs = []
    sub = df[df["age_group"] == age]

    for p in sub["audio_path"].values[:200]:  # cap for RAM safety
        try:
            emb = extract_embedding(Path(p))
            embs.append(emb)
        except Exception:
            continue

    groups[age] = np.mean(embs, axis=0)

# ---------------- BUILD DELTAS ----------------
deltas = {
    "children_to_adult": groups["adult"] - groups["children"],
    "adult_to_children": groups["children"] - groups["adult"],
}

np.save(OUT, deltas)
print("✅ ECAPA age deltas saved")
print("Keys:", deltas.keys())
print("Delta shape:", deltas["children_to_adult"].shape)