import numpy as np
import pandas as pd
from pathlib import Path

FEATURES = Path("datasets/common_voice/age_audio/features/age_features.csv")
OUT = Path("embeddings/age_deltas.npy")

df = pd.read_csv(FEATURES)

# Split groups
children = df[df["age_group"] == "children"]
adult = df[df["age_group"] == "adult"]

FEATURE_COLS = [
    "mean_pitch",
    "pitch_std",
    "spectral_centroid",
    "spectral_rolloff",
    "rms_energy",
    "speaking_rate"
]

# Compute centroids
child_centroid = children[FEATURE_COLS].mean().values
adult_centroid = adult[FEATURE_COLS].mean().values

# Age delta (GLOBAL)
age_deltas = {
    "children_to_adult": adult_centroid - child_centroid,
    "adult_to_children": child_centroid - adult_centroid
}

np.save(OUT, age_deltas)

print("✅ Age delta embeddings rebuilt")
print("Keys:", age_deltas.keys())