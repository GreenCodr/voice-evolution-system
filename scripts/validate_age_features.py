# scripts/validate_age_features.py

import pandas as pd
import numpy as np
from pathlib import Path

FEATURES_PATH = Path("datasets/common_voice/age_audio/features/age_features.csv")

if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Missing file: {FEATURES_PATH}")

df = pd.read_csv(FEATURES_PATH)

print("✅ Loaded features")
print("Total samples:", len(df))
print(df["age_group"].value_counts())
print()

# ---------------- Feature sanity ----------------

feature_cols = [
    "mean_pitch",
    "pitch_std",
    "spectral_centroid",
    "spectral_rolloff",
    "rms_energy",
    "speaking_rate",
]

print("🔍 Feature statistics (mean ± std)")
for col in feature_cols:
    mean = df[col].mean()
    std = df[col].std()
    print(f"{col:20s}: mean={mean:.3f}, std={std:.3f}")

print("\n🔎 NaN check")
print(df[feature_cols].isna().sum())

print("\n🔁 Range check (min → max)")
for col in feature_cols:
    print(f"{col:20s}: {df[col].min():.3f} → {df[col].max():.3f}")

print("\n✅ Feature validation complete")