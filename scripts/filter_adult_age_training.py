# scripts/filter_adult_age_training.py

import pandas as pd
from pathlib import Path

IN = Path("datasets/common_voice/age_audio/features/age_features.csv")
OUT = Path("datasets/common_voice/age_audio/features/age_features_adult.csv")

print("Loading features:", IN)

df = pd.read_csv(IN)

print("Before filtering:")
print(df["age_group"].value_counts())

# -------------------------------------------------
# KEEP ONLY ADULT AGE GROUPS
# -------------------------------------------------
adult_df = df[df["age_group"] == "adult"].copy()

print("\nAfter filtering (ADULT ONLY):")
print(adult_df["age_group"].value_counts())

# -------------------------------------------------
# SAVE
# -------------------------------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
adult_df.to_csv(OUT, index=False)

print("\n✅ Adult-only training data saved")
print("Output file:", OUT)
print("Total samples:", len(adult_df))