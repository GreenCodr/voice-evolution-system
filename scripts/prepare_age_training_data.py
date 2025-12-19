import pandas as pd
from pathlib import Path

IN = Path("datasets/common_voice/age_audio/features/age_features.csv")
OUT = Path("datasets/common_voice/age_audio/features/age_training.csv")

AGE_MAP = {
    "children": 10,
    "teens": 16,
    "twenties": 25,
    "thirties": 35,
    "fourties": 45,
    "fifties": 55,
    "sixties": 65,
    "seventies": 75,
    "eighties": 85,
}

df = pd.read_csv(IN)

df["age_numeric"] = df["age_group"].map(AGE_MAP)

df = df.dropna(subset=["age_numeric"])

df.to_csv(OUT, index=False)

print("✅ Age training data prepared")
print("Rows:", len(df))
print(df["age_numeric"].value_counts().sort_index())
