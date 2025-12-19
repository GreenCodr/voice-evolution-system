import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
from pathlib import Path
import numpy as np
from scripts.age_features import extract_age_features

META = Path("datasets/common_voice/age_audio/all_age_metadata.csv")
OUT_DIR = Path("datasets/common_voice/age_audio/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(META)

rows = []

print("Total samples:", len(df))

for i, row in df.iterrows():
    audio_path = Path(row["audio_path"])
    age_group = row["age_group"]
    source = row["source"]

    if not audio_path.exists():
        continue

    try:
        feats = extract_age_features(str(audio_path))
        feats["audio_path"] = str(audio_path)
        feats["age_group"] = age_group
        feats["source"] = source
        rows.append(feats)
    except Exception as e:
        print("❌ Failed:", audio_path.name, e)

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(df)}")

out_df = pd.DataFrame(rows)
out_path = OUT_DIR / "age_features.csv"
out_df.to_csv(out_path, index=False)

print("\n✅ FEATURE EXTRACTION COMPLETE")
print("Saved:", out_path)
print("Final rows:", len(out_df))
print(out_df["age_group"].value_counts())
