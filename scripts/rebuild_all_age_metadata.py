import pandas as pd
from pathlib import Path

# Adult data
ADULT_META = Path("datasets/common_voice/age_audio/features/age_features.csv")

# Children raw audio folder
CHILD_RAW = Path("datasets/common_voice/age_audio_children/raw")

rows = []

# ---------- ADULT ----------
adult_df = pd.read_csv(ADULT_META)

for _, r in adult_df.iterrows():
    rows.append({
        "audio_path": r["audio_path"],
        "age_group": r["age_group"],
        "source": "adult"
    })

# ---------- CHILDREN ----------
child_files = list(CHILD_RAW.glob("*.wav"))[:220]

for p in child_files:
    rows.append({
        "audio_path": str(p),
        "age_group": "children",
        "source": "children"
    })

df = pd.DataFrame(rows)

OUT = Path("datasets/common_voice/age_audio/all_age_metadata.csv")
df.to_csv(OUT, index=False)

print("✅ Metadata rebuilt")
print(df["age_group"].value_counts())
