import pandas as pd
from pathlib import Path

META = Path("datasets/common_voice/age_audio/all_age_metadata.csv")

df = pd.read_csv(META)

adult = df[df["age_group"] == "adult"]
children = df[df["age_group"] == "children"].sample(n=220, random_state=42)

final = pd.concat([adult, children]).reset_index(drop=True)

final.to_csv(META, index=False)

print("✅ Deduplication complete")
print(final["age_group"].value_counts())
print("Total samples:", len(final))
