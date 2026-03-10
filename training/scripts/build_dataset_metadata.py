import pandas as pd
from pathlib import Path

FINAL_DIR = Path("training/dataset/final_wav")
CHILD_META = "training/dataset/children/samromur_children_21.09/metadata.tsv"

rows = []

# -------- Load children metadata --------
child_df = pd.read_csv(CHILD_META, sep="\t")

age_map = {}

for _, r in child_df.iterrows():

    fname = r["filename"].replace(".flac", ".wav")
    age = str(r["age"])

    if age.isdigit():
        age = int(age)

        if age <= 10:
            group = "5-10"
        elif age <= 15:
            group = "10-15"
        elif age <= 20:
            group = "15-20"
        else:
            continue

        age_map[fname] = group

# -------- Process all files --------

for f in FINAL_DIR.glob("*.wav"):

    name = f.name

    # children dataset
    if name in age_map:
        rows.append([name, age_map[name]])

    # common voice dataset
    elif "twenties" in name:
        rows.append([name, "20-30"])

    elif "thirties" in name:
        rows.append([name, "30-40"])

    elif "fourties" in name:
        rows.append([name, "40-50"])

    elif "fifties" in name:
        rows.append([name, "50-60"])

    elif "sixties" in name or "seventies" in name:
        rows.append([name, "60+"])

df = pd.DataFrame(rows, columns=["file", "age_group"])

df.to_csv("training/dataset/dataset_metadata.csv", index=False)

print("Saved dataset_metadata.csv")
print("Total labeled files:", len(df))