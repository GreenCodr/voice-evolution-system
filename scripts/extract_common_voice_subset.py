import pandas as pd
import subprocess
from pathlib import Path

ZIP_PATH = Path("datasets/archive.zip")
CSV_PATH = Path("datasets/common_voice/age_subset.csv")
OUT_DIR = Path("datasets/common_voice/age_audio/raw")

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

print("Total rows:", len(df))

for i, row in df.iterrows():
    rel_path = f"cv-valid-train/{row['filename']}"
    out_file = OUT_DIR / Path(rel_path).name

    if out_file.exists():
        continue

    cmd = [
        "unzip",
        "-j",
        str(ZIP_PATH),
        rel_path,
        "-d",
        str(OUT_DIR),
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if (i + 1) % 100 == 0:
        print(f"Extracted {i + 1}/{len(df)}")

print("✅ Extraction complete")