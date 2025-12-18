# scripts/simulate_new_embedding.py
import numpy as np
from pathlib import Path
from datetime import datetime
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT / "embeddings"
MANIFEST_EMB = ROOT / "data" / "manifest_emb.csv"

# name for simulated embedding
new_name = "simulated_new_emb.npy"
new_path = EMB_DIR / new_name

# load an existing embedding to perturb
# choose one existing file (first .npy in embeddings/)
existing = None
for p in EMB_DIR.glob("*.npy"):
    existing = p
    break

if existing is None:
    print("No existing embedding found in embeddings/. Please run embed step first.")
    sys.exit(2)

emb = np.load(existing).astype("float32")

# add noise scaled so cosine sim drops below 0.75 (adjust if needed)
rng = np.random.RandomState(1234)
noise = rng.normal(scale=0.25, size=emb.shape).astype("float32")
sim = emb + noise

# save simulated embedding
EMB_DIR.mkdir(parents=True, exist_ok=True)
np.save(new_path, sim)
print("Wrote simulated embedding:", new_path)

# Append a row to data/manifest_emb.csv pointing to the new embedding and same audio
# Use existing manifest row as a template (if present)
rows = []
fields = None
if MANIFEST_EMB.exists():
    with open(MANIFEST_EMB, newline="", encoding="utf-8") as fh:
        import csv
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        for r in reader:
            rows.append(r)
else:
    print("Manifest", MANIFEST_EMB, "not found. Creating a minimal manifest.")
    fields = ["file_path","speaker_id","dob","recording_date","dataset_source","notes","preproc_path","emb_path"]

# Build a new manifest row using first existing row if any
new_row = {}
if rows:
    template = rows[0]
    new_row.update(template)
    # set emb_path to our new file (relative)
    new_row["emb_path"] = str(Path("embeddings") / new_name)
    # set recording_date to now (ISO)
    new_row["recording_date"] = datetime.utcnow().isoformat()
    # keep dob same if available else leave blank
else:
    # minimal
    new_row = {
        "file_path": "preprocessed/test_tone_preproc.wav",
        "speaker_id": "spk_test_001",
        "dob": "1995-01-01",
        "recording_date": datetime.utcnow().isoformat(),
        "dataset_source": "simulated",
        "notes": "simulated embedding",
        "preproc_path": "preprocessed/test_tone_preproc.wav",
        "emb_path": str(Path("embeddings") / new_name),
    }
    # ensure fields defined
    fields = list(new_row.keys())

# Append the row to manifest_emb.csv (if file exists keep header fields)
# if fields don't match exactly, write a simple CSV with the right columns
out_rows = rows + [new_row]
with open(MANIFEST_EMB, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    for r in out_rows:
        writer.writerow(r)

print("Appended simulated entry to:", MANIFEST_EMB)
print("Done.")