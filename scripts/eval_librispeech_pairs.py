import csv
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data/librispeech_manifest_small_emb.csv"

def load_emb(p):
    e = np.load(p).astype("float32")
    return e / np.linalg.norm(e)

def decide(sim):
    if sim >= 0.92:
        return "NO_NEW_VERSION"
    elif sim >= 0.80:
        return "CREATE_NEW_VERSION"
    else:
        return "REJECT"

# ---------------- LOAD MANIFEST ----------------
rows = []
with open(MANIFEST, newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

# Group by speaker
by_speaker = {}
for r in rows:
    spk = r["speaker_id"]
    by_speaker.setdefault(spk, []).append(r)

print("\n===== SAME SPEAKER TESTS =====")
for spk, items in by_speaker.items():
    if len(items) < 2:
        continue
    a, b = items[0], items[1]
    e1 = load_emb(PROJECT_ROOT / a["emb_path"])
    e2 = load_emb(PROJECT_ROOT / b["emb_path"])
    sim = float(np.dot(e1, e2))
    print(f"Speaker {spk} | sim={sim:.4f} | {decide(sim)}")

print("\n===== DIFFERENT SPEAKER TESTS =====")
spks = list(by_speaker.keys())
for i in range(min(5, len(spks)-1)):
    a = by_speaker[spks[i]][0]
    b = by_speaker[spks[i+1]][0]
    e1 = load_emb(PROJECT_ROOT / a["emb_path"])
    e2 = load_emb(PROJECT_ROOT / b["emb_path"])
    sim = float(np.dot(e1, e2))
    print(f"{a['speaker_id']} vs {b['speaker_id']} | sim={sim:.4f} | {decide(sim)}")