# scripts/validate_age_progression.py

import sys
from pathlib import Path
import torch
import numpy as np
from itertools import combinations

# ---------------- PATH SETUP ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------- IMPORTS ----------------

from scripts.phase3_model import AgeSpeakerModel
from scripts.phase3_dataset import UnifiedVoiceDataset

# ---------------- CONFIG ----------------

MANIFESTS = ["data/librispeech_manifest_small.csv"]
CHECKPOINT_PATH = "models/age_embedding_v1.pt"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# ---------------- LOAD MODEL ----------------

ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

# speaker head not used → set dummy value
model = AgeSpeakerModel(
    num_speakers=1,
    device=device
)

model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

print("✅ Model loaded")

# ---------------- LOAD DATA ----------------

dataset = UnifiedVoiceDataset(
    manifest_paths=MANIFESTS,
    max_samples=None
)

# Group samples by speaker
speaker_groups = {}
for s in dataset:
    spk = s["speaker_idx"]
    age = s["age"]
    if age is None:
        continue
    speaker_groups.setdefault(spk, []).append(s)

# Pick one speaker with multiple samples
speaker_id = None
for k, v in speaker_groups.items():
    if len(v) >= 3:
        speaker_id = k
        samples = v
        break

if speaker_id is None:
    print("❌ No speaker with enough age-labeled samples")
    sys.exit(0)

print(f"Speaker selected: {speaker_id} | Samples: {len(samples)}")

# ---------------- COMPUTE EMBEDDINGS ----------------

embeddings = []
for s in samples:
    audio = s["audio"].unsqueeze(0).to(device)
    age = s["age"]

    with torch.no_grad():
        emb, _ = model(audio)              # ✅ only embedding used
        emb = torch.nn.functional.normalize(emb, dim=1)

    embeddings.append((age, emb.cpu().numpy()[0]))

# Sort by age
embeddings.sort(key=lambda x: x[0])

# ---------------- DISTANCE ANALYSIS ----------------

print("\nAge progression distances:\n")

for (age1, e1), (age2, e2) in combinations(embeddings, 2):
    age_gap = abs(age2 - age1)
    dist = np.linalg.norm(e2 - e1)

    print(
        f"Age {age1:>4} → {age2:>4} | "
        f"Gap: {age_gap:>3} yrs | "
        f"Embedding distance: {dist:.4f}"
    )

print("\n✅ Age progression validation complete")