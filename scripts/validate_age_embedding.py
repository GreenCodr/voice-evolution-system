# scripts/validate_age_embedding.py

import sys
from pathlib import Path

# ---------------- PATH FIX (CRITICAL) ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------- IMPORTS ----------------

import torch
import torch.nn.functional as F

from scripts.phase3_model import AgeEmbeddingModel
from scripts.phase3_dataset import UnifiedVoiceDataset

# ---------------- CONFIG ----------------

CHECKPOINT_PATH = "models/age_embedding_v1.pt"
MANIFESTS = ["data/librispeech_manifest_small.csv"]

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

# ---------------- LOAD MODEL ----------------

ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

model = AgeEmbeddingModel(device=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

print("✅ Model loaded")

# ---------------- LOAD DATA ----------------

dataset = UnifiedVoiceDataset(
    manifest_paths=MANIFESTS,
    max_samples=3
)

print("Samples used:", len(dataset))

# ---------------- EMBEDDINGS ----------------

embeddings = []
meta = []

with torch.no_grad():
    for i in range(len(dataset)):
        sample = dataset[i]
        audio = sample["audio"].unsqueeze(0).to(device)

        emb = model(audio)
        emb = F.normalize(emb, dim=1)

        embeddings.append(emb.squeeze(0))
        meta.append(sample)

print("✅ Embeddings computed")

# ---------------- SIMILARITY ----------------

print("\nCosine Similarities:")
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        sim = F.cosine_similarity(
            embeddings[i].unsqueeze(0),
            embeddings[j].unsqueeze(0)
        ).item()

        print(
            f"Sample {i} ↔ Sample {j} | "
            f"Speaker {meta[i]['speaker_id']} ↔ {meta[j]['speaker_id']} | "
            f"Similarity = {sim:.4f}"
        )