# scripts/train_age_embedding.py

import sys
from pathlib import Path
import os

# ---------------- PATH SETUP ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------- IMPORTS ----------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from scripts.phase3_dataset import UnifiedVoiceDataset
from scripts.phase3_model import AgeEmbeddingModel
from scripts.phase3_collate import voice_collate_fn

# ---------------- CONFIG ----------------

MANIFESTS = [
    "data/librispeech_manifest_small.csv",
]

BATCH_SIZE = 1               # safe for 8GB RAM
GRAD_ACCUM_STEPS = 8         # effective batch = 8
EPOCHS = 3                   # sanity run
LR = 1e-4
CHECKPOINT_PATH = "models/age_embedding_v1.pt"

# ---------------- DEVICE ----------------

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

# ---------------- DATASET ----------------

dataset = UnifiedVoiceDataset(
    manifest_paths=MANIFESTS,
    max_samples=None
)

print("Dataset size:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    collate_fn=voice_collate_fn
)

# ---------------- MODEL ----------------

model = AgeEmbeddingModel(device=device)
model.train()

# Freeze wav2vec backbone (CRITICAL)
for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

age_loss_fn = nn.L1Loss()

# ---------------- TRAINING ----------------

global_step = 0
optimizer.zero_grad()

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")

    for batch_idx, sample in enumerate(loader):
        audio = sample["audio"].to(device)   # (B, T)
        ages = sample["age"]                 # list with None or value

        emb = model(audio)                   # (B, 128)

        # -------------------------------------------------
        # SUPERVISED TRAINING ONLY
        # -------------------------------------------------
        if ages[0] is None:
            continue  # ✅ no backward, no graph pollution

        age_tensor = torch.tensor(
            [ages[0]],
            dtype=torch.float32,
            device=device
        )

        # Safe on MPS
        pred_age = emb.contiguous().norm(dim=1)

        loss = age_loss_fn(pred_age, age_tensor)
        loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        if batch_idx % 10 == 0:
            print(f"Step {batch_idx} | Loss: {loss.item():.4f}")

print("\nTraining finished.")

# ---------------- SAVE MODEL ----------------

os.makedirs("models", exist_ok=True)

torch.save(
    {
        "model_state": model.state_dict(),
        "config": {
            "embedding_dim": 128,
            "backbone": "wav2vec2-base-960h"
        }
    },
    CHECKPOINT_PATH
)

print("✅ Model saved to:", CHECKPOINT_PATH)