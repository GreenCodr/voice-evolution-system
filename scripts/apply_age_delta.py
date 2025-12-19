# scripts/apply_age_delta.py

import numpy as np
from pathlib import Path

# ---------------- PATHS ----------------
USER_EMB = Path("versions/embeddings/user_002_1766164327.npy")
AGE_DELTAS = Path("embeddings/age_deltas.npy")
OUT = Path("embeddings/user_002_aged_adult.npy")

# ---------------- LOAD ----------------
base_emb = np.load(USER_EMB)
base_emb = base_emb / np.linalg.norm(base_emb)

age_deltas = np.load(AGE_DELTAS, allow_pickle=True).item()
delta = age_deltas["children_to_adult"]

# ---------------- APPLY AGE ----------------
alpha = 1.0  # strength of aging (0.5 = mild, 1.0 = full)

aged_emb = base_emb + alpha * delta
aged_emb = aged_emb / np.linalg.norm(aged_emb)

# ---------------- SAVE ----------------
OUT.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT, aged_emb)

print("✅ Aged embedding saved")
print("Output:", OUT)
print("Shape:", aged_emb.shape)