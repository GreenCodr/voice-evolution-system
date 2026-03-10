import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

meta = pd.read_csv("training/dataset/dataset_metadata.csv")

ecapa_dir = "training/dataset/embeddings"
hubert_dir = "training/dataset/hubert_embeddings"

X = []
y = []

for _, row in tqdm(meta.iterrows(), total=len(meta)):

    file = row["file"]
    age = row["age_group"]

    base = file.replace(".wav","")

    ecapa_path = os.path.join(ecapa_dir, base + ".npy")
    hubert_path = os.path.join(hubert_dir, base + ".npy")

    if not os.path.exists(ecapa_path):
        continue

    if not os.path.exists(hubert_path):
        continue

    ecapa = np.load(ecapa_path)
    hubert = np.load(hubert_path)

    combined = np.concatenate([ecapa, hubert])

    X.append(combined)
    y.append(age)

X = torch.tensor(np.array(X)).float()

torch.save(X, "training/dataset/combined_X.pt")
torch.save(y, "training/dataset/combined_y.pt")

print("Combined embeddings saved")
print("Samples:", len(X))