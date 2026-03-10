import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

meta = pd.read_csv("training/dataset/dataset_metadata.csv")

hubert_dir = "training/dataset/hubert_embeddings"

X = []
y = []

for _, row in tqdm(meta.iterrows(), total=len(meta)):

    file = row["file"].replace(".wav", ".npy")
    age = row["age_group"]

    path = os.path.join(hubert_dir, file)

    if not os.path.exists(path):
        continue

    emb = np.load(path)

    X.append(emb)
    y.append(age)

X = torch.tensor(np.array(X)).float()

torch.save(X, "training/dataset/hubert_X.pt")
torch.save(y, "training/dataset/hubert_y.pt")

print("Saved HuBERT tensors")
print("Samples:", len(X))