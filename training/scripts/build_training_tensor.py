import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

EMB_DIR = "training/dataset/embeddings_ecapa"
META = "training/dataset/dataset_metadata.csv"

df = pd.read_csv(META)

X = []
y = []

age_map = {
    "5-10":0,
    "10-15":1,
    "15-20":2,
    "20-30":3,
    "30-40":4,
    "40-50":5,
    "50-60":6,
    "60+":7
}

for _, row in tqdm(df.iterrows(), total=len(df)):

    emb_file = row["file"].replace(".wav",".npy")
    emb_path = os.path.join(EMB_DIR, emb_file)

    if not os.path.exists(emb_path):
        continue

    emb = torch.load(emb_path)

    X.append(emb)
    y.append(age_map[row["age_group"]])

X = np.array(X)
y = np.array(y)

torch.save(X, "training/dataset/X_embeddings.pt")
torch.save(y, "training/dataset/y_age.pt")

print("Saved training tensors")
print("Samples:", len(X))