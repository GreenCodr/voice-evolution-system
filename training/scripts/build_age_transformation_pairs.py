import torch
import random

# load embeddings
X = torch.load("training/dataset/combined_X.pt")

# load labels
y = torch.load("training/dataset/combined_y.pt")

# map string age groups to integers
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

# convert labels
y = torch.tensor([age_map[str(a)] for a in y])

pairs_src = []
pairs_tgt = []

N = len(X)

for i in range(N):

    src_emb = X[i]
    src_age = y[i]

    mask = y != src_age
    candidates = torch.where(mask)[0]

    if len(candidates) == 0:
        continue

    j = candidates[random.randint(0, len(candidates)-1)]

    tgt_emb = X[j]

    pairs_src.append(src_emb)
    pairs_tgt.append(tgt_emb)

pairs_src = torch.stack(pairs_src)
pairs_tgt = torch.stack(pairs_tgt)

torch.save(pairs_src, "training/dataset/evolve_source.pt")
torch.save(pairs_tgt, "training/dataset/evolve_target.pt")

print("Pairs created:", len(pairs_src))
print("Source shape:", pairs_src.shape)
print("Target shape:", pairs_tgt.shape)