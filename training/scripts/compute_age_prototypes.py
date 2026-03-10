import torch
import pandas as pd
import numpy as np

X = torch.load("training/dataset/X_embeddings.pt")
y = torch.load("training/dataset/y_age.pt")

age_prototypes = {}

for age in range(8):

    indices = (y == age)

    group_embeddings = X[indices]

    mean_embedding = group_embeddings.mean(axis=0)

    age_prototypes[age] = mean_embedding

torch.save(age_prototypes,"training/dataset/age_prototypes.pt")

print("Age prototypes saved")