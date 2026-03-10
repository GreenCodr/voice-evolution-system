import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = torch.load("training/dataset/X_embeddings.pt")
y = torch.load("training/dataset/y_age.pt")

pca = PCA(n_components=2)

X_reduced = pca.fit_transform(X)

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X_reduced[:,0],
    X_reduced[:,1],
    c=y,
    cmap="tab10",
    s=5
)

plt.colorbar(scatter)

plt.title("Voice Embedding Space by Age Group")

plt.show()