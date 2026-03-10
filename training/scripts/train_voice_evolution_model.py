import torch
import torch.nn as nn
import torch.optim as optim


class VoiceEvolutionModel(nn.Module):
    """
    Neural network that learns how voice embeddings evolve.
    """

    def __init__(self, emb_dim=1216):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, emb_dim)
        )

    def forward(self, x):
        delta = self.net(x)
        return x + delta


def train():

    # ---------------- Load Dataset ----------------
    src = torch.load("training/dataset/evolve_source.pt")
    tgt = torch.load("training/dataset/evolve_target.pt")

    print("Dataset loaded")
    print("Source:", src.shape)
    print("Target:", tgt.shape)

    src = src.float()
    tgt = tgt.float()

    # ---------------- Model ----------------
    model = VoiceEvolutionModel()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    epochs = 30
    batch_size = 128

    N = src.shape[0]

    # ---------------- Training Loop ----------------
    for epoch in range(epochs):

        perm = torch.randperm(N)
        total_loss = 0

        for i in range(0, N, batch_size):

            idx = perm[i:i+batch_size]

            x = src[idx]
            y = tgt[idx]

            pred = model(x)

            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss {total_loss:.4f}")

    # ---------------- Save Model ----------------
    torch.save(
        model.state_dict(),
        "training/models/voice_evolution_model.pt"
    )

    print("Voice evolution model saved")


# Only run training when script executed directly
if __name__ == "__main__":
    train()