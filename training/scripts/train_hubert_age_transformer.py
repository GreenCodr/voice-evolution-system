import torch
import torch.nn as nn
import torch.optim as optim

X = torch.load("training/dataset/hubert_X.pt")
y = torch.load("training/dataset/hubert_y.pt")

age_to_idx = {age:i for i,age in enumerate(sorted(list(set(y))))}
y_idx = torch.tensor([age_to_idx[a] for a in y])

class AgeTransformModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.age_embed = nn.Embedding(len(age_to_idx), 32)

        self.net = nn.Sequential(
            nn.Linear(1024+32,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,1024)
        )

    def forward(self,x,age):

        a = self.age_embed(age)
        z = torch.cat([x,a],dim=1)

        return self.net(z)

model = AgeTransformModel()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

epochs = 25
batch = 128

for epoch in range(epochs):

    perm = torch.randperm(len(X))

    total_loss = 0

    for i in range(0,len(X),batch):

        idx = perm[i:i+batch]

        xb = X[idx]
        ab = y_idx[idx]

        pred = model(xb,ab)

        loss = loss_fn(pred, xb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss {total_loss:.4f}")

torch.save(model.state_dict(),
           "training/models/hubert_age_transformer.pt")

print("Model saved")