import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

X = torch.load("training/dataset/X_embeddings.pt")
y = torch.load("training/dataset/y_age.pt")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

class AgeModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(192,256),
            nn.ReLU(),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,8)
        )

    def forward(self,x):
        return self.net(x)

model = AgeModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20

for epoch in range(epochs):

    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():

        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

        preds = torch.argmax(val_outputs, dim=1)
        acc = (preds == y_val).float().mean()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Train Loss {loss.item():.4f} "
        f"Val Loss {val_loss.item():.4f} "
        f"Val Acc {acc.item():.4f}"
    )

torch.save(model.state_dict(),"training/models/age_classifier.pt")

print("Model saved")