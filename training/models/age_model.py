import torch
import torch.nn as nn


class AgeModel(nn.Module):

    def __init__(self, input_dim=1216, num_ages=8):
        super().__init__()

        # age embedding used during training
        self.age_embed = nn.Embedding(num_ages, 32)

        self.net = nn.Sequential(
            nn.Linear(input_dim + 32, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, input_dim)
        )

    def forward(self, emb, age):

        age = age.long()
        age_vec = self.age_embed(age)

        x = torch.cat([emb, age_vec], dim=1)

        return self.net(x)