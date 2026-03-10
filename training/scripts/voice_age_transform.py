import torch
import numpy as np

prototypes = torch.load("training/dataset/age_prototypes.pt")

def transform_embedding(embedding, source_age, target_age, strength=1.0):

    source_proto = prototypes[source_age]
    target_proto = prototypes[target_age]

    age_direction = target_proto - source_proto

    new_embedding = embedding + strength * age_direction

    return new_embedding


if __name__ == "__main__":

    emb = torch.load("training/dataset/X_embeddings.pt")[0]

    # convert to tensor if numpy
    emb = torch.tensor(emb, dtype=torch.float32)

    new_emb = transform_embedding(
        emb,
        source_age=2,
        target_age=5
    )

    print("Original embedding norm:", torch.norm(emb).item())
    print("New embedding norm:", torch.norm(new_emb).item())