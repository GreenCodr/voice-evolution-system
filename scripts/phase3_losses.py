# scripts/phase3_losses.py

import torch
import torch.nn.functional as F


def triplet_loss(anchor, positive, negative, margin=0.3):
    """
    Standard triplet loss:
    L = max(0, d(a,p) - d(a,n) + margin)
    """
    pos_dist = F.cosine_similarity(anchor, positive)
    neg_dist = F.cosine_similarity(anchor, negative)

    loss = torch.clamp(neg_dist - pos_dist + margin, min=0.0)
    return loss.mean()