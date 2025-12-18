# scripts/speaker_verification.py
import numpy as np
from pathlib import Path

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def speaker_verification_gate(
    new_emb: np.ndarray,
    reference_embs: list,
    threshold: float = 0.80
) -> dict:
    """
    Args:
        new_emb: embedding of incoming audio
        reference_embs: list of existing embeddings for the user
    Returns:
        {
            "accepted": bool,
            "best_similarity": float,
            "reason": str | None
        }
    """

    if len(reference_embs) == 0:
        return {
            "accepted": True,
            "best_similarity": None,
            "reason": "First recording (no verification needed)"
        }

    similarities = [
        cosine_similarity(new_emb, ref)
        for ref in reference_embs
    ]

    best_sim = max(similarities)

    if best_sim < threshold:
        return {
            "accepted": False,
            "best_similarity": best_sim,
            "reason": "Speaker mismatch detected"
        }

    return {
        "accepted": True,
        "best_similarity": best_sim,
        "reason": None
    }