# scripts/speaker_verification.py

import numpy as np
from scripts.user_registry import UserRegistry

# ------------------ CORE MATH ------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


# ------------------ LOW-LEVEL GATE ------------------

def speaker_verification_gate(
    new_emb: np.ndarray,
    reference_embs: list,
    threshold: float = 0.80
) -> dict:
    """
    Low-level speaker verification gate.
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


# ------------------ PIPELINE WRAPPER ------------------

def verify_speaker(user_id: str, embedding: np.ndarray) -> bool:
    """
    High-level speaker verification used by process_new_voice().
    """

    registry = UserRegistry()
    user = registry.get_user(user_id)

    reference_embs = [
        np.load(v["embedding_path"])
        for v in user.get("voice_versions", [])
        if v.get("embedding_path")
    ]

    result = speaker_verification_gate(
        new_emb=embedding,
        reference_embs=reference_embs
    )

    return result["accepted"]
def verify_speaker(user_id: str, audio_path: str) -> bool:
    """
    High-level speaker verification entry.
    """
    from scripts.user_registry import load_user
    from scripts.embed_single_audio import extract_embedding

    user = load_user(user_id)
    reference_embs = [
        np.load(v["embedding_path"])
        for v in user.get("voice_versions", [])
        if "embedding_path" in v
    ]

    new_emb = extract_embedding(audio_path)

    result = speaker_verification_gate(
        new_emb=new_emb,
        reference_embs=reference_embs
    )

    return result["accepted"]