# scripts/age_embedding.py
import numpy as np
from pathlib import Path
from scripts.slerp import slerp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_DIR = PROJECT_ROOT / "versions"


def load_embedding(emb_path: str):
    """
    Robust loader:
    - supports paths stored as filenames
    - supports relative paths
    """
    p = Path(emb_path)

    # Case 1: already absolute
    if p.is_absolute() and p.exists():
        return np.load(p)

    # Case 2: relative to versions/
    candidate = VERSIONS_DIR / p.name
    if candidate.exists():
        return np.load(candidate)

    # Case 3: relative to project root
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return np.load(candidate)

    raise FileNotFoundError(f"Embedding not found: {emb_path}")


def get_embedding_for_age(selection):
    """
    Returns:
    - embedding vector
    - metadata dict
    """

    mode = selection["mode"]

    if mode == "EXACT":
        emb = load_embedding(selection["version"]["emb_file"])
        return emb, {"mode": "exact"}

    if mode in ("PAST_ONLY", "FUTURE_ONLY"):
        emb = load_embedding(selection["version"]["emb_file"])
        return emb, {"mode": "fallback"}

    if mode == "INTERPOLATE":
        left = selection["left"]
        right = selection["right"]
        alpha = selection["alpha"]

        emb_l = load_embedding(left["emb_file"])
        emb_r = load_embedding(right["emb_file"])

        emb = slerp(emb_l, emb_r, alpha)

        return emb, {
            "mode": "slerp",
            "alpha": alpha,
            "from_age": left["age"],
            "to_age": right["age"]
        }

    raise ValueError(f"Unsupported selection mode: {mode}")