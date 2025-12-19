# scripts/slerp_embeddings.py

import sys
import numpy as np
from pathlib import Path

def l2_norm(x):
    return x / np.linalg.norm(x)

def slerp(v0, v1, alpha):
    """
    Spherical Linear Interpolation
    """
    v0 = l2_norm(v0)
    v1 = l2_norm(v1)

    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    omega = np.arccos(dot)

    if omega < 1e-6:
        return v0

    so = np.sin(omega)
    return (
        np.sin((1 - alpha) * omega) / so * v0 +
        np.sin(alpha * omega) / so * v1
    )

def main(emb_a, emb_b, alpha, out_path):
    e0 = np.load(emb_a)
    e1 = np.load(emb_b)

    assert e0.shape == e1.shape, "Embedding shape mismatch"

    emb_interp = slerp(e0, e1, alpha)
    emb_interp = l2_norm(emb_interp)

    np.save(out_path, emb_interp)

    print("✅ SLERP embedding created")
    print("Alpha:", alpha)
    print("Saved to:", out_path)

if __name__ == "__main__":
    emb_a = sys.argv[1]
    emb_b = sys.argv[2]
    alpha = float(sys.argv[3])
    out_path = sys.argv[4]

    main(emb_a, emb_b, alpha, out_path)