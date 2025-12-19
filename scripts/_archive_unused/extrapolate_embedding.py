# scripts/extrapolate_embedding.py

import sys
import numpy as np
from pathlib import Path

def l2_norm(x):
    return x / np.linalg.norm(x)

def extrapolate(e_old, e_new, factor, mode):
    """
    mode: 'future' or 'past'
    factor: 0.1 → 0.3 (safe range)
    """
    direction = e_new - e_old

    if mode == "future":
        out = e_new + factor * direction
    elif mode == "past":
        out = e_old - factor * direction
    else:
        raise ValueError("mode must be 'future' or 'past'")

    return l2_norm(out)

def main(old_path, new_path, mode, factor, out_path):
    e_old = l2_norm(np.load(old_path))
    e_new = l2_norm(np.load(new_path))

    if e_old.shape != e_new.shape:
        raise ValueError("Embedding dimensions must match")

    if not (0.1 <= factor <= 0.3):
        raise ValueError("factor must be between 0.1 and 0.3")

    e_out = extrapolate(e_old, e_new, factor, mode)
    np.save(out_path, e_out)

    print("✅ Extrapolated voice embedding created")
    print("Mode      :", mode.upper())
    print("Factor    :", factor)
    print("Saved to  :", out_path)
    print("⚠️  Tag: PREDICTED VOICE")

if __name__ == "__main__":
    """
    Usage:
    python scripts/extrapolate_embedding.py \
        old.npy new.npy future 0.2 out.npy
    """
    _, old_p, new_p, mode, factor, out_p = sys.argv
    main(old_p, new_p, mode, float(factor), out_p)