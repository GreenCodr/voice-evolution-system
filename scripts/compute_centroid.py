# scripts/compute_centroid.py

import sys
from pathlib import Path
import numpy as np

def l2_norm(x):
    return x / np.linalg.norm(x)

def main(emb_dir, out_path):
    emb_dir = Path(emb_dir)
    files = list(emb_dir.glob("*.npy"))

    if len(files) == 0:
        raise RuntimeError("No embeddings found")

    embeddings = []
    for f in files:
        emb = np.load(f)
        emb = l2_norm(emb)
        embeddings.append(emb)

    centroid = np.mean(embeddings, axis=0)
    centroid = l2_norm(centroid)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, centroid)

    print(f"✅ Centroid saved to: {out_path}")
    print(f"   Used {len(embeddings)} embeddings")
    print(f"   Shape: {centroid.shape}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compute_centroid.py <emb_dir> <out.npy>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])