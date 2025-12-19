import numpy as np
from pathlib import Path

# ------------------ PATHS ------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LIBRI_REF = PROJECT_ROOT / "versions/ref_embedding.npy"
FSDD_EMB_DIR = PROJECT_ROOT / "datasets/fsdd/embeddings"

# ------------------ HELPERS ------------------

def load_norm(path):
    v = np.load(path).astype("float32")
    return v / np.linalg.norm(v)

def cosine(a, b):
    return float(np.dot(a, b))

def decision(sim):
    if sim >= 0.92:
        return "NO_NEW_VERSION"
    elif sim >= 0.80:
        return "CREATE_NEW_VERSION"
    else:
        return "REJECT"

# ------------------ MAIN ------------------

def main():
    ref = load_norm(LIBRI_REF)

    print("\n===== LibriSpeech → FSDD validation =====\n")

    for emb_path in sorted(FSDD_EMB_DIR.glob("*.npy")):
        emb = load_norm(emb_path)
        sim = cosine(ref, emb)

        print(
            f"{emb_path.name:30s} | "
            f"sim={sim:.4f} | "
            f"{decision(sim)}"
        )

if __name__ == "__main__":
    main()