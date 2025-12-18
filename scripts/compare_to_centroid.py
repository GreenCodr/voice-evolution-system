# scripts/compare_to_centroid.py
import sys
import numpy as np

def l2_norm(x):
    return x / np.linalg.norm(x)

if __name__ == "__main__":
    centroid_path = sys.argv[1]
    new_emb_path = sys.argv[2]

    centroid = l2_norm(np.load(centroid_path))
    new_emb = l2_norm(np.load(new_emb_path))

    similarity = float(np.dot(centroid, new_emb))

    print("ECAPA similarity to centroid:", round(similarity, 4))

    # -------- Production thresholds --------
    if similarity >= 0.92:
        print("✅ SAME VOICE — NO NEW VERSION")
    elif similarity >= 0.85:
        print("🟡 VOICE EVOLVED — REVIEW / NEW VERSION")
    else:
        print("❌ DIFFERENT SPEAKER — REJECT")