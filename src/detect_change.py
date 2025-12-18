# src/detect_change.py
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import csv

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def find_best_match(new_emb, emb_dir, exclude_fname=None):
    best_sim = -1.0
    best_file = None
    emb_dir = Path(emb_dir)
    if not emb_dir.exists():
        return best_sim, best_file
    for p in emb_dir.glob("*.npy"):
        if exclude_fname and p.name == exclude_fname:
            continue
        try:
            emb = np.load(p)
        except Exception:
            continue
        sim = cosine_similarity(new_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_file = str(p)
    return best_sim, best_file

def save_version(versions_dir, emb_path, best_sim):
    os.makedirs(versions_dir, exist_ok=True)
    fname = Path(emb_path).name
    dest = Path(versions_dir) / fname
    # move (rename) into versions folder
    os.replace(emb_path, dest)
    meta_path = Path(versions_dir) / "versions.csv"
    existed = meta_path.exists()
    with open(meta_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not existed:
            writer.writerow(["version_id","timestamp_utc","emb_file","best_similarity"])
        version_id = int(datetime.utcnow().timestamp())
        writer.writerow([version_id, datetime.utcnow().isoformat()+"Z", fname, f"{best_sim:.4f}"])
    return dest, meta_path

def main(new_emb_path, emb_dir="embeddings", versions_dir="versions", threshold=0.75):
    new_emb_path = Path(new_emb_path)
    if not new_emb_path.exists():
        print("ERROR: embedding file not found:", new_emb_path)
        sys.exit(1)

    new_emb = np.load(new_emb_path)
    exclude = new_emb_path.name

    best_sim, best_file = find_best_match(new_emb, emb_dir, exclude_fname=exclude)
    if best_file is None:
        print("No previous embeddings found. Marking as FIRST version.")
        dest, meta = save_version(versions_dir, str(new_emb_path), best_sim if best_sim!=-1.0 else 0.0)
        print("Saved new version embedding to:", dest)
        print("Metadata written to:", meta)
        return

    print("Best similarity against history: {:.4f}".format(best_sim))
    print("Best file:", best_file)
    if best_sim < threshold:
        print(f"Max similarity {best_sim:.4f} < threshold {threshold:.2f} -> CREATING NEW VERSION")
        dest, meta = save_version(versions_dir, str(new_emb_path), best_sim)
        print("Saved new version embedding to:", dest)
        print("Metadata written to:", meta)
    else:
        print(f"Max similarity {best_sim:.4f} >= threshold {threshold:.2f} -> NO new version (same voice).")
        print("Keeping embedding in:", new_emb_path)

if __name__ == "__main__":
    # run as: python src/detect_change.py embeddings/your_emb.npy --threshold 0.75
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("emb_path", help="Path to new embedding (.npy)")
    parser.add_argument("--emb_dir", default="embeddings", help="Folder with previous embeddings")
    parser.add_argument("--versions_dir", default="versions", help="Folder to store versioned embeddings and metadata")
    parser.add_argument("--threshold", type=float, default=0.75, help="cosine similarity threshold (same if >= threshold)")
    args = parser.parse_args()
    main(args.emb_path, emb_dir=args.emb_dir, versions_dir=args.versions_dir, threshold=args.threshold)