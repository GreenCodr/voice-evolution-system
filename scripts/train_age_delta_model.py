# scripts/train_age_delta_model.py

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import Ridge
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "learning" / "age_embedding_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "learning" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "age_delta_ridge.joblib"


# ------------------ SAFE LOADER ------------------

def load_embedding(path: str) -> np.ndarray | None:
    full_path = PROJECT_ROOT / path
    if not full_path.exists():
        return None
    return np.load(full_path).astype("float32")


# ------------------ MAIN ------------------

def main():
    if not DATA_FILE.exists():
        print("❌ Dataset not found:", DATA_FILE)
        return

    speaker_data = defaultdict(list)

    # -------- Load dataset safely --------
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            emb = load_embedding(row["embedding_path"])
            if emb is None:
                continue

            speaker_data[row["user_id"]].append({
                "age": int(row["age_at_recording"]),
                "embedding": emb
            })

    X, Y = [], []

    # -------- Build training pairs --------
    for user_id, records in speaker_data.items():
        if len(records) < 2:
            continue

        records = sorted(records, key=lambda r: r["age"])

        for i in range(len(records) - 1):
            a1, e1 = records[i]["age"], records[i]["embedding"]
            a2, e2 = records[i + 1]["age"], records[i + 1]["embedding"]

            delta_age = a2 - a1
            if delta_age <= 0:
                continue

            delta_emb = e2 - e1

            X.append([delta_age])
            Y.append(delta_emb)

    if not X:
        print("❌ Not enough age pairs to train")
        return

    X = np.array(X, dtype="float32")
    Y = np.array(Y, dtype="float32")

    print(f"📊 Training samples: {len(X)}")
    print(f"📐 Embedding dim: {Y.shape[1]}")

    # -------- Train model --------
    model = Ridge(alpha=1.0)
    model.fit(X, Y)

    joblib.dump(model, MODEL_PATH)

    print("✅ Age delta model trained")
    print("📦 Saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()