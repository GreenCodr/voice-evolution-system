# scripts/train_age_delta_light.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib

# ---------------- Paths ----------------
DATA_PATH = Path("datasets/common_voice/age_audio/features/age_features.csv")
MODEL_DIR = Path("models/age_delta_light")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "mean_pitch",
    "pitch_std",
    "spectral_centroid",
    "spectral_rolloff",
    "rms_energy",
    "speaking_rate",
]

# ---------------- Load data ----------------
df = pd.read_csv(DATA_PATH)

AGE_MAP = {
    "children": 10.0,
    "adult": 35.0,
}

df["age_numeric"] = df["age_group"].map(AGE_MAP)

X = df[FEATURE_COLS].values
y = df["age_numeric"].values

# ---------------- Scale ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- Train light model ----------------
model = Ridge(alpha=0.5)  # light, stable
model.fit(X_scaled, y)

# ---------------- Save ----------------
joblib.dump(model, MODEL_DIR / "age_delta_model.joblib")
joblib.dump(scaler, MODEL_DIR / "age_feature_scaler.joblib")

print("✅ Light age-delta model trained successfully")
print("Samples used:", len(df))
print("Saved to:", MODEL_DIR)