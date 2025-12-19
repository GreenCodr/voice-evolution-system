# scripts/test_age_delta_model.py

import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models/age_delta_light")
FEATURES = [
    "mean_pitch",
    "pitch_std",
    "spectral_centroid",
    "spectral_rolloff",
    "rms_energy",
    "speaking_rate",
]

# Load model
model = joblib.load(MODEL_DIR / "age_delta_model.joblib")
scaler = joblib.load(MODEL_DIR / "age_feature_scaler.joblib")

# Load some samples
df = pd.read_csv("datasets/common_voice/age_audio/features/age_features.csv")

# Pick 5 children + 5 adult
children = df[df["age_group"] == "children"].sample(5, random_state=42)
adult = df[df["age_group"] == "adult"].sample(5, random_state=42)

def predict(df_part, label):
    X = scaler.transform(df_part[FEATURES].values)
    preds = model.predict(X)
    print(f"\n🔹 {label}")
    for p in preds:
        print(f"Predicted age: {round(float(p), 1)}")

predict(children, "CHILDREN SAMPLES")
predict(adult, "ADULT SAMPLES")