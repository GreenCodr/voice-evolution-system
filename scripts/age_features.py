import librosa
import numpy as np


def extract_age_features(audio_path: str, sr: int = 16000) -> dict:
    """
    Extract speaker-agnostic, age-related acoustic features.
    """

    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # ---------------- Pitch ----------------
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )
    f0 = f0[~np.isnan(f0)]

    mean_pitch = float(np.mean(f0)) if len(f0) > 0 else 0.0
    pitch_std = float(np.std(f0)) if len(f0) > 0 else 0.0

    # ---------------- Spectral ----------------
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # ---------------- Energy ----------------
    rms = np.mean(librosa.feature.rms(y=y))

    # ---------------- Speaking Rate (proxy) ----------------
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    speaking_rate = np.mean(onset_env)

    return {
        "mean_pitch": mean_pitch,
        "pitch_std": pitch_std,
        "spectral_centroid": float(centroid),
        "spectral_rolloff": float(rolloff),
        "rms_energy": float(rms),
        "speaking_rate": float(speaking_rate),
    }