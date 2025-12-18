# scripts/audio_quality.py
import numpy as np
import soundfile as sf
import librosa

# ================= PRODUCTION CONSTANTS =================

TARGET_SR = 16000

MIN_DURATION_SEC = 8.0        # real-world safe (phone calls, notes)
MIN_ACTIVE_RATIO = 0.6        # % of non-silent speech
MIN_RMS_DB = -35.0            # silence / very weak mic
MIN_SNR_DB = 15.0             # robust SNR (speech vs noise)

# ================= HELPERS =================

def rms_db(signal: np.ndarray) -> float:
    rms = np.sqrt(np.mean(signal ** 2))
    if rms <= 1e-9:
        return -100.0
    return 20 * np.log10(rms)

def robust_snr_db(signal: np.ndarray, sr: int) -> float:
    """
    Robust SNR:
    speech = top-energy frames
    noise = bottom-energy frames
    """
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)

    frames = librosa.util.frame(signal, frame_length=frame_len, hop_length=hop)
    energy = np.mean(frames ** 2, axis=0)

    if len(energy) < 10:
        return 0.0

    energy_sorted = np.sort(energy)
    noise_energy = np.mean(energy_sorted[: max(1, int(0.1 * len(energy)))])
    speech_energy = np.mean(energy_sorted[int(0.9 * len(energy)):])

    if noise_energy <= 1e-9:
        return 40.0

    return 10 * np.log10(speech_energy / noise_energy)

def active_speech_ratio(signal: np.ndarray, sr: int) -> float:
    intervals = librosa.effects.split(signal, top_db=30)
    active = sum((end - start) for start, end in intervals)
    return active / len(signal)

# ================= MAIN GATE =================

def audio_quality_gate(audio_path: str, dev_mode: bool = False) -> dict:
    try:
        audio, sr = sf.read(audio_path)
    except Exception as e:
        return {"accepted": False, "reason": f"Read error: {e}"}

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        audio = librosa.resample(audio.astype("float32"), sr, TARGET_SR)
        sr = TARGET_SR

    duration = len(audio) / sr

    # ---- DEV MODE RELAXATION (ONLY FOR PIPELINE DEBUG) ----
    min_duration = 2.0 if dev_mode else MIN_DURATION_SEC
    min_snr = 8.0 if dev_mode else MIN_SNR_DB
    # ------------------------------------------------------

    if duration < min_duration:
        return {
            "accepted": False,
            "reason": f"Too short ({duration:.2f}s)",
            "duration": duration
        }

    rms = rms_db(audio)
    if rms < MIN_RMS_DB:
        return {
            "accepted": False,
            "reason": "Very low signal (silence / bad mic)",
            "rms_db": rms
        }

    active_ratio = active_speech_ratio(audio, sr)
    if active_ratio < MIN_ACTIVE_RATIO:
        return {
            "accepted": False,
            "reason": "Too much silence",
            "active_ratio": round(active_ratio, 2)
        }

    snr = robust_snr_db(audio, sr)
    if snr < min_snr:
        return {
            "accepted": False,
            "reason": "Low SNR (noisy environment)",
            "snr_db": round(snr, 2)
        }

    return {
        "accepted": True,
        "duration": round(duration, 2),
        "snr_db": round(snr, 2),
        "rms_db": round(rms, 2),
        "active_ratio": round(active_ratio, 2),
        "reason": None
    }