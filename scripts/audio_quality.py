# scripts/audio_quality.py

import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000
MIN_DURATION_SEC = 10.0
MIN_SNR_DB = 15.0
MIN_ACTIVE_RATIO = 0.6
MIN_RMS_DB = -35.0


def _rms_db(signal: np.ndarray) -> float:
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-9:
        return -100.0
    return 20 * np.log10(rms)


def _active_speech_ratio(signal: np.ndarray, sr: int) -> float:
    intervals = librosa.effects.split(signal, top_db=30)
    active = sum(end - start for start, end in intervals)
    return active / len(signal)


def _snr_db(signal: np.ndarray, sr: int) -> float:
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)

    frames = librosa.util.frame(signal, frame_length=frame_len, hop_length=hop)
    energy = np.mean(frames ** 2, axis=0)

    if len(energy) < 10:
        return 0.0

    energy_sorted = np.sort(energy)
    noise = np.mean(energy_sorted[: int(0.1 * len(energy))])
    speech = np.mean(energy_sorted[int(0.9 * len(energy)):])

    if noise < 1e-9:
        return 40.0

    return 10 * np.log10(speech / noise)


def audio_quality_gate(audio_path: str, dev_mode: bool = False) -> dict:
    """
    Returns:
    {
        accepted: bool,
        reason: str | None,
        duration: float,
        snr_db: float,
        rms_db: float,
        active_ratio: float
    }
    """

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
    min_duration = 2.0 if dev_mode else MIN_DURATION_SEC

    if duration < min_duration:
        return {
            "accepted": False,
            "reason": f"Audio too short ({duration:.2f}s)",
            "duration": duration
        }

    rms = _rms_db(audio)
    if rms < MIN_RMS_DB:
        return {
            "accepted": False,
            "reason": "Signal too weak / silence",
            "rms_db": rms
        }

    active_ratio = _active_speech_ratio(audio, sr)
    if active_ratio < MIN_ACTIVE_RATIO:
        return {
            "accepted": False,
            "reason": "Too much silence",
            "active_ratio": round(active_ratio, 2)
        }

    snr = _snr_db(audio, sr)
    min_snr = 8.0 if dev_mode else MIN_SNR_DB

    if snr < min_snr:
        return {
            "accepted": False,
            "reason": "Noisy recording (low SNR)",
            "snr_db": round(snr, 2)
        }

    return {
        "accepted": True,
        "reason": None,
        "duration": round(duration, 2),
        "snr_db": round(snr, 2),
        "rms_db": round(rms, 2),
        "active_ratio": round(active_ratio, 2),
    }