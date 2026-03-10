"""
DSPAgingEngine — Realistic voice aging via signal processing.

v4: Stronger aging perception for 55+ WITHOUT more noise.
The aging character comes from clean transforms:
  - Deeper pitch drop in 55-65 range (before atrophy rise at 70+)
  - More formant shift (vocal tract relaxation is progressive)
  - Stronger tremor (most recognizable aging marker)
  - More jitter (pitch instability is very noticeable)
  - Steeper spectral tilt (thinner voice = older perception)
Noise levels (breathiness, glottal) kept low for clarity.
"""

from pathlib import Path
import logging
import numpy as np
import yaml

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[3]

# ======================================================================
# Age-voice parameter table — v4
# ======================================================================
#
# v4 changes from v3 (55+ only, everything below 55 unchanged):
#
# INCREASED (make voice sound older):
#   pitch_offset:  55: -2.0 -> -2.8,  60: -2.5 -> -3.2,  65: -2.0 -> -2.5
#                  (deeper drop before atrophy rise — this is the big one)
#   formant:       55: 0.963 -> 0.955,  60: 0.957 -> 0.945,  65: 0.952 -> 0.938
#                  70: 0.947 -> 0.932,  75+: further reduced
#                  (vocal tract relaxes more aggressively — sounds genuinely older)
#   tremor_depth:  55: 0.014 -> 0.020,  60: 0.020 -> 0.030,  65: 0.028 -> 0.038
#                  (tremor is THE most recognizable aging marker to human ears)
#   tremor_rate:   60: 3.5 -> 4.0,  65: 4.5 -> 5.0
#                  (slightly faster tremor = more aged perception)
#   jitter:        55: 0.018 -> 0.025,  60: 0.025 -> 0.035,  65: 0.032 -> 0.045
#                  (pitch instability is immediately perceived as "old voice")
#   spectral_tilt: 55: -3.5 -> -4.5,  60: -4.5 -> -6.0,  65: -5.5 -> -7.0
#                  (thinner voice = older perception, but we keep cutoff floor at 4000Hz)
#
# UNCHANGED (already clean-sounding):
#   breathiness:   kept at v3 levels (max 0.030) — no more noise
#   hnr_noise:     kept at v3 levels (max 0.055) — no more noise

_AGE_VOICE_TABLE = [
    # age   pitch    formant  breath  tilt    tremor_hz  tremor_d  jitter  hnr_noise
    (5,     +5.0,    1.250,   0.005,   0.0,   0.0,       0.000,    0.000,  0.000),
    (8,     +4.0,    1.200,   0.005,   0.0,   0.0,       0.000,    0.000,  0.000),
    (12,    +2.5,    1.120,   0.008,   0.0,   0.0,       0.000,    0.005,  0.000),
    (16,    +1.0,    1.060,   0.012,   0.0,   0.0,       0.000,    0.008,  0.000),
    (20,    +0.3,    1.020,   0.008,   0.0,   0.0,       0.000,    0.003,  0.000),
    (25,     0.0,    1.000,   0.005,   0.0,   0.0,       0.000,    0.002,  0.000),
    (30,     0.0,    1.000,   0.006,   0.0,   0.0,       0.000,    0.002,  0.000),
    (35,    -0.2,    0.990,   0.008,  -0.5,   0.0,       0.000,    0.003,  0.005),
    (40,    -0.5,    0.985,   0.012,  -1.0,   0.0,       0.000,    0.005,  0.010),
    (45,    -0.8,    0.978,   0.015,  -1.8,   0.0,       0.000,    0.008,  0.015),
    (50,    -1.0,    0.970,   0.018,  -2.5,   1.5,       0.005,    0.012,  0.020),
    #                                                                              
    # === 55+ region: v4 boosted clean transforms ===                              
    #                                                                              
    (55,    -2.8,    0.955,   0.022,  -4.5,   2.8,       0.020,    0.025,  0.028),
    (60,    -3.2,    0.945,   0.025,  -6.0,   4.0,       0.030,    0.035,  0.035),
    (65,    -2.5,    0.938,   0.028,  -7.0,   5.0,       0.038,    0.045,  0.042),
    (70,    -1.5,    0.932,   0.030,  -7.5,   5.5,       0.042,    0.052,  0.048),
    (75,    +0.0,    0.928,   0.030,  -7.5,   5.8,       0.038,    0.048,  0.050),
    (80,    +1.0,    0.924,   0.030,  -7.5,   6.0,       0.042,    0.052,  0.053),
    (85,    +1.5,    0.920,   0.030,  -7.5,   6.0,       0.045,    0.055,  0.055),
]

_PARAM_NAMES = [
    "pitch_offset_semitones",
    "formant_factor",
    "breathiness",
    "spectral_tilt_db",
    "tremor_rate_hz",
    "tremor_depth",
    "jitter_factor",
    "hnr_noise",
]


def _lookup_age_params(age: float) -> dict:
    table = _AGE_VOICE_TABLE
    age = float(np.clip(age, table[0][0], table[-1][0]))
    for i in range(len(table) - 1):
        if table[i][0] <= age <= table[i + 1][0]:
            lo_age = table[i][0]
            hi_age = table[i + 1][0]
            t = (age - lo_age) / max(hi_age - lo_age, 0.1)
            result = {}
            for j, name in enumerate(_PARAM_NAMES):
                lo_val = table[i][j + 1]
                hi_val = table[i + 1][j + 1]
                result[name] = lo_val + t * (hi_val - lo_val)
            return result
    return {name: table[-1][j + 1] for j, name in enumerate(_PARAM_NAMES)}


class DSPAgingEngine:

    def __init__(self, age_profiles_path: str | None = None):
        if age_profiles_path is None:
            age_profiles_path = str(PROJECT_ROOT / "config" / "age_profiles.yaml")
        self.profiles = self._load_profiles(age_profiles_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def age_voice(self, wav: np.ndarray, sr: int, source_age: float, target_age: float) -> np.ndarray:
        if wav.ndim != 1:
            wav = wav.flatten()
        wav = wav.astype(np.float64)

        if abs(target_age - source_age) < 0.25:
            return wav.astype(np.float32)

        src_params = _lookup_age_params(source_age)
        tgt_params = _lookup_age_params(target_age)

        # Step 1: Formant shifting (vocal tract length)
        formant_ratio = tgt_params["formant_factor"] / max(src_params["formant_factor"], 0.01)
        formant_ratio = float(np.clip(formant_ratio, 0.80, 1.25))
        if abs(formant_ratio - 1.0) > 0.012:
            wav = self._shift_formants(wav, sr, formant_ratio)

        # Step 2: Pitch shifting (fundamental frequency)
        semitone_delta = tgt_params["pitch_offset_semitones"] - src_params["pitch_offset_semitones"]
        semitone_delta = float(np.clip(semitone_delta, -8.0, 8.0))
        if abs(semitone_delta) > 0.15:
            wav = self._pitch_shift_psola(wav, sr, semitone_delta)

        # Step 3: Spectral tilt (voice thins with age)
        tilt_delta = tgt_params["spectral_tilt_db"] - src_params["spectral_tilt_db"]
        if abs(tilt_delta) > 0.3:
            wav = self._apply_spectral_tilt(wav, sr, tilt_delta, target_age)

        # Step 4: Glottal noise (subtle roughness)
        hnr_delta = tgt_params["hnr_noise"] - src_params["hnr_noise"]
        if hnr_delta > 0.003:
            wav = self._add_glottal_noise(wav, sr, hnr_delta)

        # Step 5: Breathiness (subtle aspiration)
        breath_delta = tgt_params["breathiness"] - src_params["breathiness"]
        if breath_delta > 0.003:
            wav = self._add_spectral_breathiness(wav, sr, breath_delta)

        # Step 6: Vocal tremor (strong aging marker)
        tremor_rate = tgt_params["tremor_rate_hz"]
        effective_depth = tgt_params["tremor_depth"] - src_params["tremor_depth"]
        if tremor_rate > 0.5 and effective_depth > 0.002:
            wav = self._add_tremor(wav, sr, tremor_rate, effective_depth)

        # Step 7: Pitch jitter (vocal instability)
        jitter_delta = tgt_params["jitter_factor"] - src_params["jitter_factor"]
        if jitter_delta > 0.003:
            wav = self._add_pitch_jitter(wav, sr, jitter_delta)

        # Step 8: Smooth edges + normalize
        wav = self._fade_edges(wav, sr, fade_ms=15)
        wav = self._normalize(wav)

        return wav.astype(np.float32)

    def quality_check(self, audio: np.ndarray, sr: int) -> bool:
        if audio is None or len(audio) == 0:
            return False
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        if rms < 0.0001:
            logger.warning("Quality check: near-silence (RMS=%.6f)", rms)
            return False
        if len(audio) < sr:
            return rms > 0.0001
        segment = audio[:sr].astype(np.float64)
        fft = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(len(segment), 1.0 / sr)
        mag = np.abs(fft)
        speech_mask = (freqs >= 300) & (freqs <= 4000)
        total_energy = np.sum(mag ** 2)
        speech_energy = np.sum(mag[speech_mask] ** 2)
        ratio = speech_energy / max(total_energy, 1e-10)
        if ratio < 0.15:
            logger.warning("Quality check: low speech energy ratio %.2f", ratio)
            return False
        return True

    # ------------------------------------------------------------------
    # Step 1: Formant shifting
    # ------------------------------------------------------------------

    def _shift_formants(self, wav: np.ndarray, sr: int, formant_ratio: float) -> np.ndarray:
        if not HAS_PARSELMOUTH:
            return wav
        try:
            sound = parselmouth.Sound(wav, sampling_frequency=sr)
            manipulated = call(sound, "Change gender", 75, 600, formant_ratio, 0, 1.0, 1.0)
            result = manipulated.values[0]
            if len(result) == 0 or np.all(np.isnan(result)):
                logger.warning("Formant shift returned empty/NaN, skipping")
                return wav
            return result
        except Exception as e:
            logger.warning("Formant shift failed: %s", e)
            return wav

    # ------------------------------------------------------------------
    # Step 2: Pitch shifting (PSOLA)
    # ------------------------------------------------------------------

    def _pitch_shift_psola(self, wav: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        if HAS_PARSELMOUTH:
            try:
                sound = parselmouth.Sound(wav, sampling_frequency=sr)
                manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
                pitch_tier = call(manipulation, "Extract pitch tier")
                factor = 2.0 ** (semitones / 12.0)
                call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
                call([manipulation, pitch_tier], "Replace pitch tier")
                result_sound = call(manipulation, "Get resynthesis (overlap-add)")
                result = result_sound.values[0]
                if len(result) > 0 and not np.all(np.isnan(result)):
                    return result
                logger.warning("PSOLA returned empty, falling back to librosa")
            except Exception as e:
                logger.warning("PSOLA failed: %s, falling back to librosa", e)

        if HAS_LIBROSA:
            result = librosa.effects.pitch_shift(
                y=wav.astype(np.float32), sr=sr, n_steps=semitones, n_fft=2048)
            return result.astype(np.float64)

        logger.warning("No pitch shift backend available")
        return wav

    # ------------------------------------------------------------------
    # Step 3: Spectral tilt
    # ------------------------------------------------------------------

    def _apply_spectral_tilt(self, wav: np.ndarray, sr: int, tilt_delta_db: float, target_age: float) -> np.ndarray:
        nyquist = sr / 2.0

        if tilt_delta_db < 0:
            severity = min(abs(tilt_delta_db) / 10.0, 1.0)

            # Floor at 4000Hz to preserve speech clarity
            cutoff_hz = 7500 - severity * 3500
            cutoff_hz = max(cutoff_hz, 4000)
            cutoff_hz = min(cutoff_hz, nyquist - 200)

            sos = butter(2, cutoff_hz / nyquist, btype="low", output="sos")
            filtered = sosfiltfilt(sos, wav)

            # v4: increased max blend from 0.65 to 0.70
            # More thinning without going below 4000Hz cutoff = older but clear
            blend = 0.3 + severity * 0.40  # 0.3 to 0.70
            wav = blend * filtered + (1.0 - blend) * wav

        elif tilt_delta_db > 0:
            cutoff_hz = min(4000, nyquist - 200)
            sos = butter(2, cutoff_hz / nyquist, btype="high", output="sos")
            high_content = sosfiltfilt(sos, wav)
            boost = min(abs(tilt_delta_db) / 15.0, 0.2)
            wav = wav + boost * high_content

        return wav

    # ------------------------------------------------------------------
    # Step 4: Glottal noise (kept subtle)
    # ------------------------------------------------------------------

    def _add_glottal_noise(self, wav: np.ndarray, sr: int, amount: float) -> np.ndarray:
        rms = np.sqrt(np.mean(wav ** 2))
        if rms < 1e-6:
            return wav

        noise = np.random.randn(len(wav))
        nyquist = sr / 2.0
        lo = min(100 / nyquist, 0.9)
        hi = min(1500 / nyquist, 0.95)
        if lo >= hi:
            return wav

        sos = butter(2, [lo, hi], btype="band", output="sos")
        noise = sosfiltfilt(sos, noise)

        noise_rms = np.sqrt(np.mean(noise ** 2))
        if noise_rms < 1e-10:
            return wav

        # Kept at v3 reduced levels — subtle texture only
        target_level = min(amount * 0.4, 0.025) * rms
        noise = noise * (target_level / noise_rms)

        envelope = self._speech_envelope(wav, sr)
        gate = np.clip(envelope / (rms * 0.3), 0, 1)
        noise = noise * gate

        return wav + noise

    # ------------------------------------------------------------------
    # Step 5: Breathiness (kept subtle)
    # ------------------------------------------------------------------

    def _add_spectral_breathiness(self, wav: np.ndarray, sr: int, amount: float) -> np.ndarray:
        rms = np.sqrt(np.mean(wav ** 2))
        if rms < 1e-6:
            return wav

        noise = np.random.randn(len(wav))
        nyquist = sr / 2.0
        lo = min(2000 / nyquist, 0.9)
        hi = min(6000 / nyquist, 0.95)
        if lo >= hi:
            return wav

        sos = butter(2, [lo, hi], btype="band", output="sos")
        noise = sosfiltfilt(sos, noise)

        noise_rms = np.sqrt(np.mean(noise ** 2))
        if noise_rms < 1e-10:
            return wav

        # Kept at v3 reduced levels — no heavy breathing
        target_level = min(amount * 0.5, 0.03) * rms
        noise = noise * (target_level / noise_rms)

        envelope = self._speech_envelope(wav, sr)
        gate = np.clip(envelope / (rms * 0.4), 0, 1)
        noise = noise * gate

        return wav + noise

    # ------------------------------------------------------------------
    # Step 6: Vocal tremor (PRIMARY aging marker — boosted in v4)
    # ------------------------------------------------------------------

    def _add_tremor(self, wav: np.ndarray, sr: int, rate_hz: float, depth: float) -> np.ndarray:
        """
        Vocal tremor is the single most recognizable sign of an aging voice.
        v4: raised cap from 4% to 5% to make 65+ voices more distinctly old.
        """
        depth = min(depth, 0.05)  # v4: raised cap from 0.04 to 0.05
        t = np.arange(len(wav)) / sr

        # Irregular tremor — slight frequency wobble for naturalness
        phase_wobble = 0.3 * np.sin(2 * np.pi * 0.5 * t)
        modulator = 1.0 + depth * np.sin(2 * np.pi * rate_hz * t + phase_wobble)

        return wav * modulator

    # ------------------------------------------------------------------
    # Step 7: Pitch jitter (boosted in v4)
    # ------------------------------------------------------------------

    def _add_pitch_jitter(self, wav: np.ndarray, sr: int, amount: float) -> np.ndarray:
        """
        Pitch instability — the voice wavers unpredictably.
        v4: raised cap from 5% to 6% for stronger aging effect.
        """
        amount = min(amount, 0.06)  # v4: raised from 0.05 to 0.06
        mod_freq = 80
        t = np.arange(len(wav)) / sr
        random_phase = np.cumsum(np.random.randn(len(wav)) * 0.01)
        modulator = 1.0 + amount * 0.5 * np.sin(2 * np.pi * mod_freq * t + random_phase)

        envelope = self._speech_envelope(wav, sr)
        rms = np.sqrt(np.mean(wav ** 2))
        if rms < 1e-6:
            return wav
        gate = np.clip(envelope / (rms * 0.3), 0, 1)

        modulated = wav * modulator
        return gate * modulated + (1.0 - gate) * wav

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _speech_envelope(self, wav: np.ndarray, sr: int) -> np.ndarray:
        envelope = np.abs(wav)
        window = max(int(sr * 0.03), 1)
        if window < len(envelope):
            kernel = np.ones(window) / window
            envelope = np.convolve(envelope, kernel, mode="same")
        return envelope

    def _fade_edges(self, wav: np.ndarray, sr: int, fade_ms: float = 15) -> np.ndarray:
        fade_samples = int(sr * fade_ms / 1000)
        if fade_samples < 1 or len(wav) < fade_samples * 2:
            return wav
        wav = wav.copy()
        wav[:fade_samples] *= np.linspace(0, 1, fade_samples)
        wav[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        return wav

    def _normalize(self, wav: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(wav))
        if peak < 1e-6:
            return wav
        return wav * (0.95 / peak)

    # ------------------------------------------------------------------
    # Profile loading (backward compat)
    # ------------------------------------------------------------------

    def _load_profiles(self, path: str) -> dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning("Could not load age profiles from %s: %s", path, e)
            return self._default_profiles()

    def _default_profiles(self) -> dict:
        return {
            "child":   {"min_age": 5,  "max_age": 12, "pitch_shift_semitones": 4.0,  "speaking_rate": 1.25, "breathiness": 0.005, "jitter": 0.03, "shimmer": 0.01,  "tremor_rate": 0.0},
            "teen":    {"min_age": 13, "max_age": 19, "pitch_shift_semitones": 2.0,  "speaking_rate": 1.1,  "breathiness": 0.01,  "jitter": 0.02, "shimmer": 0.008, "tremor_rate": 0.0},
            "adult":   {"min_age": 20, "max_age": 50, "pitch_shift_semitones": 0.0,  "speaking_rate": 1.0,  "breathiness": 0.01,  "jitter": 0.01, "shimmer": 0.005, "tremor_rate": 0.0},
            "senior":  {"min_age": 51, "max_age": 70, "pitch_shift_semitones": -1.5, "speaking_rate": 0.9,  "breathiness": 0.03,  "jitter": 0.04, "shimmer": 0.02,  "tremor_rate": 4.0},
            "elderly": {"min_age": 71, "max_age": 90, "pitch_shift_semitones": -2.5, "speaking_rate": 0.85, "breathiness": 0.06,  "jitter": 0.06, "shimmer": 0.035, "tremor_rate": 6.0},
        }

    def _interpolate_profile(self, age: float) -> dict:
        ordered = sorted(self.profiles.values(), key=lambda p: p["min_age"])
        if age <= ordered[0]["min_age"]:
            return dict(ordered[0])
        if age >= ordered[-1]["max_age"]:
            return dict(ordered[-1])
        for i in range(len(ordered) - 1):
            lo = ordered[i]
            hi = ordered[i + 1]
            lo_mid = (lo["min_age"] + lo["max_age"]) / 2.0
            hi_mid = (hi["min_age"] + hi["max_age"]) / 2.0
            if lo_mid <= age <= hi_mid:
                t = (age - lo_mid) / max(hi_mid - lo_mid, 1.0)
                result = {}
                for key in lo:
                    if isinstance(lo[key], (int, float)) and key in hi:
                        result[key] = lo[key] + t * (hi[key] - lo[key])
                    else:
                        result[key] = lo[key]
                return result
        mid_ages = [(p, (p["min_age"] + p["max_age"]) / 2.0) for p in ordered]
        nearest = min(mid_ages, key=lambda x: abs(x[1] - age))
        return dict(nearest[0])


# Backward-compatible standalone function for tests
def _profile_for_age(age: float) -> dict:
    engine = DSPAgingEngine()
    return engine._interpolate_profile(age)