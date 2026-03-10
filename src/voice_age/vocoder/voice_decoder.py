"""
src/voice_age/vocoder/voice_decoder.py

VoiceDecoder — unified interface for converting a reference recording
into an age-adjusted voice output.

Architecture
------------
This module wraps both:
  1. A pretrained neural voice conversion model (optional, higher quality)
  2. DSPAgingEngine (always available, guaranteed to produce real speech)

Neural back-end priority (auto-detection):
  A) TTS / XTTS-v2 (Coqui)  — if `TTS` package is installed
  B) DSP only               — always works

The `decode()` method:
  1. Tries the neural back-end
  2. Runs quality_check on the output
  3. Falls back to DSPAgingEngine if the check fails or neural is unavailable

All audio arrays are float32 mono at the project sample rate (16 kHz).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import librosa

from voice_age.vocoder.dsp_aging import DSPAgingEngine
from voice_age.config import DEFAULT_SR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DecodeResult:
    audio: np.ndarray           # float32 mono
    sr: int
    method_used: str            # "xtts" | "dsp"
    quality_ok: bool
    source_age: float
    target_age: float


# ---------------------------------------------------------------------------
# VoiceDecoder
# ---------------------------------------------------------------------------

class VoiceDecoder:
    """
    Converts a reference voice recording to sound like the target age.

    Parameters
    ----------
    model_name : "auto" | "xtts" | "dsp"
        "auto"  → try XTTS first, fall back to DSP
        "xtts"  → force XTTS (raises if unavailable)
        "dsp"   → force DSP-only
    """

    def __init__(self, model_name: str = "auto"):
        self._model_name = model_name.lower().strip()
        self._dsp = DSPAgingEngine()
        self._xtts = None

        if self._model_name in ("auto", "xtts"):
            self._xtts = self._try_load_xtts()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def decode(
        self,
        reference_audio_path: str,
        source_age: float,
        target_age: float,
        sr: int = DEFAULT_SR,
    ) -> DecodeResult:
        """
        Generate age-adjusted audio from the reference recording.

        Parameters
        ----------
        reference_audio_path : path to the stored voice WAV
        source_age           : age of the speaker in the reference recording
        target_age           : desired output age
        sr                   : target sample rate for output

        Returns
        -------
        DecodeResult with float32 mono audio
        """
        wav = self._load_reference(reference_audio_path, sr)

        # ----------------------------------------------------------------
        # If source age == target age, return reference as-is
        # ----------------------------------------------------------------
        if abs(target_age - source_age) < 0.25:
            return DecodeResult(
                audio=wav,
                sr=sr,
                method_used="direct",
                quality_ok=True,
                source_age=source_age,
                target_age=target_age,
            )

        # ----------------------------------------------------------------
        # Try neural path (XTTS)
        # ----------------------------------------------------------------
        if self._xtts is not None and self._model_name in ("auto", "xtts"):
            try:
                neural_audio = self._decode_xtts(
                    wav, reference_audio_path, source_age, target_age, sr
                )
                if neural_audio is not None and self._dsp.quality_check(neural_audio, sr):
                    return DecodeResult(
                        audio=neural_audio,
                        sr=sr,
                        method_used="xtts",
                        quality_ok=True,
                        source_age=source_age,
                        target_age=target_age,
                    )
                else:
                    logger.info("XTTS output failed quality check; falling back to DSP.")
            except Exception as exc:
                logger.warning("XTTS decode error: %s — falling back to DSP.", exc)

        # ----------------------------------------------------------------
        # DSP path (always works)
        # ----------------------------------------------------------------
        dsp_audio = self._dsp.age_voice(wav, sr, source_age, target_age)
        quality_ok = self._dsp.quality_check(dsp_audio, sr)

        if not quality_ok:
            logger.warning(
                "DSP output for age %g→%g failed quality check (RMS too low?).",
                source_age, target_age,
            )

        return DecodeResult(
            audio=dsp_audio,
            sr=sr,
            method_used="dsp",
            quality_ok=quality_ok,
            source_age=source_age,
            target_age=target_age,
        )

    def is_neural_available(self) -> bool:
        """Return True if a neural back-end is loaded."""
        return self._xtts is not None

    # ------------------------------------------------------------------
    # Private — XTTS
    # ------------------------------------------------------------------

    @staticmethod
    def _try_load_xtts():
        """Attempt to import and initialise XTTS-v2.  Returns None on failure."""
        try:
            from TTS.api import TTS  # type: ignore
            logger.info("Loading XTTS-v2 (Coqui TTS)…")
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            logger.info("XTTS-v2 loaded successfully.")
            return tts
        except ModuleNotFoundError:
            logger.info(
                "TTS package not installed; neural voice conversion disabled. "
                "Install with: pip install TTS"
            )
        except Exception as exc:
            logger.info("XTTS-v2 could not be loaded (%s); DSP fallback will be used.", exc)
        return None

    def _decode_xtts(
        self,
        wav: np.ndarray,
        reference_path: str,
        source_age: float,
        target_age: float,
        sr: int,
    ) -> Optional[np.ndarray]:
        """
        Use XTTS voice cloning to generate speech at target characteristics.

        XTTS clones the speaker identity from reference_path, then synthesises
        the same content with that identity.  We then apply DSP age adjustments
        on top to encode the age-specific characteristics.
        """
        if self._xtts is None:
            return None

        # XTTS requires text input.  We use a short neutral sentence as a
        # stand-in since we want to preserve the original voice identity.
        # For voice transformation (not TTS), we apply DSP on the clone.
        import tempfile, os
        placeholder_text = (
            "This is my voice."
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        try:
            self._xtts.tts_to_file(
                text=placeholder_text,
                speaker_wav=str(reference_path),
                language="en",
                file_path=out_path,
            )
            cloned_wav, cloned_sr = librosa.load(out_path, sr=sr, mono=True)
            cloned_wav = cloned_wav.astype(np.float32)

            # Apply age DSP on top of cloned voice
            aged = self._dsp.age_voice(cloned_wav, sr, source_age, target_age)
            return aged
        except Exception as exc:
            logger.warning("XTTS synthesis failed: %s", exc)
            return None
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Private — Audio loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_reference(path: str, sr: int) -> np.ndarray:
        """Load and resample reference audio to target sample rate."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Reference audio not found: {path}")
        try:
            wav, orig_sr = librosa.load(str(p), sr=sr, mono=True)
            return wav.astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"Could not load audio {path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Age characteristic helpers (used externally if needed)
    # ------------------------------------------------------------------

    @staticmethod
    def age_pitch_hz(age: float, gender: str = "unknown") -> float:
        """
        Approximate fundamental frequency (F0) for a given age.
        Based on average F0 measurements across age groups.
        """
        if gender == "female":
            # Female F0: higher baseline, gradual lowering with age
            if age < 12:   return 280.0
            if age < 18:   return 230.0
            if age < 35:   return 210.0
            if age < 65:   return 195.0
            return 190.0
        else:
            # Male / unknown F0 approximations
            if age < 8:    return 265.0
            if age < 14:   return 240.0
            if age < 18:   return 160.0
            if age < 30:   return 120.0
            if age < 50:   return 115.0
            if age < 70:   return 120.0
            return 130.0   # elderly slight raise due to vocal fold stiffening
