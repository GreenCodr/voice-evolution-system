"""
src/voice_age/age/playback.py

AgePlaybackService — master pipeline for Part 2 (Age-Specific Voice Playback).

Public API
----------
    svc = AgePlaybackService(user_id="Harshit")

    result = svc.play_at_age(25.0)
    result = svc.play_at_year(2015)
    result = svc.play_in_future(10.0)

    timeline = svc.get_timeline()
    earliest, latest, predicted_max = svc.get_available_range()

Each call returns an AudioResult dataclass containing:
  - audio       : float32 mono numpy array
  - sr          : sample rate
  - method      : what was done ("direct" | "dsp" | "xtts" | …)
  - source_info : human-readable description of which versions were used
  - confidence  : 0.0–1.0 estimate of output quality
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from voice_age.age.timeline import VoiceTimeline, TimelineEntry, TimelineResult
from voice_age.age.transformer import AgeTransformer
from voice_age.vocoder.voice_decoder import VoiceDecoder
from voice_age.vocoder.dsp_aging import DSPAgingEngine
from voice_age.config import DEFAULT_SR

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[3]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AudioResult:
    """Return value of every AgePlaybackService.play_* call."""
    audio: np.ndarray           # float32 mono
    sr: int                     # sample rate (16 kHz by default)
    target_age: float
    source_age: float
    method: str                 # "direct" | "dsp" | "xtts" | "extrapolate_dsp"
    source_info: str            # human-readable provenance string
    confidence: float           # 0.0–1.0
    versions_used: List[str] = field(default_factory=list)   # version_ids
    interp_weight: Optional[float] = None


# ---------------------------------------------------------------------------
# AgePlaybackService
# ---------------------------------------------------------------------------

class AgePlaybackService:
    """
    Orchestrates the full age-playback pipeline for a single user.

    Parameters
    ----------
    user_id         : must match a JSON file in users/
    date_of_birth   : override stored DOB (optional)
    decoder_model   : passed to VoiceDecoder ("auto" | "xtts" | "dsp")
    """

    # Ages within this distance use the stored recording directly
    _DSP_ONLY_THRESHOLD_YEARS = 2.0

    def __init__(
        self,
        user_id: str,
        date_of_birth: Optional[date] = None,
        decoder_model: str = "auto",
    ):
        self.user_id = user_id
        self._timeline  = VoiceTimeline(user_id, date_of_birth=date_of_birth)
        self._transformer = AgeTransformer()
        self._decoder   = VoiceDecoder(model_name=decoder_model)
        self._dsp       = DSPAgingEngine()

    # ------------------------------------------------------------------
    # Primary entry points
    # ------------------------------------------------------------------

    def play_at_age(self, target_age: float) -> AudioResult:
        """
        Generate audio of the user's voice at target_age.

        Pipeline:
        1. Query VoiceTimeline for nearest version(s)
        2. If empty → raise ValueError
        3. If exact match and age gap ≤ threshold → direct playback
        4. If interpolated (between two recordings) → slerp + DSP/decoder
        5. If future → VoiceEvolutionModel + decoder
        6. If extrapolate (before first recording) → DSP only
        7. Quality check; fall back to pure DSP if needed
        """
        target_age = float(target_age)
        result = self._timeline.get_version_at_age(target_age)
        return self._resolve(result)

    def play_at_year(self, year: int) -> AudioResult:
        """Convert calendar year to age, then delegate to play_at_age."""
        result = self._timeline.get_version_at_year(year)
        return self._resolve(result)

    def play_in_future(self, years_from_now: float) -> AudioResult:
        """Calculate future age, then delegate."""
        result = self._timeline.get_version_in_future(years_from_now)
        return self._resolve(result)

    # ------------------------------------------------------------------
    # Timeline information
    # ------------------------------------------------------------------

    def get_timeline(self) -> List[Dict]:
        """Return all stored versions as dicts (for UI visualisation)."""
        entries = self._timeline.get_full_timeline()
        out = []
        for e in entries:
            out.append({
                "version_id": e.version_id,
                "recorded_utc": e.recorded_utc,
                "age_at_recording": e.age_at_recording,
                "audio_path": str(e.audio_abs_path),
                "confidence": e.confidence,
                "type": e.voice_type,
                "audio_exists": e.audio_abs_path.exists(),
            })
        return out

    def get_available_range(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (earliest_age, latest_age, predicted_max_age)."""
        return self._timeline.get_available_range()

    @property
    def current_age(self) -> Optional[float]:
        return self._timeline.current_age

    @property
    def is_neural_available(self) -> bool:
        return self._decoder.is_neural_available()

    # ------------------------------------------------------------------
    # Internal pipeline resolver
    # ------------------------------------------------------------------

    def _resolve(self, lookup: TimelineResult) -> AudioResult:
        """Dispatch the lookup result to the correct rendering strategy."""

        target_age = lookup.target_age

        # --- No versions stored ---
        if lookup.is_empty:
            raise ValueError(
                f"User '{self.user_id}' has no stored voice versions. "
                "Record at least one voice sample in Part 1 first."
            )

        # --- Exact match (or only one recording) ---
        if lookup.is_exact:
            entry = lookup.nearest
            age_gap = abs(target_age - (entry.age_at_recording or target_age))

            if age_gap < self._DSP_ONLY_THRESHOLD_YEARS:
                # Use stored recording as-is (maybe tiny DSP)
                audio = self._load_audio(entry)
                if audio is None:
                    # Temp file deleted — find any valid audio
                    fallback = self._find_any_valid_entry(preferred=entry)
                    if fallback is not None:
                        audio = self._load_audio(fallback)
                        entry = fallback
                if audio is not None:
                    if age_gap > 0.1:
                        audio = self._dsp.age_voice(
                            audio, DEFAULT_SR,
                            float(entry.age_at_recording),
                            target_age,
                        )
                    return AudioResult(
                        audio=audio,
                        sr=DEFAULT_SR,
                        target_age=target_age,
                        source_age=float(entry.age_at_recording or target_age),
                        method="direct",
                        source_info=(
                            f"Direct playback of recording at age "
                            f"{entry.age_at_recording:.1f} "
                            f"(version {entry.version_id})"
                        ),
                        confidence=entry.confidence,
                        versions_used=[entry.version_id],
                    )

            # Age gap > threshold — still use DSP on the nearest version
            return self._render_dsp_from_nearest(lookup.nearest, target_age, "direct_dsp")

        # --- Between two stored recordings (interpolate) ---
        if lookup.is_interpolated:
            return self._render_interpolated(lookup)

        # --- Future prediction ---
        if lookup.is_future:
            return self._render_future(lookup)

        # --- Before first recording (extrapolate backwards) ---
        if lookup.is_extrapolate:
            return self._render_dsp_from_nearest(
                lookup.nearest, target_age, "extrapolate_dsp"
            )

        # Fallback (shouldn't reach here)
        return self._render_dsp_from_nearest(lookup.nearest, target_age, "dsp")

    # ------------------------------------------------------------------
    # Rendering strategies
    # ------------------------------------------------------------------

    def _render_dsp_from_nearest(
        self,
        entry: TimelineEntry,
        target_age: float,
        method_label: str,
    ) -> AudioResult:
        """Apply DSP aging on the nearest stored recording.

        If the primary entry's audio is missing, walks all stored versions to
        find any valid audio file (robustness for deleted temp files).
        """
        audio = self._load_audio(entry)
        if audio is None:
            # Primary entry audio gone — find any valid version
            entry = self._find_any_valid_entry(preferred=entry)
            if entry is None:
                raise FileNotFoundError(
                    "No stored audio file found on disk for any version. "
                    "Please upload a new recording in Part 1."
                )
            audio = self._load_audio(entry)
            if audio is None:
                raise FileNotFoundError(
                    "No stored audio file could be loaded. "
                    "Please upload a new recording in Part 1."
                )

        source_age = float(entry.age_at_recording or target_age)
        aged_audio = self._dsp.age_voice(audio, DEFAULT_SR, source_age, target_age)

        quality_ok = self._dsp.quality_check(aged_audio, DEFAULT_SR)
        confidence = entry.confidence * (0.9 if quality_ok else 0.5)

        return AudioResult(
            audio=aged_audio,
            sr=DEFAULT_SR,
            target_age=target_age,
            source_age=source_age,
            method=method_label,
            source_info=(
                f"DSP aging applied to recording at age {source_age:.1f} "
                f"(version {entry.version_id}) → target age {target_age:.1f}"
            ),
            confidence=round(confidence, 3),
            versions_used=[entry.version_id],
        )

    def _render_interpolated(self, lookup: TimelineResult) -> AudioResult:
        """Interpolate between two stored recordings."""
        lower = lookup.lower
        upper = lookup.upper
        weight = lookup.interp_weight
        target_age = lookup.target_age

        transform_result = self._transformer.interpolate_between_versions(
            version_young=lower,
            version_old=upper,
            target_age=target_age,
            interp_weight=weight,
        )

        decode_result = self._decoder.decode(
            reference_audio_path=transform_result.reference_audio_path,
            source_age=transform_result.source_age,
            target_age=target_age,
        )

        if not decode_result.quality_ok:
            logger.warning("Decoder output failed QC; re-running with pure DSP.")
            return self._render_dsp_from_nearest(lookup.nearest, target_age, "dsp_fallback")

        confidence = (
            lower.confidence * (1.0 - weight)
            + upper.confidence * weight
        ) * (1.0 if decode_result.quality_ok else 0.5)

        return AudioResult(
            audio=decode_result.audio,
            sr=decode_result.sr,
            target_age=target_age,
            source_age=transform_result.source_age,
            method=decode_result.method_used,
            source_info=(
                f"Interpolated (weight={weight:.2f}) between recordings at "
                f"age {lower.age_at_recording:.1f} and {upper.age_at_recording:.1f} "
                f"via {transform_result.method}"
            ),
            confidence=round(float(confidence), 3),
            versions_used=[lower.version_id, upper.version_id],
            interp_weight=weight,
        )

    def _render_future(self, lookup: TimelineResult) -> AudioResult:
        """Generate a future-age voice using the VoiceEvolutionModel + decoder."""
        latest = lookup.nearest
        target_age = lookup.target_age
        current_age = float(latest.age_at_recording or self._timeline.current_age or 25.0)

        transform_result = self._transformer.predict_future_voice(
            latest_version=latest,
            current_age=current_age,
            target_age=target_age,
        )

        decode_result = self._decoder.decode(
            reference_audio_path=transform_result.reference_audio_path,
            source_age=transform_result.source_age,
            target_age=target_age,
        )

        if not decode_result.quality_ok:
            logger.warning("Future decoder output failed QC; using pure DSP fallback.")
            return self._render_dsp_from_nearest(latest, target_age, "future_dsp_fallback")

        years_ahead = target_age - current_age
        confidence = latest.confidence * max(0.3, 1.0 - years_ahead / 80.0)

        return AudioResult(
            audio=decode_result.audio,
            sr=decode_result.sr,
            target_age=target_age,
            source_age=transform_result.source_age,
            method=f"future_{decode_result.method_used}",
            source_info=(
                f"Future prediction: {years_ahead:.1f} years ahead of latest recording "
                f"(age {current_age:.1f}) using {transform_result.method}"
            ),
            confidence=round(float(confidence), 3),
            versions_used=[latest.version_id],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_audio(entry: TimelineEntry) -> Optional[np.ndarray]:
        """Load audio from a TimelineEntry.  Returns None if file missing."""
        p = entry.audio_abs_path
        if not p.exists():
            logger.warning("Audio file missing: %s", p)
            return None
        try:
            import librosa
            wav, _ = librosa.load(str(p), sr=DEFAULT_SR, mono=True)
            return wav.astype(np.float32)
        except Exception as exc:
            logger.error("Could not load audio %s: %s", p, exc)
            return None

    def _find_any_valid_entry(
        self, preferred: Optional[TimelineEntry] = None
    ) -> Optional[TimelineEntry]:
        """
        Walk all stored versions and return the first one whose audio file
        exists on disk.  Prefers the entry closest to preferred.age_at_recording
        if provided.
        """
        all_entries = self._timeline.get_full_timeline()
        valid = [e for e in all_entries if e.audio_abs_path.exists()]
        if not valid:
            return None
        if preferred is None or preferred.age_at_recording is None:
            return valid[-1]  # latest
        # Sort by proximity to preferred age
        ref_age = float(preferred.age_at_recording)
        valid.sort(key=lambda e: abs((e.age_at_recording or 0) - ref_age))
        return valid[0]

    # ------------------------------------------------------------------
    # Convenience: parse natural language age queries
    # ------------------------------------------------------------------

    def parse_and_play(self, query: str) -> AudioResult:
        """
        Parse a natural-language age query and return audio.

        Supported patterns:
          "age 25"       → play_at_age(25)
          "at 25"        → play_at_age(25)
          "year 2015"    → play_at_year(2015)
          "in 2015"      → play_at_year(2015)
          "10 years from now" → play_in_future(10)
          "in 10 years"       → play_in_future(10)
        """
        q = query.lower().strip()

        # "age X" or "at age X"
        m = re.search(r"(?:age|at)\s+(\d+(?:\.\d+)?)", q)
        if m:
            return self.play_at_age(float(m.group(1)))

        # "in YYYY" or "year YYYY" (four-digit year)
        m = re.search(r"(?:in\s+|year\s+)((?:19|20)\d{2})", q)
        if m:
            return self.play_at_year(int(m.group(1)))

        # "X years from now"
        m = re.search(r"(\d+(?:\.\d+)?)\s+years?\s+from\s+now", q)
        if m:
            return self.play_in_future(float(m.group(1)))

        # "in X years"
        m = re.search(r"in\s+(\d+(?:\.\d+)?)\s+years?", q)
        if m:
            return self.play_in_future(float(m.group(1)))

        # Plain number → treat as age
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", q)
        if m:
            return self.play_at_age(float(m.group(1)))

        raise ValueError(
            f"Could not parse age query: '{query}'. "
            "Try: 'age 25', 'year 2015', '10 years from now'."
        )

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def post_process(audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Normalize and trim trailing silence.
        """
        # Trim leading/trailing silence
        try:
            audio, _ = librosa_trim(audio, sr)
        except Exception:
            pass

        # Normalize to -1 dB peak
        mx = float(np.max(np.abs(audio)) + 1e-9)
        if mx > 0:
            audio = audio / mx * 0.9

        return audio.astype(np.float32)


def librosa_trim(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, ...]:
    """Trim leading/trailing silence with librosa.effects.trim."""
    import librosa
    trimmed, idx = librosa.effects.trim(audio, top_db=30)
    return trimmed, idx
