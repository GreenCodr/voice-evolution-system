"""
src/voice_age/age/transformer.py

AgeTransformer — transforms voice embeddings to represent a different age.

Two strategies:
  1. interpolate_between_versions — for ages BETWEEN two stored recordings:
       slerp (spherical interpolation) in ECAPA embedding space, weighted by
       age distance from each recording.

  2. predict_future_voice — for ages BEYOND stored recordings (future):
       runs the trained VoiceEvolutionModel to shift the latest embedding
       toward an "older" direction, then clamps the shift by years_delta.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from voice_age.age.timeline import TimelineEntry

logger = logging.getLogger(__name__)

# Project root (voices-evolution-system/)
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[3]

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class TransformResult:
    """Output of AgeTransformer — the reference audio to feed the decoder,
    plus any transformed embedding that characterises the target age."""

    reference_audio_path: str       # path to the closest stored WAV
    source_age: float               # age of the reference recording
    target_age: float               # requested target age

    # ECAPA embedding for the target age (192-dim numpy float32)
    ecapa_embedding: Optional[np.ndarray] = None

    # Whether we used the VoiceEvolutionModel (future) or slerp (interpolation)
    method: str = "dsp_only"        # "slerp" | "future_model" | "dsp_only"

    # Interpolation weight used (0 → lower, 1 → upper)
    interp_weight: float = 0.0


# ---------------------------------------------------------------------------
# AgeTransformer
# ---------------------------------------------------------------------------

class AgeTransformer:
    """
    Handles embedding-space transformation for age rendering.

    Parameters
    ----------
    evolution_model_path : str | Path | None
        Path to the trained VoiceEvolutionModel weights
        (default: training/models/voice_evolution_model.pt).
        If the file is missing the model is skipped gracefully.
    """

    _EMB_DIM_COMBINED = 1216    # 192 ECAPA + 1024 HuBERT
    _EMB_DIM_ECAPA    = 192

    def __init__(
        self,
        evolution_model_path: Optional[str | Path] = None,
    ):
        self._evolution_model = None

        default_path = PROJECT_ROOT / "training" / "models" / "voice_evolution_model.pt"
        model_path = Path(evolution_model_path) if evolution_model_path else default_path

        if model_path.exists():
            try:
                self._evolution_model = self._load_evolution_model(model_path)
                logger.info("VoiceEvolutionModel loaded from %s", model_path)
            except Exception as exc:
                logger.warning("Could not load VoiceEvolutionModel: %s", exc)
        else:
            logger.warning(
                "VoiceEvolutionModel weights not found at %s — "
                "future prediction will fall back to DSP only.",
                model_path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpolate_between_versions(
        self,
        version_young: TimelineEntry,
        version_old: TimelineEntry,
        target_age: float,
        interp_weight: float,
    ) -> TransformResult:
        """
        Slerp between the ECAPA embeddings of two stored versions.

        Parameters
        ----------
        version_young : nearest older version with age ≤ target_age
        version_old   : nearest newer version with age ≥ target_age
        target_age    : the requested age
        interp_weight : 0.0 → entirely young, 1.0 → entirely old
        """
        emb_young = self._load_ecapa_embedding(version_young)
        emb_old   = self._load_ecapa_embedding(version_old)

        if emb_young is not None and emb_old is not None:
            interp_emb = _slerp(emb_young, emb_old, interp_weight)
            method = "slerp"
        else:
            # If either embedding is missing, use nearest version's embedding
            interp_emb = emb_young if emb_young is not None else emb_old
            method = "dsp_only"
            logger.warning(
                "One or both ECAPA embeddings missing for slerp; "
                "falling back to DSP."
            )

        # Reference audio = the version closest to target_age
        ref_version = version_young if interp_weight <= 0.5 else version_old
        return TransformResult(
            reference_audio_path=str(ref_version.audio_abs_path),
            source_age=float(ref_version.age_at_recording),
            target_age=float(target_age),
            ecapa_embedding=interp_emb,
            method=method,
            interp_weight=interp_weight,
        )

    def predict_future_voice(
        self,
        latest_version: TimelineEntry,
        current_age: float,
        target_age: float,
    ) -> TransformResult:
        """
        Project the latest embedding into the future using the trained
        VoiceEvolutionModel.

        The model predicts a single "ageing step".  For large age gaps we
        apply it proportionally (blended with the original embedding).

        Parameters
        ----------
        latest_version : most recently stored version
        current_age    : age at the latest recording
        target_age     : requested future age
        """
        ecapa_emb = self._load_ecapa_embedding(latest_version)
        years_delta = max(0.0, target_age - current_age)

        if self._evolution_model is None or ecapa_emb is None:
            logger.info(
                "Future prediction falling back to DSP only "
                "(model=%s, emb=%s)",
                self._evolution_model is not None,
                ecapa_emb is not None,
            )
            return TransformResult(
                reference_audio_path=str(latest_version.audio_abs_path),
                source_age=float(current_age),
                target_age=float(target_age),
                ecapa_embedding=ecapa_emb,
                method="dsp_only",
            )

        # The VoiceEvolutionModel was trained on ECAPA (192-dim), but the
        # training script used combined 1216-dim.  We pad with zeros if needed.
        emb_input = _pad_to_combined_dim(ecapa_emb, self._EMB_DIM_COMBINED)
        evolved   = self._apply_evolution_model(emb_input, years_delta)

        # Return only ECAPA-sized slice so downstream is consistent
        evolved_ecapa = evolved[: self._EMB_DIM_ECAPA].astype(np.float32)

        return TransformResult(
            reference_audio_path=str(latest_version.audio_abs_path),
            source_age=float(current_age),
            target_age=float(target_age),
            ecapa_embedding=evolved_ecapa,
            method="future_model",
        )

    def adjust_nearby_version(
        self,
        nearest_version: TimelineEntry,
        target_age: float,
    ) -> TransformResult:
        """
        For ages very close to a stored version (±2 years), return the
        stored version as-is; the DSP engine will handle the tiny adjustment.
        """
        ecapa_emb = self._load_ecapa_embedding(nearest_version)
        return TransformResult(
            reference_audio_path=str(nearest_version.audio_abs_path),
            source_age=float(nearest_version.age_at_recording or target_age),
            target_age=float(target_age),
            ecapa_embedding=ecapa_emb,
            method="dsp_only",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_ecapa_embedding(entry: TimelineEntry) -> Optional[np.ndarray]:
        """Load ECAPA embedding from disk.  Returns None if not found."""
        p = entry.embedding_abs_path
        if p is None or not p.exists():
            logger.debug("Embedding not found on disk: %s", p)
            return None
        try:
            emb = np.load(p).astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 1e-9 else emb
        except Exception as exc:
            logger.warning("Could not load embedding %s: %s", p, exc)
            return None

    @staticmethod
    def _load_evolution_model(path: Path):
        """Load VoiceEvolutionModel from training directory."""
        import sys
        scripts_path = str(PROJECT_ROOT / "training" / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

        from train_voice_evolution_model import VoiceEvolutionModel  # type: ignore

        model = VoiceEvolutionModel(emb_dim=1216)
        state = torch.load(str(path), map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    def _apply_evolution_model(
        self, emb: np.ndarray, years_delta: float
    ) -> np.ndarray:
        """
        Apply the VoiceEvolutionModel proportionally.

        For a 1-year shift we fully apply the model delta.
        For larger shifts we linearly scale the delta, capped at 40 years.
        """
        MAX_YEARS = 40.0
        scale = min(years_delta / max(years_delta, 1.0), 1.0)
        # Clamp extreme futures
        if years_delta > MAX_YEARS:
            scale = 1.0

        tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            evolved = self._evolution_model(tensor).squeeze(0).numpy()

        # Blend between original and fully-evolved based on years_delta
        blended = (1.0 - scale) * emb + scale * evolved
        # Re-normalise
        norm = np.linalg.norm(blended)
        return blended / norm if norm > 1e-9 else blended


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def _slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two unit vectors.
    t=0 → v0,  t=1 → v1.
    """
    v0 = v0 / (np.linalg.norm(v0) + 1e-9)
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    dot = float(np.clip(np.dot(v0, v1), -1.0, 1.0))

    # If vectors are nearly identical, fall back to linear
    if abs(dot) > 0.9995:
        result = (1.0 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta = np.sin(theta_0)
        result = (
            np.sin((1.0 - t) * theta_0) / sin_theta * v0
            + np.sin(t * theta_0) / sin_theta * v1
        )

    norm = np.linalg.norm(result)
    return (result / norm).astype(np.float32)


def _pad_to_combined_dim(ecapa: np.ndarray, target_dim: int = 1216) -> np.ndarray:
    """
    Pad a 192-dim ECAPA vector to target_dim by appending zeros.
    If ecapa is already ≥ target_dim, truncate.
    """
    current = ecapa.shape[0]
    if current >= target_dim:
        return ecapa[:target_dim].astype(np.float32)
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[:current] = ecapa
    return padded
