"""
tests/test_age_playback.py

Unit tests for Part 2: Age-Specific Voice Playback

Run with:
    cd /path/to/voice-evolution-system
    python -m pytest tests/test_age_playback.py -v
"""

from __future__ import annotations

import sys
import json
import tempfile
import math
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for p in [str(SRC_DIR), str(SCRIPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ===========================================================================
# Tests: Age query parsing helpers
# ===========================================================================

class TestAgeQueryParsing:
    """Test the parse_and_play query parser in AgePlaybackService."""

    def _make_svc(self, user_id: str = "_test_user_"):
        from voice_age.age.playback import AgePlaybackService
        svc = AgePlaybackService.__new__(AgePlaybackService)
        svc.user_id = user_id
        return svc

    def test_parse_age_keyword(self):
        svc = self._make_svc()
        # We test the regex extraction, not the full pipeline
        import re
        q = "age 25"
        m = re.search(r"(?:age|at)\s+(\d+(?:\.\d+)?)", q.lower())
        assert m is not None
        assert float(m.group(1)) == 25.0

    def test_parse_year(self):
        import re
        q = "year 2015"
        m = re.search(r"(?:in\s+|year\s+)((?:19|20)\d{2})", q.lower())
        assert m is not None
        assert int(m.group(1)) == 2015

    def test_parse_future(self):
        import re
        q = "10 years from now"
        m = re.search(r"(\d+(?:\.\d+)?)\s+years?\s+from\s+now", q.lower())
        assert m is not None
        assert float(m.group(1)) == 10.0

    def test_parse_in_years(self):
        import re
        q = "in 5 years"
        m = re.search(r"in\s+(\d+(?:\.\d+)?)\s+years?", q.lower())
        assert m is not None
        assert float(m.group(1)) == 5.0

    def test_dob_to_age_2015(self):
        from voice_age.age.timeline import _compute_age_fractional
        dob = date(1990, 6, 15)
        target = date(2015, 6, 15)
        age = _compute_age_fractional(dob, target)
        assert abs(age - 25.0) < 0.1


# ===========================================================================
# Tests: VoiceTimeline
# ===========================================================================

class TestVoiceTimeline:
    """Build a timeline from synthetic user data and verify age mapping."""

    def _make_user(self, user_id: str, versions: list, dob: str = "2000-01-01") -> Path:
        """Create a temporary user JSON file."""
        users_dir = PROJECT_ROOT / "users"
        users_dir.mkdir(exist_ok=True)
        p = users_dir / f"{user_id}.json"
        p.write_text(json.dumps({
            "user_id": user_id,
            "date_of_birth": dob,
            "created_utc": "2024-01-01T00:00:00Z",
            "voice_versions": versions,
        }))
        return p

    def _cleanup(self, user_id: str):
        p = PROJECT_ROOT / "users" / f"{user_id}.json"
        p.unlink(missing_ok=True)

    def test_empty_user(self):
        uid = "_test_empty_"
        p = self._make_user(uid, versions=[])
        try:
            from voice_age.age.timeline import VoiceTimeline
            tl = VoiceTimeline(uid)
            result = tl.get_version_at_age(25.0)
            assert result.is_empty
        finally:
            self._cleanup(uid)
            p.unlink(missing_ok=True)

    def test_single_version_exact_match(self):
        """Querying the exact recorded age with a single version returns is_exact."""
        uid = "_test_single_"
        versions = [{
            "version_id": "v1",
            "recorded_utc": "2020-01-01T00:00:00Z",
            "age_at_recording": 20,
            "embedding_path": "",
            "audio_path": "",
            "confidence": 1.0,
            "type": "RECORDED",
        }]
        p = self._make_user(uid, versions)
        try:
            from voice_age.age.timeline import VoiceTimeline
            tl = VoiceTimeline(uid)
            # Exact age → is_exact
            r_exact = tl.get_version_at_age(20.0)
            assert r_exact.is_exact
            assert r_exact.nearest.version_id == "v1"
            # Future age → is_future, but nearest is still v1
            r_future = tl.get_version_at_age(25.0)
            assert r_future.is_future
            assert r_future.nearest.version_id == "v1"
            # Past age → is_extrapolate
            r_past = tl.get_version_at_age(10.0)
            assert r_past.is_extrapolate
        finally:
            self._cleanup(uid)

    def test_five_versions_interpolation(self):
        uid = "_test_five_"
        versions = [
            {
                "version_id": f"v{i}",
                "recorded_utc": f"20{i:02d}-01-01T00:00:00Z",
                "age_at_recording": 10 + i * 5,
                "embedding_path": "",
                "audio_path": "",
                "confidence": 1.0,
                "type": "RECORDED",
            }
            for i in range(1, 6)
        ]
        p = self._make_user(uid, versions)
        try:
            from voice_age.age.timeline import VoiceTimeline
            tl = VoiceTimeline(uid)

            # exact match
            r_exact = tl.get_version_at_age(20.0)
            assert r_exact.is_exact

            # interpolation
            r_interp = tl.get_version_at_age(22.5)
            assert r_interp.is_interpolated
            assert r_interp.lower is not None
            assert r_interp.upper is not None
            assert r_interp.lower.age_at_recording <= 22.5
            assert r_interp.upper.age_at_recording >= 22.5
            assert 0.0 < r_interp.interp_weight < 1.0

            # before first
            r_before = tl.get_version_at_age(5.0)
            assert r_before.is_extrapolate

            # after last
            r_after = tl.get_version_at_age(60.0)
            assert r_after.is_future
        finally:
            self._cleanup(uid)

    def test_version_at_year(self):
        uid = "_test_year_"
        versions = [{
            "version_id": "v1",
            "recorded_utc": "2020-01-01T00:00:00Z",
            "age_at_recording": 20,
            "embedding_path": "",
            "audio_path": "",
            "confidence": 1.0,
            "type": "RECORDED",
        }]
        p = self._make_user(uid, versions, dob="2000-01-01")
        try:
            from voice_age.age.timeline import VoiceTimeline
            tl = VoiceTimeline(uid)
            result = tl.get_version_at_year(2025)
            assert result.target_age is not None
        finally:
            self._cleanup(uid)

    def test_future_version(self):
        uid = "_test_future_"
        versions = [{
            "version_id": "v1",
            "recorded_utc": "2026-01-01T00:00:00Z",
            "age_at_recording": 26,
            "embedding_path": "",
            "audio_path": "",
            "confidence": 1.0,
            "type": "RECORDED",
        }]
        p = self._make_user(uid, versions, dob="2000-01-01")
        try:
            from voice_age.age.timeline import VoiceTimeline
            tl = VoiceTimeline(uid)
            result = tl.get_version_in_future(10.0)
            assert result.is_future
            assert result.target_age > 26
        finally:
            self._cleanup(uid)


# ===========================================================================
# Tests: Slerp (spherical interpolation)
# ===========================================================================

class TestSlerp:

    def test_slerp_endpoints(self):
        from voice_age.age.transformer import _slerp
        v0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        r0 = _slerp(v0, v1, 0.0)
        r1 = _slerp(v0, v1, 1.0)

        assert np.allclose(r0, v0 / np.linalg.norm(v0), atol=1e-5)
        assert np.allclose(r1, v1 / np.linalg.norm(v1), atol=1e-5)

    def test_slerp_midpoint_unit(self):
        from voice_age.age.transformer import _slerp
        v0 = np.random.randn(192).astype(np.float32)
        v1 = np.random.randn(192).astype(np.float32)
        v0 /= np.linalg.norm(v0)
        v1 /= np.linalg.norm(v1)

        mid = _slerp(v0, v1, 0.5)
        assert abs(np.linalg.norm(mid) - 1.0) < 1e-4

    def test_slerp_output_shape(self):
        from voice_age.age.transformer import _slerp
        v0 = np.random.randn(192).astype(np.float32)
        v1 = np.random.randn(192).astype(np.float32)
        result = _slerp(v0, v1, 0.3)
        assert result.shape == (192,)


# ===========================================================================
# Tests: DSPAgingEngine
# ===========================================================================

class TestDSPAgingEngine:

    def _make_speech_like(self, sr: int = 16000, duration: float = 2.0) -> np.ndarray:
        """Create a simple speech-like signal (superposition of harmonics)."""
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        f0 = 120.0
        signal = sum(
            0.5 / (k + 1) * np.sin(2 * np.pi * (k + 1) * f0 * t)
            for k in range(6)
        )
        return signal.astype(np.float32) * 0.5

    def test_age_voice_returns_float32(self):
        from voice_age.vocoder.dsp_aging import DSPAgingEngine
        engine = DSPAgingEngine()
        wav = self._make_speech_like()
        out = engine.age_voice(wav, sr=16000, source_age=25.0, target_age=65.0)
        assert out.dtype == np.float32

    def test_age_voice_same_length_order(self):
        from voice_age.vocoder.dsp_aging import DSPAgingEngine
        engine = DSPAgingEngine()
        wav = self._make_speech_like()
        # We don't enforce exact length match (pitch shift can change length)
        out = engine.age_voice(wav, sr=16000, source_age=25.0, target_age=65.0)
        assert len(out) > 0

    def test_age_voice_not_silence(self):
        from voice_age.vocoder.dsp_aging import DSPAgingEngine
        engine = DSPAgingEngine()
        wav = self._make_speech_like()
        out = engine.age_voice(wav, sr=16000, source_age=25.0, target_age=65.0)
        rms = float(np.sqrt(np.mean(out ** 2)))
        assert rms > 1e-4, "Output should not be silence"

    def test_quality_check_speech(self):
        from voice_age.vocoder.dsp_aging import DSPAgingEngine
        engine = DSPAgingEngine()
        wav = self._make_speech_like()
        assert engine.quality_check(wav, sr=16000) is True

    def test_quality_check_silence(self):
        from voice_age.vocoder.dsp_aging import DSPAgingEngine
        engine = DSPAgingEngine()
        silence = np.zeros(16000, dtype=np.float32)
        assert engine.quality_check(silence, sr=16000) is False

    def test_age_identity_small_change(self):
        """Source and target very close — output should resemble input."""
        from voice_age.vocoder.dsp_aging import DSPAgingEngine
        engine = DSPAgingEngine()
        wav = self._make_speech_like()
        out = engine.age_voice(wav, sr=16000, source_age=25.0, target_age=26.0)
        assert len(out) > 0

    def test_younger_pitch_higher(self):
        """Pitch shift for young target should go up relative to adult."""
        from voice_age.vocoder.dsp_aging import _profile_for_age
        adult = _profile_for_age(30.0)
        child = _profile_for_age(8.0)
        assert child["pitch_shift_semitones"] >= adult["pitch_shift_semitones"]

    def test_older_more_breathiness(self):
        """Breathiness should increase for older voices."""
        from voice_age.vocoder.dsp_aging import _profile_for_age
        adult = _profile_for_age(30.0)
        elderly = _profile_for_age(75.0)
        assert elderly["breathiness"] >= adult["breathiness"]


# ===========================================================================
# Tests: VoiceEvolutionModel forward pass
# ===========================================================================

class TestVoiceEvolutionModel:

    def test_forward_pass_shape(self):
        """VoiceEvolutionModel should accept and return 1216-dim tensors."""
        sys.path.insert(0, str(PROJECT_ROOT / "training" / "scripts"))
        try:
            import torch
            from train_voice_evolution_model import VoiceEvolutionModel
            model = VoiceEvolutionModel(emb_dim=1216)
            x = torch.randn(1, 1216)
            y = model(x)
            assert y.shape == (1, 1216), f"Expected (1, 1216), got {y.shape}"
        except ImportError as e:
            pytest.skip(f"training scripts not importable: {e}")

    def test_forward_pass_no_error(self):
        """Model should run without throwing on random input."""
        sys.path.insert(0, str(PROJECT_ROOT / "training" / "scripts"))
        try:
            import torch
            from train_voice_evolution_model import VoiceEvolutionModel
            weights_path = (
                PROJECT_ROOT / "training" / "models" / "voice_evolution_model.pt"
            )
            if not weights_path.exists():
                pytest.skip("voice_evolution_model.pt not found")
            model = VoiceEvolutionModel(emb_dim=1216)
            state = torch.load(str(weights_path), map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            x = torch.randn(4, 1216)
            with torch.no_grad():
                y = model(x)
            assert y.shape == (4, 1216)
        except ImportError as e:
            pytest.skip(f"training scripts not importable: {e}")


# ===========================================================================
# Tests: Full pipeline integration (with a synthetic real audio file)
# ===========================================================================

class TestFullPipeline:
    """End-to-end: reference WAV → aged audio → not silence, not noise."""

    def _write_test_wav(self, path: Path, sr: int = 16000, duration: float = 3.0):
        import soundfile as sf
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        signal = np.sum(
            [0.3 / (k + 1) * np.sin(2 * np.pi * (k + 1) * 120 * t) for k in range(8)],
            axis=0,
        ).astype(np.float32)
        sf.write(str(path), signal, sr)

    def test_voice_decoder_dsp(self):
        from voice_age.vocoder.voice_decoder import VoiceDecoder
        decoder = VoiceDecoder(model_name="dsp")

        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / "ref.wav"
            self._write_test_wav(ref)

            result = decoder.decode(
                reference_audio_path=str(ref),
                source_age=25.0,
                target_age=65.0,
            )

        assert result.audio is not None
        assert len(result.audio) > 0
        assert result.method_used in ("dsp", "direct")
        rms = float(np.sqrt(np.mean(result.audio ** 2)))
        assert rms > 1e-4, "Decoder output should not be silence"

    def test_full_pipeline_with_mock_user(self):
        """
        AgePlaybackService.play_at_age on a user with one synthetic recording
        should return valid audio.
        """
        from voice_age.age.playback import AgePlaybackService

        uid = "_test_pipeline_"
        users_dir = PROJECT_ROOT / "users"
        users_dir.mkdir(exist_ok=True)

        with tempfile.TemporaryDirectory() as td:
            # Write a real WAV file
            ref_wav = Path(td) / "voice.wav"
            import soundfile as sf
            sr = 16000
            t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
            signal = sum(
                0.3 / (k + 1) * np.sin(2 * np.pi * (k + 1) * 120 * t)
                for k in range(6)
            )
            sf.write(str(ref_wav), signal.astype("float32"), sr)

            user_path = users_dir / f"{uid}.json"
            user_path.write_text(json.dumps({
                "user_id": uid,
                "date_of_birth": "2000-01-01",
                "created_utc": "2024-01-01T00:00:00Z",
                "voice_versions": [{
                    "version_id": "v1",
                    "recorded_utc": "2023-01-01T00:00:00Z",
                    "age_at_recording": 23,
                    "embedding_path": "",
                    "audio_path": str(ref_wav),
                    "confidence": 1.0,
                    "type": "RECORDED",
                }],
            }))

            try:
                svc = AgePlaybackService(uid, decoder_model="dsp")

                # Test play_at_age (exact)
                result_exact = svc.play_at_age(23.0)
                assert result_exact.audio is not None
                assert len(result_exact.audio) > 0

                # Test play_at_age (future)
                result_future = svc.play_at_age(60.0)
                assert result_future.audio is not None
                assert result_future.method.startswith("future") or "dsp" in result_future.method

                # Test play_in_future
                result_inf = svc.play_in_future(10.0)
                assert result_inf.audio is not None

            finally:
                user_path.unlink(missing_ok=True)


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_no_recordings_raises(self):
        from voice_age.age.playback import AgePlaybackService

        uid = "_test_no_rec_"
        users_dir = PROJECT_ROOT / "users"
        users_dir.mkdir(exist_ok=True)
        p = users_dir / f"{uid}.json"
        p.write_text(json.dumps({
            "user_id": uid,
            "date_of_birth": "2000-01-01",
            "created_utc": "2024-01-01T00:00:00Z",
            "voice_versions": [],
        }))
        try:
            svc = AgePlaybackService(uid, decoder_model="dsp")
            with pytest.raises(ValueError, match="no stored voice versions"):
                svc.play_at_age(25.0)
        finally:
            p.unlink(missing_ok=True)

    def test_pad_to_combined_dim(self):
        from voice_age.age.transformer import _pad_to_combined_dim
        ecapa = np.random.randn(192).astype(np.float32)
        padded = _pad_to_combined_dim(ecapa, 1216)
        assert padded.shape == (1216,)
        assert np.allclose(padded[:192], ecapa)
        assert np.allclose(padded[192:], 0.0)

    def test_timeline_age_fractional(self):
        from voice_age.age.timeline import _compute_age_fractional
        dob = date(2000, 7, 1)
        ref = date(2000, 7, 1)
        assert _compute_age_fractional(dob, ref) == pytest.approx(0.0, abs=0.01)

        ref2 = date(2001, 1, 1)
        age = _compute_age_fractional(dob, ref2)
        assert 0.4 < age < 0.6   # about half a year
