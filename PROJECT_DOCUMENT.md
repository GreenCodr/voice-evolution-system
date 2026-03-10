# Voice Evolution System — Complete Project Document

**Purpose of this document:** Full technical reference covering every directory, every file, every decision, every bug fix, and every future improvement opportunity. Written so a new AI assistant can continue this project without any prior context.

---

## 1. What This Project Does

The Voice Evolution System is a personal voice-tracking and voice-aging application. It has two independently working parts:

**Part 1 — Voice Change Detection (COMPLETE, DO NOT TOUCH)**
Records voice samples over time, detects when the voice has meaningfully changed (puberty, age, illness, etc.), and stores each distinct voice as a versioned snapshot with a timestamp and age tag.

**Part 2 — Age-Specific Voice Playback (COMPLETE)**
Given a user's stored voice recordings, generates audio of what their voice sounds like at any requested age — past (before recordings), between recordings, current, or future (predicted). Example queries:
- "Play my voice at age 10" → applies DSP to the earliest recording to simulate a younger voice
- "Play my voice 20 years from now" → uses VoiceEvolutionModel + DSP to predict an older voice
- "Play my voice as it sounded in 2015" → converts year to age using DOB, finds nearest recording

---

## 2. Technology Stack

| Technology | Version | Role |
|---|---|---|
| Python | 3.10.19 | Runtime |
| PyTorch | 2.1.2 | Deep learning framework |
| torchaudio | 2.1.2 | Audio tensor processing |
| SpeechBrain | 0.5.16 | ECAPA-TDNN speaker embedding |
| Transformers | 4.36.2 | HuBERT feature extraction |
| librosa | latest | Audio I/O and DSP |
| soundfile | latest | WAV read/write |
| scipy | latest | Butter filters, signal processing |
| numpy | <2 | Array math |
| streamlit | 1.30.0 | Web frontend |
| PyYAML | latest | Config files |
| praat-parselmouth | latest | Formant analysis (available, not yet used in Part 2) |
| faiss | optional | Embedding similarity search (Part 1) |

Virtual environment: `.venv/` (Python 3.10, pip-based)

Run app: `streamlit run frontend/app.py` from the project root directory.

Run tests: `python -m pytest tests/test_age_playback.py -v` from project root.

---

## 3. Complete Directory Tree

```
voice-evolution-system/
│
├── frontend/
│   └── app.py                         # Streamlit web app (both Part 1 and Part 2 UI)
│
├── src/
│   └── voice_age/
│       ├── __init__.py                # Empty package marker
│       ├── config.py                  # DEFAULT_SR=16000, MIN_AGE=5, MAX_AGE=70, BASE_AGE=23
│       ├── cli.py                     # Command-line entry point (not used in app)
│       │
│       ├── age/
│       │   ├── timeline.py            # [PART 2] VoiceTimeline — age lookup
│       │   ├── transformer.py         # [PART 2] AgeTransformer — slerp + VoiceEvolutionModel
│       │   ├── playback.py            # [PART 2] AgePlaybackService — master pipeline
│       │   ├── age_encoder.py         # Empty stub (unused)
│       │   └── dsp_age.py             # Original simple DSP (used by old code, still present)
│       │
│       ├── vocoder/
│       │   ├── dsp_aging.py           # [PART 2] DSPAgingEngine — clean pitch/spectral aging
│       │   ├── voice_decoder.py       # [PART 2] VoiceDecoder — DSP + optional XTTS
│       │   └── hifigan.py             # HiFi-GAN vocoder wrapper (mel→audio, used by old code)
│       │
│       ├── models/
│       │   └── mel_generator.py       # BROKEN: randomly initialized MelGenerator, DO NOT USE
│       │
│       ├── engines/
│       │   └── post.py                # Post-processing stub (RVC placeholder, not used)
│       │
│       ├── io/
│       │   └── audio.py               # load_audio(), write_wav(), duration_seconds()
│       │
│       └── tools/
│           └── generate_samples.py    # Batch sample generation script (standalone)
│
├── scripts/
│   ├── process_new_voice.py           # [PART 1] Main voice ingestion pipeline
│   ├── user_registry.py               # [PART 1] UserRegistry class + legacy load/save helpers
│   ├── faiss_change_detector.py       # [PART 1] FAISS-based embedding change detector
│   ├── embed_ecapa.py                 # ECAPA embedding extraction from audio file
│   ├── audio_preprocess.py            # Normalize audio to 16kHz mono WAV
│   ├── audio_quality.py               # SNR, duration quality gate
│   ├── speaker_verification.py        # Speaker identity gate (cosine similarity)
│   ├── confidence_engine.py           # Confidence score from quality + similarity + device
│   ├── version_decision.py            # Final version creation decision logic
│   ├── audio_utils.py                 # get_audio_duration() and helpers
│   ├── audio_cache.py                 # Cache for expensive audio loads
│   ├── config_loader.py               # Loads voice_config.yaml
│   ├── structured_logger.py           # log_event() for structured JSON logging
│   ├── device_fingerprint.py          # Microphone/device fingerprinting
│   ├── rate_limiter.py                # Rate limiting for version creation
│   ├── age_text_shaper.py             # Text formatting for age display
│   ├── speaker_verification.py        # Cosine similarity speaker gate
│   ├── save_predicted_version.py      # Save a predicted (non-recorded) version
│   ├── synthesize_aged_voice.py       # CLI: generate aged audio (uses old pipeline)
│   ├── synthesize_from_embedding.py   # CLI: generate from embedding (uses old pipeline)
│   ├── synthesize_predicted_voice.py  # CLI: predict future voice (uses old pipeline)
│   ├── test_aged_playback.py          # Manual test script
│   ├── validate_age_features.py       # Validate ECAPA features for age tasks
│   ├── list_versions_by_time.py       # Print all versions for a user
│   ├── migrate_versions_to_user.py    # One-time migration script
│   ├── ingest_fsdd.py                 # Ingest Free Spoken Digit Dataset
│   ├── ingest_librispeech_small.py    # Ingest LibriSpeech for training data
│   ├── ingest_realvoice_small2.py     # Ingest real voice data
│   ├── phase3_*.py                    # Phase 3 training collation (advanced, unused in app)
│   ├── preprocess_manifest.py         # Preprocess audio manifest
│   └── _archive_unused/               # Old prototype scripts, kept for reference
│
├── training/
│   ├── models/
│   │   ├── voice_evolution_model.pt   # *** TRAINED WEIGHTS — VoiceEvolutionModel ***
│   │   ├── age_classifier.pt          # Trained age classifier (not used in app)
│   │   ├── final_age_model.pt         # Alternative age model (not used in app)
│   │   └── hubert_age_transformer.pt  # HuBERT-based age transformer (not used in app)
│   │
│   ├── dataset/
│   │   ├── VCTK-Corpus-0.92.zip       # Source dataset (large, VCTK speakers)
│   │   ├── X_embeddings.pt            # Training embeddings tensor
│   │   ├── y_age.pt                   # Age labels tensor
│   │   ├── combined_X.pt              # Combined ECAPA+HuBERT embeddings (1216-dim)
│   │   ├── combined_y.pt              # Combined age labels
│   │   ├── evolve_source.pt           # Source embeddings for evolution pairs
│   │   ├── evolve_target.pt           # Target embeddings for evolution pairs
│   │   ├── hubert_X.pt                # HuBERT-only embeddings
│   │   ├── hubert_y.pt                # HuBERT age labels
│   │   ├── age_prototypes.pt          # Age group prototype embeddings
│   │   └── age_groups/                # WAV files grouped by age (10_15, 16_20, etc.)
│   │
│   └── scripts/
│       ├── train_voice_evolution_model.py  # *** Defines VoiceEvolutionModel class + training ***
│       ├── train_age_model.py              # Train age classifier
│       ├── train_final_age_model.py        # Train final age model
│       ├── train_hubert_age_transformer.py # Train HuBERT-based model
│       ├── build_age_pairs.py              # Build young/old embedding pairs for training
│       ├── build_age_transformation_pairs.py # Build transformation pairs
│       ├── build_combined_embeddings.py    # Combine ECAPA + HuBERT into 1216-dim tensors
│       ├── build_training_tensor.py        # Build ECAPA training tensors
│       ├── build_hubert_training_tensor.py # Build HuBERT training tensors
│       ├── extract_ecapa_embeddings.py     # Extract ECAPA from dataset WAVs
│       ├── extract_ecapa_embeddings_all.py # Extract ECAPA from all WAVs
│       ├── extract_hubert_embeddings.py    # Extract HuBERT from dataset WAVs
│       ├── compute_age_prototypes.py       # Compute per-age-group centroid embeddings
│       ├── analyze_age_distribution.py     # Dataset analysis
│       ├── analyze_final_dataset.py        # Dataset analysis
│       ├── check_age_distribution.py       # Verify age balance
│       ├── build_dataset_metadata.py       # Build metadata CSV
│       ├── voice_age_inference.py          # Run inference with trained model
│       ├── voice_age_transform.py          # Transform voice embeddings
│       └── visualize_embedding_space.py    # PCA/t-SNE visualization of embeddings
│
├── config/
│   ├── voice_config.yaml              # Part 1 config (thresholds, rate limits, etc.)
│   ├── age_profiles.yaml              # DSP parameters per age group (child/teen/adult/senior/elderly)
│   └── age_playback.yaml              # Part 2 config (decoder model, interpolation, quality)
│
├── users/
│   └── Harshit.json                   # User data file (one file per user)
│
├── versions/
│   ├── audio/                         # Stable copies of recorded WAV files
│   │   └── Harshit_*.wav
│   └── embeddings/                    # ECAPA embeddings per version (.npy files)
│       └── Harshit_*.npy
│
├── models/
│   └── hifigan/
│       ├── config.json                # HiFi-GAN model config
│       └── generator_v3               # HiFi-GAN pretrained weights
│
├── hifi-gan/                          # HiFi-GAN source code (models.py, env.py, etc.)
│
├── data/
│   ├── inputs/                        # Raw audio inputs (temporary)
│   └── outputs/
│       ├── ui_single/                 # Single-age generated WAV files (old pipeline)
│       ├── ui_packs/                  # Sample pack ZIPs (old pipeline)
│       └── age_playback/              # Part 2 generated WAV files
│
├── embeddings/
│   └── ecapa/                         # Temporary ECAPA embeddings before version creation
│
├── logs/
│   └── voice_evolution.log            # Application log
│
├── tests/
│   └── test_age_playback.py           # 28 unit tests for Part 2 (all passing)
│
├── src/api/
│   └── main.py                        # FastAPI REST API (exists but not deployed)
│
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Package build config (setuptools)
├── runtime.txt                        # Streamlit Cloud Python version spec
├── packages.txt                       # Streamlit Cloud system packages (ffmpeg)
└── dockerignore                       # Docker ignore file
```

---

## 4. User Data Format

Every user has one JSON file in `users/{user_id}.json`. This is the central data store for Part 1 and Part 2.

```json
{
  "user_id": "Harshit",
  "date_of_birth": "2003-01-01",
  "created_utc": "2026-02-23T15:12:24Z",
  "voice_versions": [
    {
      "version_id": "1773042076",
      "recorded_utc": "2026-03-09T07:41:16Z",
      "age_at_recording": 23,
      "embedding_path": "versions/embeddings/Harshit_1773042076.npy",
      "audio_path": "/Users/.../versions/audio/Harshit_1773042074.wav",
      "confidence": 1.0,
      "type": "RECORDED"
    }
  ]
}
```

Key fields:
- `date_of_birth` — YYYY-MM-DD, required for age mapping
- `voice_versions` — ordered list of stored voice snapshots
- `embedding_path` — relative to project root, ECAPA 192-dim `.npy` file
- `audio_path` — can be absolute or relative; **some early versions may have temp paths that are now deleted**
- `age_at_recording` — integer age when the recording was made
- `confidence` — 0.0–1.0, quality score

**Managed by:** `scripts/user_registry.py` → `UserRegistry` class

---

## 5. Part 1 — Voice Change Detection (DO NOT MODIFY)

### 5.1 Entry Point

`scripts/process_new_voice.py` → `process_new_voice(user_id, audio_path)` → returns dict

### 5.2 Pipeline

```
User uploads audio file
        ↓
scripts/audio_preprocess.py
  normalize_audio() → 16kHz mono WAV, remove DC bias
        ↓
Copy to stable path: versions/audio/{user_id}_{timestamp}.wav
  (critical for Streamlit Cloud — temp files get deleted)
        ↓
scripts/audio_quality.py
  audio_quality_gate() → checks duration (≥10s) and SNR (≥20dB)
  → soft fail only: records flag, reduces confidence, does not reject
        ↓
scripts/embed_ecapa.py
  extract_embedding() → 192-dim ECAPA vector
  SpeechBrain "spkrec-ecapa-voxceleb" model
        ↓
If NO existing versions:
  → Save embedding to versions/embeddings/{user_id}_{ts}.npy
  → Add version to user JSON with type="RECORDED"
  → Return CREATE_BASELINE
        ↓
scripts/speaker_verification.py
  speaker_verification_gate() → cosine similarity to all stored embeddings
  threshold=0.70 (STRICT_SPEAKER_THRESHOLD)
  → if similarity < 0.70: reject as "Different speaker"
        ↓
scripts/device_fingerprint.py (soft check)
  Microphone fingerprint matching (advisory only)
        ↓
scripts/confidence_engine.py
  compute_confidence() → 0.0–1.0 score from: duration, SNR, similarity, device, history
        ↓
scripts/version_decision.py
  decide_voice_version() → one of:
    REJECT            (similarity < hard threshold, or wrong speaker)
    NO_NEW_VERSION    (voice stable / insufficient confidence)
    CREATE_VERSION    (significant change detected, high confidence)
        ↓
If CREATE_VERSION:
  → Save embedding .npy
  → Add new voice version to user JSON
```

### 5.3 Change Detection Details

- Hard reject similarity threshold: `0.65` (config: `similarity_reject_hard`)
- No-change threshold: `0.85` (config: `similarity_no_change`)
- New version created when: `0.65 ≤ similarity < 0.85` AND confidence ≥ `0.85`
- Minimum days between versions: `30` (config: `min_days_between_versions`)
- FAISS used for batch mode; numpy cosine for real-time pipeline

### 5.4 Key Classes (Part 1)

**`UserRegistry`** (`scripts/user_registry.py`)
- `get_versions()` → list of version dicts
- `get_latest_version()` → most recent version dict
- `add_voice_version(version_id, embedding_path, audio_path, confidence, voice_type)`
- `set_date_of_birth(dob_str)`
- `calculate_age(recording_date)` → integer age

---

## 6. Part 2 — Age-Specific Voice Playback

### 6.1 Architecture Overview

```
User query: "Play my voice at age 10" / "year 2015" / "10 years from now"
        ↓
AgePlaybackService (src/voice_age/age/playback.py)
  parse_and_play(query)  OR  play_at_age(n)  OR  play_at_year(n)  OR  play_in_future(n)
        ↓
VoiceTimeline (src/voice_age/age/timeline.py)
  get_version_at_age(target_age) → TimelineResult
  Result is tagged as one of:
    is_exact        → age matches a stored recording (within ±0.5 yr)
    is_interpolated → age is between two stored recordings
    is_future       → age is beyond the latest recording
    is_extrapolate  → age is before the earliest recording
    is_empty        → no recordings stored at all
        ↓
AgeTransformer (src/voice_age/age/transformer.py)
  For is_interpolated:
    interpolate_between_versions() → slerp between ECAPA embeddings + picks reference audio
  For is_future:
    predict_future_voice() → VoiceEvolutionModel shifts embedding + picks reference audio
  For is_exact / is_extrapolate:
    adjust_nearby_version() → returns reference audio as-is
        ↓
VoiceDecoder (src/voice_age/vocoder/voice_decoder.py)
  decode(reference_audio_path, source_age, target_age) → DecodeResult
  Primary: loads reference audio, passes to DSPAgingEngine
  Optional (if TTS package installed): XTTS-v2 voice cloning
        ↓
DSPAgingEngine (src/voice_age/vocoder/dsp_aging.py)
  age_voice(wav, sr, source_age, target_age) → float32 numpy array
  Steps:
    1. Pitch shift (delta from age_profiles.yaml)
    2. Spectral roll-off (bright for young, muted for old)
    3. Subtle amplitude wobble for breathiness (max ±3%)
    4. Tremor (amplitude modulation at 4–6 Hz for senior/elderly, depth=2%)
    5. Normalize
        ↓
Quality check:
  DSPAgingEngine.quality_check() → checks RMS > 0.0001 and speech energy ratio > 30%
  If fails: log warning (output still returned — DSP is always valid speech)
        ↓
AudioResult returned to frontend:
  .audio          → float32 mono numpy array
  .sr             → 16000
  .method         → "direct" | "extrapolate_dsp" | "future_dsp" | "dsp" | etc.
  .source_info    → human-readable description
  .confidence     → 0.0–1.0
  .versions_used  → list of version IDs
```

### 6.2 VoiceTimeline — Age Resolution Logic

**File:** `src/voice_age/age/timeline.py`

**`TimelineEntry`** dataclass — one per stored version:
- `audio_abs_path` — resolves both absolute and relative paths
- `embedding_abs_path` — resolves both absolute and relative paths

**`TimelineResult`** dataclass — result of a lookup:
- Exactly one flag is True: `is_exact`, `is_interpolated`, `is_extrapolate`, `is_future`, `is_empty`
- `nearest` — closest single version (always populated when not is_empty)
- `lower`, `upper` — bracketing versions (only for is_interpolated)
- `interp_weight` — 0.0 = use lower entirely, 1.0 = use upper entirely

**Resolution logic (`get_version_at_age`):**
1. No versions → `is_empty`
2. One version → `is_future` if target > recorded+0.5, `is_extrapolate` if target < recorded-0.5, else `is_exact`
3. Multiple versions:
   - target < earliest → `is_extrapolate`
   - target > latest → `is_future`
   - within ±0.5 yr of a stored version → `is_exact`
   - between two versions → `is_interpolated`

**Year and future queries:**
- `get_version_at_year(year)` → converts `date(year, 1, 1)` to fractional age via DOB
- `get_version_in_future(n)` → target = `current_age + n`
- `get_version_at_date(date)` → converts calendar date to fractional age via DOB

### 6.3 AgeTransformer — Embedding Space Transformation

**File:** `src/voice_age/age/transformer.py`

**Slerp (spherical linear interpolation)** — for ages between two recordings:
```python
def _slerp(v0, v1, t):
    # t=0 → v0, t=1 → v1
    # Uses great-circle path on unit sphere
    # Falls back to linear if vectors nearly identical (dot > 0.9995)
```
Interpolation weight `t` = (target_age - lower_age) / (upper_age - lower_age)

**VoiceEvolutionModel** — for future ages:
- Loaded from `training/models/voice_evolution_model.pt`
- Architecture: `Linear(1216→1024) → ReLU → Linear(1024→1024) → ReLU → Linear(1024→1216)` with residual: `output = x + delta`
- Input: 1216-dim combined embedding (192 ECAPA + 1024 HuBERT padded with zeros if only ECAPA available)
- Output: 1216-dim shifted embedding representing an older voice
- Scale factor: `min(years_delta / max(years_delta, 1.0), 1.0)` — blends between original and fully shifted
- Final output blended: `(1 - scale) * original + scale * evolved`

**Important:** The VoiceEvolutionModel does NOT take target age as input. It simply predicts a single "ageing step" from the input embedding. For large future gaps, the delta is scaled proportionally.

**`TransformResult`** dataclass:
- `reference_audio_path` — path to WAV file to use as the audio source
- `source_age` — age of that reference recording
- `ecapa_embedding` — transformed 192-dim embedding
- `method` — "slerp" | "future_model" | "dsp_only"

### 6.4 DSPAgingEngine — Signal Processing

**File:** `src/voice_age/vocoder/dsp_aging.py`

**Age profiles** loaded from `config/age_profiles.yaml`:

| Profile | Representative Age | pitch_shift_semitones | breathiness | tremor_rate |
|---|---|---|---|---|
| child | 8 | +4.0 | 0.005 | 0.0 |
| teen | 16 | +2.0 | 0.010 | 0.0 |
| adult | 30 | 0.0 | 0.010 | 0.0 |
| senior | 60 | -1.5 | 0.030 | 4.0 |
| elderly | 80 | -2.5 | 0.060 | 6.0 |

For any target age, parameters are linearly interpolated between the two nearest group profiles.

**Processing steps in `age_voice(wav, sr, source_age, target_age)`:**
1. `pitch_shift_semitones = tgt_profile["pitch_shift_semitones"] - src_profile["pitch_shift_semitones"]` → `librosa.effects.pitch_shift()`
2. Spectral roll-off: pre-emphasis filter for ages < 25 (brighter), lowpass for ages > 55 (muted, cutoff = `max(6000 - (age-55)*100, 3000)` Hz)
3. Breathiness: slow sine wobble at 2.7 Hz and 3.3 Hz, amplitude max ±3% (NOT white noise — that caused the rain sound bug)
4. Tremor: `1 + 0.02 * sin(2π * tremor_rate * t)` amplitude modulation (only for senior/elderly)
5. Normalize output to [-1, 1]

**Known limitation of jitter:** Per-chunk pitch shifts at chunk boundaries create click artifacts (like rain sound). `_add_jitter()` is therefore a no-op pass-through. A proper implementation would require sample-level time-domain PSOLA.

### 6.5 VoiceDecoder — Decoder Interface

**File:** `src/voice_age/vocoder/voice_decoder.py`

Wraps the audio generation with a clean interface:
- `model_name="dsp"` (default, always works, no downloads)
- `model_name="xtts"` (requires `pip install TTS`, downloads XTTS-v2 ~2GB)
- `model_name="auto"` (tries XTTS first, falls back to DSP)

**`decode(reference_audio_path, source_age, target_age)`:**
1. Load reference audio at 16kHz mono
2. If source_age == target_age (within 0.25 yr) → return audio as-is with method="direct"
3. If XTTS available → clone speaker identity → apply DSP aging on clone
4. Otherwise → load reference audio → apply `DSPAgingEngine.age_voice()`
5. Run quality_check → if fails, log warning (still return the audio)

Returns `DecodeResult` with `.audio`, `.sr`, `.method_used`, `.quality_ok`

### 6.6 AgePlaybackService — Master Orchestrator

**File:** `src/voice_age/age/playback.py`

**Public API:**
```python
svc = AgePlaybackService("Harshit", decoder_model="dsp")
result = svc.play_at_age(10.0)        # past voice
result = svc.play_at_age(23.0)        # exact match
result = svc.play_at_age(60.0)        # future voice
result = svc.play_at_year(2015)       # by year
result = svc.play_in_future(10.0)     # 10 years from now
result = svc.parse_and_play("age 25") # natural language
timeline = svc.get_timeline()          # all versions as list of dicts
```

**Rendering strategies:**
- `is_exact` + gap < 2yr → load stored audio, tiny DSP if gap > 0.1yr
- `is_exact` + gap ≥ 2yr → `_render_dsp_from_nearest()`
- `is_interpolated` → `_render_interpolated()` (slerp + decoder)
- `is_future` → `_render_future()` (VoiceEvolutionModel + decoder)
- `is_extrapolate` → `_render_dsp_from_nearest()` (DSP on earliest recording)

**Robustness fix (Bug 1):** `_render_dsp_from_nearest()` and the `is_exact` path both call `_find_any_valid_entry()` when the primary version's audio file is missing. This handles the case where early versions stored temp file paths that have since been deleted. The fallback walks all stored versions and picks the closest one with a valid audio file on disk.

**`AudioResult`** dataclass returned to frontend:
- `.audio` — float32 mono numpy array at 16kHz
- `.sr` — sample rate (16000)
- `.target_age` — requested age
- `.source_age` — age of the reference recording used
- `.method` — "direct" | "extrapolate_dsp" | "future_dsp" | "dsp" | "dsp_fallback" | etc.
- `.source_info` — human-readable description for UI display
- `.confidence` — 0.0–1.0
- `.versions_used` — list of version_id strings
- `.interp_weight` — float if interpolated, else None

---

## 7. Frontend — Streamlit App

**File:** `frontend/app.py`

Single-page app with two major sections:

### 7.1 Part 1 Section — Voice Ingestion

- Create New User form (user_id + DOB)
- User selector dropdown
- User metrics (versions count, DOB, created date)
- File uploader (WAV/MP3, minimum 10 seconds)
- Calls `scripts/process_new_voice.process_new_voice(user_id, audio_path)`
- Displays: change detected, decision, confidence, similarity

### 7.2 Part 2 Section — Age Playback

- Shows DOB and current age
- Timeline visualisation: one column per stored version, showing age label, date, and inline audio player
- Shows available age range (earliest → latest → predicted max)
- Three query modes (radio buttons):
  - "Play at age" → number input → `svc.play_at_age(n)`
  - "Play as of year" → year input → `svc.play_at_year(year)`
  - "Play N years from now" → number input → `svc.play_in_future(n)`
- Generate button → displays audio player + download button
- Generation info panel: target age, source age, confidence, method, source info, versions used, interpolation weight
- Age sweep ZIP: generates WAV at every age in a user-defined range, zips and offers download

**Caching:** `@st.cache_resource` on `_get_service(uid)` so the service is not re-initialized on every Streamlit rerun.

**Note:** `ffmpeg_to_wav_16k_mono()` helper used in Part 1 upload section requires `ffmpeg` system binary (listed in `packages.txt` for Streamlit Cloud).

---

## 8. Configuration Files

### 8.1 `config/voice_config.yaml` (Part 1)

```yaml
audio_quality:
  min_duration_sec: 10.0
  min_snr_db: 20.0

speaker_verification:
  similarity_reject_hard: 0.65
  similarity_no_change: 0.85

confidence:
  reject_below: 0.60
  create_above: 0.85

device:
  min_match_score: 0.60

faiss:
  similarity_threshold: 0.75

rate_limit:
  max_requests: 5
  window_sec: 60
```

### 8.2 `config/age_profiles.yaml` (DSP parameters per age group)

```yaml
child:   { min_age: 5,  max_age: 12, pitch_shift_semitones: 4.0,  speaking_rate: 1.25, breathiness: 0.005, jitter: 0.03, shimmer: 0.01,  tremor_rate: 0.0 }
teen:    { min_age: 13, max_age: 19, pitch_shift_semitones: 2.0,  speaking_rate: 1.1,  breathiness: 0.01,  jitter: 0.02, shimmer: 0.008, tremor_rate: 0.0 }
adult:   { min_age: 20, max_age: 50, pitch_shift_semitones: 0.0,  speaking_rate: 1.0,  breathiness: 0.01,  jitter: 0.01, shimmer: 0.005, tremor_rate: 0.0 }
senior:  { min_age: 51, max_age: 70, pitch_shift_semitones: -1.5, speaking_rate: 0.9,  breathiness: 0.03,  jitter: 0.04, shimmer: 0.02,  tremor_rate: 4.0 }
elderly: { min_age: 71, max_age: 90, pitch_shift_semitones: -2.5, speaking_rate: 0.85, breathiness: 0.06,  jitter: 0.06, shimmer: 0.035, tremor_rate: 6.0 }
```

### 8.3 `config/age_playback.yaml` (Part 2)

```yaml
age_playback:
  decoder_model: "dsp"           # auto | xtts | dsp
  decoder_weights_dir: "pretrained_models/voice_decoder/"
  interpolation_method: "spherical"   # linear | spherical
  min_age_for_dsp_only: 2.0
  quality_check_snr_threshold: 10.0
  fallback_to_dsp: true
  max_future_prediction_years: 40
  output_dir: "data/outputs/age_playback/"
  sample_rate: 16000
```

---

## 9. Key Models

### 9.1 ECAPA-TDNN (SpeechBrain)

- **Model:** `speechbrain/spkrec-ecapa-voxceleb`
- **Output:** 192-dim L2-normalized speaker embedding
- **Used for:** Speaker verification in Part 1; stored per version in `versions/embeddings/`
- **Loaded by:** `scripts/embed_ecapa.py` → `extract_embedding(audio_path)`
- **Cache location:** `pretrained_models/ecapa/`

### 9.2 HuBERT (Facebook)

- **Model:** `facebook/hubert-large-ls960-ft`
- **Output:** 1024-dim feature embedding (mean-pooled over time)
- **Used for:** Combined with ECAPA to create 1216-dim input for VoiceEvolutionModel
- **Note:** HuBERT embeddings are NOT stored on disk — extracted on-the-fly when needed
- **Was used in:** Old `src/voice_age/age/control.py` (now deleted)

### 9.3 VoiceEvolutionModel (Custom, Trained)

- **File:** `training/scripts/train_voice_evolution_model.py` — contains both class definition and training code
- **Weights:** `training/models/voice_evolution_model.pt`
- **Architecture:**
  ```
  Input: 1216-dim embedding (ECAPA 192 + HuBERT 1024)
  Linear(1216 → 1024) → ReLU
  Linear(1024 → 1024) → ReLU
  Linear(1024 → 1216)
  Output = Input + delta   (residual connection)
  ```
- **Training data:** Pairs of (young_speaker_embedding, old_speaker_embedding) from VCTK corpus, stored as `evolve_source.pt` and `evolve_target.pt`
- **Training:** 30 epochs, batch size 128, Adam lr=1e-4, MSE loss
- **What it does:** Given a current voice embedding, predicts what that voice will sound like when older. Does NOT accept target age as input — it predicts a generic "ageing direction"
- **Important limitation:** In Part 2, when we only have ECAPA (192-dim) for stored versions, we pad with zeros to get 1216-dim before feeding VoiceEvolutionModel. The zero-padded HuBERT portion reduces model accuracy but it still produces a reasonable shift direction

### 9.4 HiFi-GAN (Pretrained Vocoder)

- **Config:** `models/hifigan/config.json`
- **Weights:** `models/hifigan/generator_v3`
- **Code:** `hifi-gan/` directory (external repo, added to sys.path)
- **Wrapper:** `src/voice_age/vocoder/hifigan.py` → `HiFiGANVocoder.mel_to_audio(mel)`
- **Status:** Works correctly, but is NOT used in Part 2 (Part 2 does not use mel spectrograms). Only needed by old pipeline (old `control.py` which is deleted).

### 9.5 MelGenerator (BROKEN — DO NOT USE)

- **File:** `src/voice_age/models/mel_generator.py`
- **Problem:** Randomly initialized — never trained. Outputs meaningless noise, not mel spectrograms. Has never been trained and training it would require massive GPU time and speech data.
- **Status:** File still exists but nothing imports it anymore. Safe to delete if desired.

---

## 10. Bugs Fixed During Development

### Bug 1: Past Voice Generation Fails with FileNotFoundError

**Symptom:** Generating voice at age 10 or 19 (when user only has recordings at age 23) threw `FileNotFoundError`.

**Root cause:** Early voice versions (e.g., the first Harshit recording) stored the audio as a temp file path like `/var/folders/.../tmpdm22m9lj.mp3`. macOS deletes temp files. When the timeline picked this version as "nearest" for a past query, it tried to load the deleted file.

**Fix:** Added `_find_any_valid_entry()` method in `AgePlaybackService`. Both `_render_dsp_from_nearest()` and the `is_exact` branch now call this fallback if the primary entry's audio is missing. The fallback scans all stored versions, finds any with a valid audio file, and picks the one closest in age to the requested target.

**Location:** `src/voice_age/age/playback.py` — `_find_any_valid_entry()` and its call sites.

### Bug 2: Generated Voice Has "Rain" / Crackling Noise

**Symptom:** Any future/past generated voice had a persistent background sound like rain falling.

**Root cause 1 — Breathiness noise at 50% RMS:**
The original `_add_breathiness()` injected band-limited white noise (2–8 kHz band) scaled by `amount * 10.0`. With an adult→elderly breathiness delta of `0.06 - 0.01 = 0.05`, the noise was added at `0.5 × RMS` of the speech — audible as a hiss/rain sound.

**Root cause 2 — Jitter creates click artifacts:**
The original `_add_jitter()` ran `librosa.effects.pitch_shift()` on 100ms audio chunks back-to-back. Pitch shift slightly changes the length of each chunk, creating a discontinuity at every boundary — 10 clicks per second, sounding like rain or crackling.

**Fix 1:** Replaced white noise breathiness with a gentle amplitude wobble: two low-frequency sine waves at 2.7 Hz and 3.3 Hz, amplitude clamped to max 3% of signal level. No noise injected at all.

**Fix 2:** Made `_add_jitter()` a no-op pass-through function. The pitch shift step already handles the dominant age-related change. Proper jitter requires sample-level time-domain processing (PSOLA algorithm).

**Location:** `src/voice_age/vocoder/dsp_aging.py` — `_add_breathiness()`, `_add_jitter()`, and the `age_voice()` call site.

### Bug 3: PROJECT_ROOT Path Resolution Error

**Symptom:** `age_profiles.yaml` not found — DSP engine fell back to default values.

**Root cause:** New Part 2 modules used `parents[4]` instead of `parents[3]` to walk up to the project root. Files in `src/voice_age/{subdir}/file.py` are 4 directory levels deep from the file, so `parents[3]` is the project root, not `parents[4]`.

**Fix:** Changed all four new modules to use `_HERE.parents[3]`.

### Bug 4: Single-Version User Returns Wrong Timeline Flag

**Symptom:** A user with one recording at age 23 querying age 10 returned `is_exact=True` instead of `is_extrapolate=True`.

**Root cause:** The original single-version fast path always returned `is_exact=True` regardless of the age gap.

**Fix:** Added proper boundary checks in the single-version branch of `get_version_at_age()`: returns `is_future` if target > recorded + 0.5, `is_extrapolate` if target < recorded - 0.5, and `is_exact` only for true matches.

---

## 11. Tests

**File:** `tests/test_age_playback.py`
**Status:** 28 tests, all passing

| Test class | What it tests |
|---|---|
| `TestAgeQueryParsing` | Regex parsing of "age 25", "year 2015", "10 years from now" |
| `TestVoiceTimeline` | Empty user, single version, five versions, year queries, future queries |
| `TestSlerp` | Endpoint correctness, unit norm preservation, output shape |
| `TestDSPAgingEngine` | Output dtype, length, RMS not silence, quality check, profile ordering |
| `TestVoiceEvolutionModel` | Forward pass shape, real weights load and run |
| `TestFullPipeline` | VoiceDecoder DSP, full AgePlaybackService with mock user + real WAV |
| `TestEdgeCases` | No recordings raises ValueError, pad_to_combined_dim, age fraction math |

---

## 12. Limitations and What Could Be Improved

### 12.1 Voice Quality (Most Important)

**Current state:** DSP only applies pitch shifting, spectral roll-off, subtle amplitude wobble, and tremor. The resulting voice is recognizably the same pitch-shifted version of the original, but does not change formants, vocal tract characteristics, or speaking style. It sounds like "same voice, different pitch" rather than "genuinely younger/older voice."

**Better approaches:**

1. **Install XTTS-v2 (Coqui TTS):** `pip install TTS`. The `VoiceDecoder` already has XTTS support — set `decoder_model="xtts"` in `config/age_playback.yaml`. XTTS clones the speaker identity from the reference recording, which gives a much higher-quality base. Then DSP aging is applied on top of the clone.

2. **Formant shifting (praat-parselmouth):** `praat-parselmouth` is already installed. Use Praat's `change_gender()` or formant manipulation functions to shift F1/F2/F3 formants. Formant frequencies change with age (vocal tract length changes). This would make the voice sound genuinely different in character, not just different in pitch.

3. **RVC (Retrieval-based Voice Conversion):** A properly set up RVC model with a pretrained checkpoint can do excellent voice conversion. The `engines/post.py` already has a placeholder for RVC integration.

4. **FreeVC or OpenVoice:** These voice conversion models take source audio and target speaker embedding → output voice with source content but target speaker characteristics. They could replace the DSP path entirely for much higher quality.

### 12.2 VoiceEvolutionModel Accuracy

**Current state:** The model predicts a single direction of ageing without knowing the target age. For very large age jumps (e.g., 50 years into the future), the single model pass is insufficient.

**Improvement:** Modify `predict_future_voice()` to run the model iteratively (e.g., 5 passes for 50 years, 1 pass per 10 years) instead of scaling a single delta. Or retrain the model to accept target age as a conditioning input.

### 12.3 HuBERT Embedding Storage

**Current state:** Stored version embeddings are ECAPA only (192-dim). When feeding VoiceEvolutionModel (needs 1216-dim), the HuBERT portion is zero-padded. This degrades model accuracy.

**Improvement:** Extract and store HuBERT embeddings (1024-dim) alongside ECAPA embeddings at version creation time in `process_new_voice.py`. Store as `versions/embeddings/{user_id}_{ts}_hubert.npy`. Then `AgeTransformer` can build the full 1216-dim vector from real embeddings.

### 12.4 Short Audio Files

**Current state:** Part 1 requires 10-second minimum recordings. Short files may cause `librosa.effects.pitch_shift` to produce artifacts at the edges.

**Improvement:** Pad short files before pitch shifting, then trim. Apply a short fade-in/fade-out (20ms) to all generated audio to eliminate any edge clicks.

### 12.5 Audio Path Stability

**Current state:** Some early stored versions have temp file paths. The `_find_any_valid_entry()` fallback handles this gracefully. However, it would be better to have clean paths for all versions.

**Improvement:** Add a migration script that finds all versions with missing audio paths and marks them as `"audio_path": null` or re-downloads from any available source.

### 12.6 No Comparison Mode

The frontend currently shows one audio player at a time. A side-by-side comparison between current voice and aged voice would be a good UX addition.

### 12.7 No Age Scrubber

A slider to smoothly scrub through ages (like a timeline) and hear the voice transition in real-time would be a good UX feature. The sweep ZIP generation already provides the underlying data — it just needs a better UI.

---

## 13. How to Extend / Continue This Project

### Adding a New Voice Quality Engine

1. Add a new method to `VoiceDecoder` (e.g., `_decode_freevc()`)
2. Try it in `decode()` before the DSP path
3. Update `model_name` options and config

### Improving DSP with Formant Shifting

```python
import parselmouth
# Load audio as parselmouth Sound
sound = parselmouth.Sound(wav, sampling_frequency=sr)
# Manipulate formants
manipulated = parselmouth.praat.call(sound, "Change gender",
    75, 600, 1.2, 0, 0, 1.0)
# Get numpy array back
wav_out = manipulated.values[0]
```
Add this between pitch shift and spectral roll-off in `DSPAgingEngine.age_voice()`.

### Adding a New User Field

Edit `scripts/user_registry.py` → `UserRegistry.__init__()` to add the field to the default `self.data` dict, and update `add_voice_version()` if the field should be set at version creation time.

### Adding a REST API Endpoint

`src/api/main.py` contains a FastAPI skeleton. Add new endpoints there and run with `uvicorn src.api.main:app`.

### Training VoiceEvolutionModel on Your Own Voice Data

1. Prepare pairs of (young_speaker, old_speaker) audio files
2. Run `training/scripts/extract_ecapa_embeddings_all.py` to get ECAPA embeddings
3. Run `training/scripts/extract_hubert_embeddings.py` to get HuBERT embeddings
4. Run `training/scripts/build_combined_embeddings.py` to create 1216-dim tensors
5. Run `training/scripts/build_age_transformation_pairs.py` to create source/target pairs
6. Run `training/scripts/train_voice_evolution_model.py` — saves to `training/models/voice_evolution_model.pt`

---

## 14. Important Code Locations Quick Reference

| What you need | Where to find it |
|---|---|
| Add a new Part 1 feature | `scripts/process_new_voice.py` |
| Change version creation logic | `scripts/version_decision.py` |
| Change quality thresholds | `config/voice_config.yaml` |
| Change DSP aging parameters | `config/age_profiles.yaml` |
| Change age playback config | `config/age_playback.yaml` |
| Age resolution (which version for which age) | `src/voice_age/age/timeline.py` |
| Embedding interpolation (slerp) | `src/voice_age/age/transformer.py` |
| Future voice prediction (VoiceEvolutionModel) | `src/voice_age/age/transformer.py` — `predict_future_voice()` |
| DSP pitch/spectral processing | `src/voice_age/vocoder/dsp_aging.py` |
| Add neural voice decoder | `src/voice_age/vocoder/voice_decoder.py` |
| Master playback pipeline | `src/voice_age/age/playback.py` |
| Streamlit UI | `frontend/app.py` |
| User data storage | `scripts/user_registry.py` |
| VoiceEvolutionModel class definition | `training/scripts/train_voice_evolution_model.py` |
| VoiceEvolutionModel weights | `training/models/voice_evolution_model.pt` |
| Run tests | `python -m pytest tests/test_age_playback.py -v` |

---

## 15. Environment Setup (Fresh Machine)

```bash
# Clone project
git clone <repo_url> voice-evolution-system
cd voice-evolution-system

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the voice_age package in editable mode
pip install -e .

# Run the app
streamlit run frontend/app.py
```

System dependency: `ffmpeg` must be installed (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Linux).

For neural voice cloning (optional, 2GB download):
```bash
pip install TTS
# Then in config/age_playback.yaml: decoder_model: "auto"
```

---

*Document generated: March 2026. Covers all work done through the end of the session where Part 2 was built, bugs fixed, and tests written.*
