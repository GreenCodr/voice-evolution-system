---
title: Voice Evolution System
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.30.0
app_file: frontend/app.py
python_version: "3.10"
pinned: false
---

# 🎙️ Voice Evolution System

> **Track how your voice changes over time — and hear what it sounded like at any age, past or future.**

The Voice Evolution System is an AI-powered application that:
- **Records & detects** meaningful changes in your voice over time (puberty, aging, illness, etc.)
- **Stores versioned snapshots** of your voice with timestamps and age tags
- **Generates realistic audio** of your voice at any age — age 10, age 60, or 20 years from now

Built with Python, PyTorch, SpeechBrain, and Streamlit.

---

## 📸 Demo

| Feature | Description |
|---|---|
| Voice Ingestion | Upload a WAV/MP3 recording → system detects if your voice has changed |
| Voice Timeline | Visual timeline of all your stored voice versions |
| Age Playback | Type "Play my voice at age 10" and hear the result |
| Future Prediction | "How will I sound in 20 years?" — uses a trained neural model |
| Age Sweep | Generate a ZIP of your voice at every age from X to Y |

---

## 🗂️ Project Structure

```
voice-evolution-system/
├── frontend/
│   └── app.py                  # Streamlit web UI (run this to start)
│
├── src/voice_age/
│   ├── age/
│   │   ├── playback.py         # Master orchestrator for age-based playback
│   │   ├── timeline.py         # Maps ages to stored voice versions
│   │   └── transformer.py      # Slerp interpolation + future model
│   ├── vocoder/
│   │   ├── dsp_aging.py        # DSP pitch/spectral aging engine
│   │   └── voice_decoder.py    # Decoder interface (DSP or XTTS)
│   └── io/audio.py             # Audio load/write helpers
│
├── scripts/
│   ├── process_new_voice.py    # Part 1: Voice ingestion pipeline
│   ├── user_registry.py        # User data management
│   ├── embed_ecapa.py          # ECAPA speaker embedding extraction
│   └── ...                     # Other processing scripts
│
├── training/
│   ├── models/
│   │   └── voice_evolution_model.pt   # Trained neural model weights
│   └── scripts/
│       └── train_voice_evolution_model.py  # Model definition + training
│
├── config/
│   ├── voice_config.yaml       # Part 1 thresholds (SNR, similarity, etc.)
│   ├── age_profiles.yaml       # DSP parameters per age group
│   └── age_playback.yaml       # Part 2 playback config
│
├── tests/
│   └── test_age_playback.py    # 28 unit tests (all passing)
│
├── requirements.txt
└── pyproject.toml
```

---

## ⚙️ Prerequisites

Make sure the following are installed on your machine before you begin.

### System Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | **3.10** | Exactly 3.10 — SpeechBrain and PyTorch 2.1.2 require it |
| FFmpeg | Any recent | Required for audio conversion |
| Git | Any | For cloning the repo |

### Installing FFmpeg

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg -y

# Windows (via Chocolatey)
choco install ffmpeg

# Windows (via winget)
winget install ffmpeg
```

---

## 🚀 Local Setup — Step by Step

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/voice-evolution-system.git
cd voice-evolution-system
```

> Replace `YOUR_USERNAME` with your GitHub username.

---

### 2. Create a Python 3.10 Virtual Environment

**Option A — Using `venv` (built-in, recommended)**

```bash
# Create the virtual environment
python3.10 -m venv .venv

# Activate it
# macOS / Linux:
source .venv/bin/activate

# Windows (Command Prompt):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

**Option B — Using Conda**

```bash
conda create -n voice-evo python=3.10 -y
conda activate voice-evo
```

---

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs: PyTorch 2.1.2, SpeechBrain, Transformers (HuBERT), librosa, Streamlit, scipy, soundfile, and all other required packages.

> ⚠️ **Note on PyTorch:** `requirements.txt` installs the CPU version of PyTorch by default. If you have a CUDA GPU and want faster inference, replace the torch lines with the appropriate CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

---

### 4. Install the Package in Editable Mode

```bash
pip install -e .
```

This makes the `voice_age` package importable from anywhere inside the project.

---

### 5. (First Run) Download the ECAPA Speaker Model

The SpeechBrain ECAPA-TDNN model (~80 MB) is automatically downloaded from HuggingFace the first time you run the app. No manual step needed — just make sure you have an internet connection on first launch.

The model is cached at `pretrained_models/ecapa/`.

---

### 6. Run the App

```bash
streamlit run frontend/app.py
```

Open your browser at **http://localhost:8501** — the Streamlit UI will load automatically.

---

## 🎯 How to Use the App

### Step 1 — Create Your Profile

In the **"Voice Ingestion"** section:
1. Enter a **User ID** (e.g., your name)
2. Enter your **Date of Birth** (used to calculate your age at each recording)
3. Click **"Create User"**

### Step 2 — Record Your Voice

Record yourself speaking for **at least 10 seconds** (longer is better). Save it as a WAV or MP3 file.

> **Tip:** Read a paragraph of text aloud. Anything works — a news article, a poem, or just counting. The system cares about your voice characteristics, not what you say.

### Step 3 — Upload Your Recording

In the **"Upload Voice Recording"** section:
1. Select your user from the dropdown
2. Upload your WAV or MP3 file
3. The system will analyze it and create a voice version

### Step 4 — Hear Your Voice at Any Age

In the **"Age Playback"** section:
1. Choose a query mode:
   - **"Play at age"** — enter any age number
   - **"Play as of year"** — enter a year (e.g., 2010)
   - **"Play N years from now"** — enter how many years ahead
2. Click **"Generate"**
3. Listen to the result and download if you like

---

## 🧪 Running Tests

```bash
python -m pytest tests/test_age_playback.py -v
```

All 28 tests should pass. Tests cover: age query parsing, voice timeline logic, slerp interpolation, DSP aging engine, the VoiceEvolutionModel, and the full playback pipeline.

---

## 🔧 Configuration

All tunable parameters live in the `config/` directory — no code changes needed.

### `config/voice_config.yaml` — Voice Change Detection

```yaml
audio_quality:
  min_duration_sec: 10.0    # Minimum recording length
  min_snr_db: 20.0          # Minimum signal-to-noise ratio

speaker_verification:
  similarity_reject_hard: 0.65   # Below this → rejected as wrong speaker
  similarity_no_change: 0.85     # Above this → voice hasn't changed enough

confidence:
  create_above: 0.85        # Confidence needed to create a new version
```

### `config/age_profiles.yaml` — DSP Age Parameters

Controls how pitch, breathiness, and tremor change across age groups (child → elderly).

### `config/age_playback.yaml` — Playback Settings

```yaml
age_playback:
  decoder_model: "dsp"      # "dsp" (default, no download) or "auto" (tries XTTS first)
  max_future_prediction_years: 40
```

---

## 🤖 How the AI Works

### Part 1 — Voice Change Detection

```
Upload audio
    ↓
Normalize to 16kHz mono WAV
    ↓
Quality check (duration + SNR)
    ↓
ECAPA-TDNN → 192-dim speaker embedding
    ↓
Compare to stored embeddings (cosine similarity)
    ↓
Confidence score (quality + similarity + device)
    ↓
Decision: BASELINE / NO_CHANGE / NEW_VERSION
```

### Part 2 — Age-Based Voice Generation

```
User query: "Play my voice at age 10"
    ↓
VoiceTimeline → finds nearest stored version(s)
    ↓
AgeTransformer:
  • Past/between recordings → SLERP interpolation of embeddings
  • Future → VoiceEvolutionModel (neural network) shifts embedding
    ↓
DSPAgingEngine:
  • Pitch shift (librosa)
  • Spectral roll-off (bright for young, muted for elderly)
  • Breathiness wobble
  • Tremor (for senior/elderly)
    ↓
Output: WAV file + audio player in UI
```

### The VoiceEvolutionModel

A custom-trained neural network (`training/models/voice_evolution_model.pt`) that predicts how a voice embedding shifts as a person ages. Trained on VCTK corpus speaker pairs (young/old), with architecture:

```
Input: 1216-dim embedding (ECAPA 192 + HuBERT 1024)
Linear(1216 → 1024) → ReLU
Linear(1024 → 1024) → ReLU
Linear(1024 → 1216)
Output = Input + predicted_delta   (residual connection)
```

---

## 🔊 Optional: Better Voice Quality with XTTS

By default the app uses a DSP-only engine (no download, runs anywhere). For higher-quality voice cloning, install Coqui TTS:

```bash
pip install TTS
```

Then in `config/age_playback.yaml`, change:

```yaml
age_playback:
  decoder_model: "auto"   # Will use XTTS-v2 if installed, fallback to DSP
```

> **Note:** XTTS-v2 downloads ~2 GB of model weights on first use. It clones your speaker identity before applying age transformations, producing more realistic results.

---

## ❓ Troubleshooting

**`ModuleNotFoundError: No module named 'voice_age'`**
→ Run `pip install -e .` from the project root.

**`FileNotFoundError` on first launch**
→ The ECAPA model is downloading. Wait a minute and try again.

**`ffmpeg not found` error**
→ Install FFmpeg (see Prerequisites above).

**App is slow on first query**
→ SpeechBrain loads the ECAPA model into memory on first use. Subsequent queries are fast.

**Generated voice sounds unchanged**
→ Make sure the target age is more than 2 years different from your recording age. The DSP aging effect is subtle for small age gaps by design.

---

## 📋 Requirements Summary

```
Python == 3.10
torch == 2.1.2
torchaudio == 2.1.2
speechbrain == 0.5.16
transformers == 4.36.2
librosa
soundfile
scipy
numpy < 2
streamlit == 1.30.0
praat-parselmouth
ffmpeg (system binary)
```

---

## 📄 License

This project is for personal and research use. The VCTK dataset used for training is subject to its own license — see [VCTK corpus](https://datashare.ed.ac.uk/handle/10283/3443).

---

*Built by Harshit Yadav — March 2026*
