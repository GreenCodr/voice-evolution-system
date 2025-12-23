🎙️ Voice Evolution System

Automatic Voice Change Detection, Age-Based Playback & Future Voice Prediction

A real-world AI system that continuously tracks how a person’s voice evolves over time, automatically detects meaningful vocal changes, and enables realistic playback of a voice at any age — past, present, or future.

⸻

🚀 Problem This Project Solves

Human voices change naturally due to:
	•	Age
	•	Health
	•	Emotion
	•	Environment
	•	Recording devices

But today, no system preserves voice evolution intelligently.

This project solves that by:
	•	Automatically detecting significant voice changes
	•	Creating voice versions over time
	•	Allowing playback like:
	•	“Play my voice at age 8”
	•	“How will my voice sound at 60?”
	•	“Play my voice from 2015”

⸻

🧠 Core Features

✅ Phase 1 — Automatic Voice Change Detection
	•	Audio quality gating (duration, SNR)
	•	Speaker verification (ECAPA / wav2vec embeddings)
	•	Device fingerprint matching
	•	Confidence scoring
	•	FAISS similarity search
	•	Automatic version creation
	•	Version history storage

⸻

✅ Phase 2 — Age-Specific Voice Playback
	•	Voice timeline per user
	•	Age mapping using Date of Birth
	•	Closest-age voice selection
	•	SLERP interpolation between versions
	•	Past & future extrapolation
	•	Clear labeling:
	•	✅ RECORDED
	•	🟡 INTERPOLATED
	•	⚠️ PREDICTED
	•	XTTS-based voice synthesis
	•	Rate limiting & audio caching
	•	Metadata tagging

⸻

✅ Phase 3 — Lightweight Learning (Optional)
	•	Builds an age-embedding dataset
	•	Tries learning age-to-voice deltas
	•	Uses small auxiliary models
	•	No heavy GPU training required
	•	Falls back safely to rule-based logic when data is insufficient

⚠️ The system is intentionally robust even without training data.
🧩 System Architecture (High Level)
Audio Input
   ↓
Quality Gate → Speaker Verification → Device Check
   ↓
Confidence Engine
   ↓
FAISS Similarity Search
   ↓
Version Decision Engine
   ↓
User Voice Timeline
   ↓
Playback Engine (Recorded / Interpolated / Predicted)
🖥️ Frontend (Streamlit)

The project includes a Streamlit web app that allows:
	•	User selection
	•	Voice timeline visualization
	•	Age-based voice playback
	•	Clear explanations of playback decisions
	•	Real-time synthesis output
  streamlit run frontend/app.py
  📁 Project Structure (Simplified)
  voice-evolution/
├── frontend/          # Streamlit UI
├── scripts/           # Core system logic
├── config/            # Central config & thresholds
├── users/             # User profiles (runtime)
├── versions/          # Voice versions (runtime)
├── learning/          # Optional lightweight learning
├── src/               # API / core modules
├── README.md
└── .gitignore
🧪 Real-World Design Principles
	•	✔️ Explainable decisions
	•	✔️ Safe fallbacks
	•	✔️ No hallucinated audio
	•	✔️ Confidence-aware outputs
	•	✔️ Production-ready architecture
	•	✔️ Minimal hardware requirements
   Use Cases
	•	Personal voice archiving
	•	Voice aging research
	•	Speech therapy tracking
	•	Digital legacy preservation
	•	Forensic & historical voice analysis
	•	AI assistants with temporal voice memory

	🚀 How to Run the Voice Evolution System Locally
	
1️⃣ Prerequisites

Make sure the following are installed on your system:
	•	Git
	•	Anaconda / Miniconda
	•	Python 3.9 or 3.10 (via Conda – recommended)
	•	FFmpeg (required for audio processing)
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg

2️⃣ Clone the Repository
git clone https://github.com/GreenCodr/voice-evolution-system.git

cd voice-evolution-system

3️⃣ Create & Activate Conda Environment

step1 - conda create -n voice-evo python=3.10 -y

step 2-conda activate voice-evo

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Project Structure Overview (Important)
voice-evolution-system/
│
├── frontend/            # Streamlit UI
├── scripts/             # Core pipelines (age, DSP, playback, detection)
├── users/               # User metadata (JSON)
├── config/              # Age profiles & configs
├── outputs/             # Generated audio (gitignored)
├── cache/               # Audio cache (gitignored)
├── models/              # Trained models (gitignored)
└── README.md

6️⃣ Run the Frontend (Recommended)
streamlit run frontend/app.py

7️⃣ Run Test age-based playback directly
Backend Only (Optional)
python - << 'EOF'
from scripts.playback_service import play_voice

result = play_voice(
    user_id="user_002",
    target_age=60,
    text="Hello, this is how my voice may sound in the future."
)

print(result)
EOF

Generated audio will appear in:
outputs/

8️⃣ Creating a New User (If Needed)
users/user_001.json
users/user_002.json

9️⃣ Important Notes
conda activate voice-evo


	
  
