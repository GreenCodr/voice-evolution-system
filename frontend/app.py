# frontend/app.py

import sys
import os
import json
import tempfile
from pathlib import Path

import streamlit as st

# ------------------ FIX PYTHON PATH (CRITICAL) ------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------ BACKEND IMPORTS ------------------
from scripts.process_new_voice import process_new_voice
from scripts.playback_service import play_voice
from scripts.playback_explainer import explain_playback

USERS_DIR = PROJECT_ROOT / "users"

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Voice Evolution System",
    layout="centered"
)

# ==============================================================
# HEADER
# ==============================================================

st.title("🎙️ Voice Evolution System")
st.caption("Automatic voice change detection & age-based playback")
st.divider()

# ==============================================================
# USER SELECTION
# ==============================================================

st.header("👤 User Dashboard")

user_files = list(USERS_DIR.glob("*.json"))
if not user_files:
    st.error("No users found. Please create a user first.")
    st.stop()

user_ids = [f.stem for f in user_files]
selected_user = st.selectbox("Select User", user_ids)

user_path = USERS_DIR / f"{selected_user}.json"
user = json.loads(user_path.read_text())

col1, col2 = st.columns(2)
with col1:
    st.metric("User ID", user["user_id"])
    st.metric("Date of Birth", user.get("date_of_birth", "Unknown"))
with col2:
    st.metric("Total Voice Versions", len(user.get("voice_versions", [])))
    st.metric("Account Created", user.get("created_utc", "")[:10])

st.divider()

# ==============================================================
# PHASE 1 — VOICE INGESTION
# ==============================================================

st.header("🎙️ Upload Voice Sample")

uploaded = st.file_uploader(
    "Upload voice sample (WAV / MP3, minimum 10 seconds)",
    type=["wav", "mp3"]
)

if uploaded:
    st.audio(uploaded)
    st.success("Voice file received ✔️")

st.divider()

st.header("🔍 Voice Analysis Result")

if uploaded:
    suffix = Path(uploaded.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing voice sample..."):
        result = process_new_voice(
            user_id=selected_user,
            audio_path=tmp_path
        )

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    if not result.get("accepted", False):
        st.error(f"❌ {result.get('reason', 'Rejected')}")
    else:
        decision = result.get("decision", {})

        st.success("Voice analyzed successfully")
        st.write(f"• Change detected: `{result.get('change_detected')}`")
        st.write(f"• Decision: `{decision.get('action')}`")

        if "reason" in decision:
            st.write(f"• Reason: {decision['reason']}")

        st.metric("Confidence", result.get("confidence", 0.0))
        st.metric("Similarity", result.get("similarity", 0.0))

        if result.get("audio_quality_soft_fail"):
            st.warning("⚠️ Audio quality was suboptimal (soft penalty applied)")
else:
    st.info("Waiting for voice input...")

st.divider()

# ==============================================================
# VOICE TIMELINE
# ==============================================================

st.header("🧬 Voice Timeline")

# Reload user after ingestion
user = json.loads(user_path.read_text())
versions = user.get("voice_versions", [])

if versions:
    table = []
    for v in versions:
        table.append({
            "Recorded Date": v.get("recorded_utc", "")[:10],
            "Age": v.get("age_at_recording"),
            "Type": v.get("type", "RECORDED"),
            "Confidence": v.get("confidence", "-"),
        })
    st.dataframe(table, use_container_width=True)
else:
    st.warning("No voice history available")

st.divider()

# ==============================================================
# PHASE 2 — AGE-BASED PLAYBACK
# ==============================================================

st.header("🎧 Age-Based Voice Playback")

target_age = st.slider(
    "Select target age",
    min_value=5,
    max_value=90,
    value=60
)

text_to_speak = st.text_area(
    "Text to speak (longer text = longer voice)",
    value=(
        "Hello, this is how my voice may sound in the future. "
        "This system demonstrates realistic voice evolution over time, "
        "preserving my identity while reflecting natural age-related changes. "
        "The voice you are hearing is generated using a real-world "
        "voice evolution pipeline."
    ),
    height=220
)

if st.button("▶️ Play Voice"):
    with st.spinner("Preparing voice playback..."):
        result = play_voice(
            user_id=selected_user,
            target_age=target_age,
            text=text_to_speak
        )

    if result["mode"] == "ERROR":
        st.error(result["reason"])
    else:
        explanation = explain_playback(result)
        st.markdown(f"### {explanation['icon']} {explanation['label']}")
        st.info(explanation["message"])
        st.audio(result["audio_path"])

st.divider()
st.caption("Voice Evolution System — Phase 2 complete")