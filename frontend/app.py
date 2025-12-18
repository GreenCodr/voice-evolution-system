# frontend/app.py

import json
import streamlit as st
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USERS_DIR = PROJECT_ROOT / "users"

# ------------------ PAGE CONFIG ------------------

st.set_page_config(
    page_title="Voice Evolution System",
    layout="centered"
)

st.title("🎙️ Voice Evolution System")
st.subheader("Real-world voice timeline & playback")

st.markdown("---")

# ------------------ USER SELECTION ------------------

user_files = list(USERS_DIR.glob("*.json"))

if not user_files:
    st.error("No users found. Please create a user first.")
    st.stop()

user_ids = [f.stem for f in user_files]
selected_user = st.selectbox("Select User", user_ids)

user_path = USERS_DIR / f"{selected_user}.json"

with open(user_path, "r", encoding="utf-8") as f:
    user = json.load(f)

st.success(f"Active user: **{selected_user}**")

st.markdown("---")

# ------------------ USER PROFILE ------------------

st.header("👤 User Profile")

col1, col2 = st.columns(2)

with col1:
    st.write("**User ID**")
    st.write(user["user_id"])

    st.write("**Date of Birth**")
    st.write(user.get("date_of_birth", "Unknown"))

with col2:
    st.write("**Created On**")
    created = user.get("created_utc")
    if created:
        st.write(datetime.fromisoformat(created.replace("Z", "")).strftime("%Y-%m-%d"))
    else:
        st.write("Unknown")

    st.write("**Total Voice Versions**")
    st.write(len(user.get("voice_versions", [])))

# ------------------ TIMELINE SUMMARY ------------------

versions = user.get("voice_versions", [])

if versions:
    ages = [v["age_at_recording"] for v in versions if v.get("age_at_recording") is not None]

    if ages:
        st.markdown("---")
        st.header("📈 Voice Timeline Summary")
        st.write(f"**Earliest age recorded:** {min(ages)}")
        st.write(f"**Latest age recorded:** {max(ages)}")
else:
    st.warning("No voice versions available for this user.")

st.markdown("---")
st.info("Profile loaded successfully ✔️")

# ------------------ VOICE TIMELINE TABLE ------------------

st.header("🧬 Voice Timeline")

if not versions:
    st.warning("No voice history available.")
else:
    versions = sorted(versions, key=lambda v: v.get("recorded_utc", ""))

    table_data = []
    for v in versions:
        table_data.append({
            "Recorded Time": v.get("recorded_utc", "—"),
            "Age": v.get("age_at_recording", "—"),
            "Type": v.get("type", "RECORDED"),
            "Confidence": v.get("confidence", "—"),
            "Has Audio": "✔️" if v.get("audio_path") else "—"
        })

    st.dataframe(table_data, use_container_width=True)

st.markdown("---")
st.info("Timeline loaded ✔️")

# ------------------ VOICE PLAYBACK ------------------

st.subheader("🎧 Voice Playback")

target_age = st.slider(
    "Select target age",
    min_value=0,
    max_value=100,
    value=26,
    step=1
)

text = st.text_input(
    "Text to speak",
    "Hello, this is my voice at this age."
)

if st.button("▶️ Play Voice"):
    with st.spinner("Preparing voice..."):
        from scripts.hybrid_playback_decider import decide_playback_mode
        from scripts.playback_explainer import explain_playback

        decision = decide_playback_mode(selected_user, target_age)
        explanation = explain_playback(decision)

        st.markdown(f"### {explanation['icon']} {explanation['label']}")
        st.info(explanation["message"])

        tmp_out = Path(tempfile.gettempdir()) / "voice_output.wav"

        try:
            if decision["mode"] == "RECORDED":
                audio_path = decision["version"].get("audio_path")

                if not audio_path:
                    st.error("No recorded audio available for this version.")
                    st.stop()

                subprocess.run(
                    [
                        "python",
                        "scripts/synthesize_from_embedding.py",
                        text,
                        str(tmp_out),
                        "--speaker_wav",
                        audio_path
                    ],
                    check=True
                )

            else:
                nearest = decision.get("nearest", {})
                audio_path = nearest.get("audio_path")

                if not audio_path:
                    st.error("No reference audio available for prediction.")
                    st.stop()

                subprocess.run(
                    [
                        "python",
                        "scripts/synthesize_predicted_voice.py",
                        text,
                        str(tmp_out),
                        "--speaker_wav",
                        audio_path
                    ],
                    check=True
                )

            st.audio(str(tmp_out))

        except subprocess.CalledProcessError:
            st.error("Voice synthesis failed. Please try again.")

# ------------------ FOOTER ------------------

st.markdown("---")
st.caption("⚠️ Predicted voices are AI-generated and may not reflect real recordings.")