# frontend/app.py

import sys
import os
import json
import tempfile
import time
import io
import zipfile
import subprocess
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

# ------------------ PATH FIX ------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ✅ ADD THIS: allow imports from src/ (needed for voice_age on Streamlit Cloud)
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

USERS_DIR = PROJECT_ROOT / "users"


# ------------------ HELPERS ------------------

def ffmpeg_to_wav_16k_mono(in_path: Path, out_path: Path, sr: int = 16000):
    """
    Convert any audio to SR Hz mono WAV using ffmpeg.
    This makes Phase-2 accept mp3/m4a/etc reliably.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        err = (p.stderr or "")[-2000:]
        raise RuntimeError(err if err else "ffmpeg failed")


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def make_zip_bytes(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.glob("*.wav")):
            z.write(p, arcname=p.name)
    buf.seek(0)
    return buf.read()


def _safe_user_id(raw: str) -> str:
    raw = (raw or "").strip()
    raw = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw)
    raw = raw.strip("_")
    return raw


def _validate_dob(dob: str) -> str:
    dob = (dob or "").strip()
    # expected YYYY-MM-DD
    datetime.strptime(dob, "%Y-%m-%d")
    return dob


def _create_user_file(user_id: str, dob: str) -> Path:
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    user_path = USERS_DIR / f"{user_id}.json"
    if user_path.exists():
        raise ValueError("User already exists")

    dob = _validate_dob(dob)

    user_obj = {
        "user_id": user_id,
        "date_of_birth": dob,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "voice_versions": []
    }
    user_path.write_text(json.dumps(user_obj, indent=2))
    return user_path


# ------------------ APP ------------------

def run_app():
    st.set_page_config(page_title="Voice Evolution System", layout="centered")

    # ==============================================================
    # HEADER
    # ==============================================================
    st.title("🎙️ Voice Evolution System")
    st.caption("Automatic voice change detection & age-based playback")
    st.divider()

    # ==============================================================
    # CREATE NEW USER
    # ==============================================================
    st.header("➕ Create New User")

    with st.form("create_user_form", clear_on_submit=False):
        new_user_id_raw = st.text_input("Choose a User ID (letters/numbers/_/-)")
        new_dob = st.text_input("Date of Birth (YYYY-MM-DD)")
        create_btn = st.form_submit_button("Create User")

    if create_btn:
        new_user_id = _safe_user_id(new_user_id_raw)
        if not new_user_id:
            st.error("User ID cannot be empty.")
        else:
            try:
                p = _create_user_file(new_user_id, new_dob)
                st.success(f"✅ Created user: {new_user_id}")
                st.caption(f"Saved at: {p}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

    st.divider()

    # ==============================================================
    # USER SELECTION
    # ==============================================================
    st.header("👤 User Dashboard")

    USERS_DIR.mkdir(parents=True, exist_ok=True)
    user_files = sorted(USERS_DIR.glob("*.json"))
    if not user_files:
        st.error("No users found. Create a user above.")
        st.stop()

    user_ids = [f.stem for f in user_files]
    selected_user = st.selectbox("Select User", user_ids)

    user_path = USERS_DIR / f"{selected_user}.json"
    user = json.loads(user_path.read_text())

    col1, col2 = st.columns(2)
    with col1:
        st.metric("User ID", user.get("user_id", selected_user))
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
        type=["wav", "mp3"],
        key="phase1_upload",
    )

    if uploaded:
        st.audio(uploaded)
        st.success("Voice file received ✔️")

        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Analyzing voice sample..."):
            from scripts.process_new_voice import process_new_voice

            result = process_new_voice(
                user_id=selected_user,
                audio_path=tmp_path,
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
    # PART 2 — AGE-SPECIFIC VOICE PLAYBACK (from stored versions)
    # ==============================================================
    st.header("🎧 Age-Specific Voice Playback")
    st.caption(
        "Plays your stored voice recordings at any age — past, present, or future. "
        "Requires at least one voice version recorded via Part 1 above."
    )

    import numpy as np
    import soundfile as sf
    import librosa
    from voice_age.config import DEFAULT_SR

    # Lazy import so Part 1 still works if Part 2 has an issue
    try:
        from voice_age.age.playback import AgePlaybackService
        _playback_available = True
    except Exception as _p2e:
        _playback_available = False
        st.error(f"Age Playback module could not be loaded: {_p2e}")

    if _playback_available:

        # ── Service (one per user+version-count, so cache refreshes after upload) ──
        _n_versions = len(user.get("voice_versions", []))

        @st.cache_resource(show_spinner=False)
        def _get_service(uid: str, _n: int) -> "AgePlaybackService":
            return AgePlaybackService(uid, decoder_model="dsp")

        svc = _get_service(selected_user, _n_versions)
        timeline_entries = svc.get_timeline()
        earliest_age, latest_age, predicted_max = svc.get_available_range()

        # ── DOB / current age display ───────────────────────────────────────
        dob_str = user.get("date_of_birth")
        if dob_str:
            ca = svc.current_age
            st.success(
                f"Date of Birth: **{dob_str}**  ·  "
                f"Current estimated age: **{ca:.0f}**" if ca is not None
                else f"Date of Birth: **{dob_str}**"
            )
        else:
            st.warning(
                "No date of birth set for this user — "
                "set one when creating the user to enable age mapping."
            )

        # ── Timeline visualisation ──────────────────────────────────────────
        if timeline_entries:
            st.subheader("Voice Version Timeline")
            dated = [e for e in timeline_entries if e["age_at_recording"] is not None]
            if dated:
                num_cols = min(len(dated), 6)
                cols = st.columns(num_cols)
                for i, entry in enumerate(dated[:num_cols]):
                    with cols[i]:
                        st.caption(
                            f"**Age {entry['age_at_recording']:.0f}**\n"
                            f"{entry['recorded_utc'][:10]}\n"
                            f"conf: {entry['confidence']:.2f}"
                        )
                        if entry["audio_exists"]:
                            st.audio(Path(entry["audio_path"]).read_bytes(), format="audio/wav")
                        else:
                            st.caption("_(audio file missing)_")
            else:
                st.info("No recordings with age information found yet.")
        else:
            st.info(
                "No voice versions stored yet. "
                "Upload a recording in the section above first."
            )

        # ── Available range ─────────────────────────────────────────────────
        if earliest_age is not None:
            st.info(
                f"Stored recordings span **age {earliest_age:.0f}** to **age {latest_age:.0f}**. "
                f"Future predictions available up to **age {predicted_max:.0f}**."
            )

        st.divider()

        # ── Query section ───────────────────────────────────────────────────
        st.subheader("Generate Voice at a Specific Age")

        query_mode = st.radio(
            "Query type",
            ["Play at age", "Play as of year", "Play N years from now"],
            horizontal=True,
            key="p2_query_mode",
        )

        p2_age = p2_year = p2_future_years = None
        if query_mode == "Play at age":
            p2_age = st.number_input(
                "Target age (years)", min_value=1, max_value=100, value=25, step=1,
                key="p2_age_input",
            )
        elif query_mode == "Play as of year":
            p2_year = st.number_input(
                "Year", min_value=1950, max_value=datetime.utcnow().year + 50,
                value=datetime.utcnow().year - 5, step=1,
                key="p2_year_input",
            )
        else:
            p2_future_years = st.number_input(
                "Years from now", min_value=1, max_value=40, value=10, step=1,
                key="p2_future_input",
            )

        btn_generate = st.button("Generate Voice", key="p2_generate_btn")

        if btn_generate:
            if not timeline_entries:
                st.error("No voice versions stored. Upload a recording in Part 1 first.")
            else:
                with st.spinner("Generating age-adjusted voice…"):
                    try:
                        if query_mode == "Play at age":
                            result = svc.play_at_age(float(p2_age))
                        elif query_mode == "Play as of year":
                            result = svc.play_at_year(int(p2_year))
                        else:
                            result = svc.play_in_future(float(p2_future_years))

                        # Normalize output
                        audio_out = result.audio.astype("float32")
                        mx = float(np.abs(audio_out).max() + 1e-9)
                        if mx > 1.0:
                            audio_out = audio_out / mx * 0.9

                        # Save to disk
                        outdir = PROJECT_ROOT / "data" / "outputs" / "age_playback"
                        outdir.mkdir(parents=True, exist_ok=True)
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        out_path = (
                            outdir
                            / f"{selected_user}_age_{result.target_age:.0f}_{ts}.wav"
                        )
                        sf.write(str(out_path), audio_out, result.sr)

                        st.success("Generated successfully!")
                        st.audio(out_path.read_bytes(), format="audio/wav")
                        st.download_button(
                            "Download WAV",
                            data=out_path.read_bytes(),
                            file_name=out_path.name,
                            mime="audio/wav",
                            key="p2_download",
                        )

                        # Generation info panel
                        st.subheader("Generation Info")
                        ci1, ci2, ci3 = st.columns(3)
                        ci1.metric("Target Age", f"{result.target_age:.1f}")
                        ci2.metric("Source Age", f"{result.source_age:.1f}")
                        ci3.metric("Confidence", f"{result.confidence:.2f}")
                        st.write(f"**Method:** `{result.method}`")
                        st.write(f"**Source:** {result.source_info}")
                        if result.versions_used:
                            st.write(f"**Versions used:** {', '.join(result.versions_used)}")
                        if result.interp_weight is not None:
                            st.write(f"**Interpolation weight:** {result.interp_weight:.2f}")

                    except ValueError as ve:
                        st.error(str(ve))
                    except FileNotFoundError as fe:
                        st.error(f"Audio file missing: {fe}")
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")
                        st.exception(exc)

        st.divider()

        # ── Age sweep sample pack ───────────────────────────────────────────
        st.subheader("Age Sweep Sample Pack")
        st.caption(
            "Generates a ZIP of your voice at every age in a range, "
            "using stored recordings as the reference signal."
        )

        if not timeline_entries:
            st.info("Upload recordings in Part 1 to enable sample packs.")
        else:
            sw1, sw2 = st.columns(2)
            sweep_min = sw1.number_input(
                "Start age", min_value=5, max_value=95, value=20, step=1,
                key="sweep_min",
            )
            sweep_max = sw2.number_input(
                "End age", min_value=6, max_value=100, value=70, step=1,
                key="sweep_max",
            )
            sweep_step = st.selectbox(
                "Step (years)", [1, 2, 5, 10], index=2, key="sweep_step"
            )
            btn_sweep = st.button("Generate Sweep ZIP", key="sweep_btn")

            if btn_sweep:
                if int(sweep_min) >= int(sweep_max):
                    st.error("Start age must be less than end age.")
                else:
                    sweep_ages = list(
                        range(int(sweep_min), int(sweep_max) + 1, int(sweep_step))
                    )
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    sweep_dir = (
                        PROJECT_ROOT / "data" / "outputs" / "age_playback"
                        / f"sweep_{selected_user}_{ts}"
                    )
                    sweep_dir.mkdir(parents=True, exist_ok=True)
                    prog = st.progress(0)
                    errors = 0
                    for i, a in enumerate(sweep_ages, start=1):
                        try:
                            res = svc.play_at_age(float(a))
                            wav_out = res.audio.astype("float32")
                            mx = float(np.abs(wav_out).max() + 1e-9)
                            if mx > 1.0:
                                wav_out = wav_out / mx * 0.9
                            sf.write(str(sweep_dir / f"age_{a:03d}.wav"), wav_out, res.sr)
                        except Exception as exc:
                            errors += 1
                        prog.progress(int(i * 100 / len(sweep_ages)))

                    zip_bytes = make_zip_bytes(sweep_dir)
                    msg = f"Sweep ready: {len(sweep_ages) - errors} files"
                    if errors:
                        msg += f" ({errors} skipped)"
                    st.success(msg)
                    st.download_button(
                        "Download Sweep ZIP",
                        data=zip_bytes,
                        file_name=f"{selected_user}_sweep_{ts}.zip",
                        mime="application/zip",
                        key="sweep_download",
                    )


if __name__ == "__main__" or True:
    run_app()