# scripts/voice_evolution_demo.py
import argparse
from pathlib import Path
import subprocess
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
VERSIONS = PROJECT_ROOT / "versions"
EMB = PROJECT_ROOT / "embeddings"

def run(cmd):
    print(">", " ".join(map(str, cmd)))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True, help='e.g. "Play my voice 10 years from now"')
    ap.add_argument("--text", required=True)
    ap.add_argument("--speaker_wav", required=True)
    ap.add_argument("--out", default="output.wav")
    args = ap.parse_args()

    req = args.request.lower()

    # Decide target (recorded vs predicted)
    if "future" in req or "years from now" in req:
        mode = "future"
    elif "past" in req or "years ago" in req:
        mode = "past"
    else:
        mode = "recorded"

    if mode == "recorded":
        # Pick closest recorded version by relative age 0.5 (middle) as demo
        run([
            sys.executable, SCRIPTS / "select_version_by_relative_age.py", "0.5"
        ])
        # Synthesize recorded voice
        run([
            sys.executable, SCRIPTS / "synthesize_from_embedding.py",
            args.text, args.out,
            "--speaker_wav", args.speaker_wav
        ])
        print("✅ RECORDED voice output:", args.out)

    else:
        # Predicted voice (future/past)
        run([
            sys.executable, SCRIPTS / "synthesize_predicted_voice.py",
            args.text, args.out,
            "--speaker_wav", args.speaker_wav
        ])
        print("⚠️  PREDICTED voice output:", args.out)

if __name__ == "__main__":
    main()