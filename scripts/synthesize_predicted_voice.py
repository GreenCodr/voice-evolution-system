# scripts/synthesize_predicted_voice.py

import argparse
import soundfile as sf
from pathlib import Path
from TTS.api import TTS
from voice_label import write_voice_metadata
from rate_limiter import check_rate_limit

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

def main(text: str, out_path: str, speaker_wav: str):
    user_id = "user_001"

    # -------- Step 6.1: Rate limiting --------
    rate = check_rate_limit(user_id)
    if not rate["allowed"]:
        print("❌ Rate limit exceeded")
        print(f"⏳ Try again in {rate['reset_in_sec']} seconds")
        return

    out_path = Path(out_path)

    print("🔮 Generating PREDICTED voice (future extrapolation)")

    tts = TTS(model_name=MODEL_NAME)

    wav = tts.tts(
        text=text,
        speaker_wav=speaker_wav,
        language="en"
    )

    sf.write(out_path, wav, 24000)

    # -------- Step 6.3.3: PREDICTED label --------
    write_voice_metadata(
        audio_path=out_path,
        voice_type="PREDICTED",
        note="This voice is generated via extrapolation and is not a real recording"
    )

    print(f"✅ Output saved to: {out_path}")
    print("⚠️  NOTE: This voice is PREDICTED, not recorded")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("out")
    parser.add_argument("--speaker_wav", required=True)

    args = parser.parse_args()
    main(args.text, args.out, args.speaker_wav)