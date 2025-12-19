from pathlib import Path
import sys

# ✅ Fix Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hybrid_playback_decider import decide_playback_mode
from scripts.synthesize_from_embedding import synthesize_from_embedding

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

def main():
    decision = decide_playback_mode("user_002", 60)

    assert decision["mode"] == "AGED", "Expected AGED mode"

    out_wav = OUT_DIR / "aged_60.wav"

    synthesize_from_embedding(
        text="Hello. This is how I may sound when I am older.",
        out_path=str(out_wav),
        speaker_embedding=decision["embedding"],
        reference_wav=decision["reference_wav"],
    )

    print("✅ Aged voice generated:", out_wav)

if __name__ == "__main__":
    main()