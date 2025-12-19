# scripts/extract_speech_commands_children.py

import random
import shutil
from pathlib import Path

SRC_ROOT = Path("datasets/speech_commands")
DST_ROOT = Path("datasets/common_voice/age_audio_children/raw")
TARGET_COUNT = 220

def main():
    wav_files = []

    for d in SRC_ROOT.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("_"):
            continue  # skip _background_noise_

        wav_files.extend(list(d.glob("*.wav")))

    print("Total available child clips:", len(wav_files))

    random.shuffle(wav_files)
    selected = wav_files[:TARGET_COUNT]

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(selected, 1):
        dst = DST_ROOT / f"child_{i:04d}.wav"
        shutil.copy(src, dst)

        if i % 50 == 0:
            print(f"Copied {i}/{TARGET_COUNT}")

    print("✅ Child audio extraction complete")

if __name__ == "__main__":
    main()
