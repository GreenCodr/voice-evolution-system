# scripts/voice_label.py

import json
from pathlib import Path
from datetime import datetime

def write_voice_metadata(
    audio_path: str,
    voice_type: str,
    note: str = ""
):
    meta = {
        "voice_type": voice_type,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "note": note
    }

    meta_path = Path(audio_path).with_suffix(".meta.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"🏷️ Voice type: {voice_type}")
    print(f"📄 Metadata saved: {meta_path}")