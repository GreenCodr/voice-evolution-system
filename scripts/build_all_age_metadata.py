import pandas as pd
from pathlib import Path

rows = []

# ---------- ADULT (COMMON VOICE) ----------
adult_root = Path("datasets/common_voice/age_audio/raw")

for p in adult_root.glob("*.mp3"):
    rows.append({
        "audio_path": str(p),
        "age_group": "adult",
        "source": "common_voice"
    })

print("Adult samples:", len(rows))

# ---------- CHILDREN (SPEECH COMMANDS) ----------
child_root = Path("datasets/common_voice/age_audio_children/raw")

child_rows = []
for p in child_root.glob("*.wav"):
    child_rows.append({
        "audio_path": str(p),
        "age_group": "children",
        "source": "speech_commands"
    })

print("Children samples:", len(child_rows))

rows.extend(child_rows)

# ---------- SAVE ----------
df = pd.DataFrame(rows)
out = Path("datasets/common_voice/age_audio/all_age_metadata.csv")
df.to_csv(out, index=False)

print("\n✅ METADATA SAVED")
print("Total samples:", len(df))
print(df["age_group"].value_counts())