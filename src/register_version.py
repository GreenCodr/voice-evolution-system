# src/register_version.py
import os
import sys
import shutil
import csv
from pathlib import Path
from datetime import datetime, date
import argparse
import numpy as np

def iso_to_date(s):
    return datetime.fromisoformat(s).date()

def compute_age(dob_date, when_date):
    # dob_date and when_date are date objects
    years = when_date.year - dob_date.year
    if (when_date.month, when_date.day) < (dob_date.month, dob_date.day):
        years -= 1
    return years

def register_version(emb_path, audio_path, dob, versions_dir="versions", notes=""):
    versions_dir = Path(versions_dir)
    versions_dir.mkdir(parents=True, exist_ok=True)

    # version id as current UTC timestamp integer
    ts = datetime.utcnow()
    version_id = int(ts.timestamp())

    # destination audio path
    audio_dst_dir = versions_dir / "audio"
    audio_dst_dir.mkdir(parents=True, exist_ok=True)
    audio_ext = Path(audio_path).suffix or ".wav"
    audio_dst_name = f"{version_id}{audio_ext}"
    audio_dst = audio_dst_dir / audio_dst_name

    # copy (or move) audio into versions/audio
    shutil.copy2(audio_path, audio_dst)

    # compute age at recording from DOB (ISO YYYY-MM-DD)
    try:
        dob_date = iso_to_date(dob)
    except Exception as e:
        raise ValueError("DOB must be in ISO format YYYY-MM-DD") from e

    # use file modification time as recording time (UTC)
    mtime = Path(audio_path).stat().st_mtime
    rec_dt = datetime.utcfromtimestamp(mtime)
    age_at_recording = compute_age(dob_date, rec_dt.date())

    # metadata CSV
    meta_file = versions_dir / "versions.csv"
    existed = meta_file.exists()
    with open(meta_file, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not existed:
            writer.writerow(["version_id","timestamp_utc","age_at_recording","emb_file","audio_file","notes"])
        emb_fname = Path(emb_path).name
        writer.writerow([version_id, rec_dt.isoformat()+"Z", age_at_recording, emb_fname, audio_dst_name, notes])

    print("Registered version:", version_id)
    print("Audio copied to:", audio_dst)
    print("Metadata saved to:", meta_file)
    return version_id, str(audio_dst), str(meta_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True, help="Path to version embedding (.npy) inside versions/ (already moved there by detector)")
    parser.add_argument("--audio", required=True, help="Path to representative audio file (preprocessed wav)")
    parser.add_argument("--dob", required=True, help="User DOB in YYYY-MM-DD")
    parser.add_argument("--versions_dir", default="versions", help="Versions folder")
    parser.add_argument("--notes", default="", help="Optional notes")
    args = parser.parse_args()

    if not Path(args.emb).exists():
        print("ERROR: embedding not found:", args.emb); sys.exit(1)
    if not Path(args.audio).exists():
        print("ERROR: audio not found:", args.audio); sys.exit(1)

    register_version(args.emb, args.audio, args.dob, versions_dir=args.versions_dir, notes=args.notes)