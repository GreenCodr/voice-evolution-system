# scripts/validate_manifest.py
"""
Stricter manifest validator.
Checks:
 - required columns present
 - file exists and is readable
 - duration >= min_duration (seconds)
 - simple SNR estimate >= min_snr_db (dB)
Writes an annotated manifest: data/manifest_validated.csv
"""

import csv
import sys
from pathlib import Path
import argparse
import soundfile as sf
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# simple SNR estimator using frame energy
def estimate_snr_db(audio, sr, frame_len=1024, hop=512):
    # mono audio assumed
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # frame energies
    frames = []
    for i in range(0, len(audio) - frame_len + 1, hop):
        f = audio[i:i+frame_len]
        frames.append(np.sum(f * f) / (len(f) + 1e-12))
    if not frames:
        return float("-inf")
    frames = np.array(frames)
    # estimate noise floor as lower percentile energy
    noise_floor = np.percentile(frames, 10)
    signal_floor = np.percentile(frames, 90)
    # avoid divide by zero
    if noise_floor <= 0:
        noise_floor = 1e-12
    snr = signal_floor / noise_floor
    snr_db = 10.0 * np.log10(snr + 1e-12)
    return float(snr_db)

def validate_row(row, min_duration, min_snr_db, required_cols):
    fp = PROJECT_ROOT / row.get("file_path", "")
    result = {"file_path": str(fp), "exists": False, "duration_s": None, "snr_db": None, "valid": False, "reject_reason": ""}
    # existence
    if not fp.exists():
        result["reject_reason"] = "missing_file"
        return result
    result["exists"] = True
    # read info
    try:
        info = sf.info(str(fp))
        duration = float(info.frames) / float(info.samplerate)
    except Exception:
        # fallback to read
        try:
            audio, sr = sf.read(str(fp))
            duration = len(audio) / float(sr)
        except Exception as e:
            result["reject_reason"] = f"read_error:{e}"
            return result

    result["duration_s"] = float(duration)
    if duration < min_duration:
        result["reject_reason"] = f"short_duration<{min_duration}s"
        return result

    # compute SNR estimate
    try:
        audio, sr = sf.read(str(fp))
        snr_db = estimate_snr_db(audio, sr)
        result["snr_db"] = float(snr_db)
        if min_snr_db is not None and snr_db < min_snr_db:
            result["reject_reason"] = f"low_snr<{min_snr_db}dB"
            return result
    except Exception:
        # if SNR fails, don't reject by SNR; just leave blank
        result["snr_db"] = None

    # check required metadata fields presence
    for c in required_cols:
        val = (row.get(c) or "").strip()
        if val == "":
            result["reject_reason"] = f"missing_meta:{c}"
            return result

    # passed all checks
    result["valid"] = True
    return result

def main(manifest_path, min_duration=10.0, min_snr_db=20.0, out_manifest="data/manifest_validated.csv"):
    manifest_path = PROJECT_ROOT / manifest_path
    if not manifest_path.exists():
        print("Manifest not found:", manifest_path)
        return 2

    required_cols = ["file_path", "speaker_id", "dob", "recording_date", "dataset_source"]

    rows = []
    summary = {"total": 0, "valid": 0, "invalid": 0}

    with open(manifest_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        # ensure required columns present (report but continue)
        missing_columns = [c for c in required_cols if c not in fieldnames]
        if missing_columns:
            print("WARNING: manifest missing required columns:", missing_columns)
        for i, r in enumerate(reader, 1):
            summary["total"] += 1
            res = validate_row(r, min_duration=min_duration, min_snr_db=min_snr_db, required_cols=required_cols)
            out = dict(r)  # copy original
            out.update({
                "duration_s": "" if res["duration_s"] is None else f"{res['duration_s']:.3f}",
                "snr_db": "" if res["snr_db"] is None else f"{res['snr_db']:.2f}",
                "valid": "true" if res["valid"] else "false",
                "reject_reason": res["reject_reason"]
            })
            rows.append(out)
            if res["valid"]:
                summary["valid"] += 1
                print(f"[OK] row {i}: {r.get('file_path')} dur={out['duration_s']}s snr={out['snr_db']}")
            else:
                summary["invalid"] += 1
                print(f"[REJECT] row {i}: {r.get('file_path')} reason={res['reject_reason']} dur={out['duration_s']} snr={out['snr_db']}")

    # write annotated manifest
    outp = PROJECT_ROOT / out_manifest
    out_fieldnames = fieldnames + ["duration_s", "snr_db", "valid", "reject_reason"]
    with open(outp, "w", newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Wrote validated manifest:", outp)
    print("Summary:", summary)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Path to manifest CSV relative to project root (e.g. data/manifest.csv)")
    parser.add_argument("--min-duration", type=float, default=10.0, help="Minimum duration in seconds")
    parser.add_argument("--min-snr-db", type=float, default=20.0, help="Minimum SNR in dB (simple estimator)")
    parser.add_argument("--out", default="data/manifest_validated.csv", help="Output annotated manifest")
    args = parser.parse_args()
    sys.exit(main(args.manifest, min_duration=args.min_duration, min_snr_db=args.min_snr_db, out_manifest=args.out))