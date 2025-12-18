# scripts/preprocess_manifest.py
import csv
import sys
from pathlib import Path
import subprocess
import shlex
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_preprocess(input_path, out_dir="preprocessed", target_sr=16000, min_segment_s=0.5):
    # Calls your existing preprocess script (keeps logic in one place)
    in_full = str(PROJECT_ROOT / input_path)
    out_dir_full = str(PROJECT_ROOT / out_dir)
    cmd = f"python {PROJECT_ROOT/'src'/'preprocess.py'} --input {shlex.quote(in_full)} --out_dir {shlex.quote(out_dir_full)} --target_sr {target_sr} --min_segment_s {min_segment_s}"
    try:
        print("Running:", cmd)
        out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT, text=True)
        print(out)
    except subprocess.CalledProcessError as e:
        print("Preprocess failed for", input_path)
        print("Output:", e.output)
        return None

    # expected output file name (same stem + _preproc.wav in out_dir)
    stem = Path(input_path).stem
    out_path = Path(out_dir) / f"{stem}_preproc.wav"
    return str(out_path)

def preprocess_manifest(manifest_in="data/manifest.csv", manifest_out="data/manifest_preproc.csv"):
    manifest_in = PROJECT_ROOT / manifest_in
    if not manifest_in.exists():
        print("Manifest not found:", manifest_in)
        return 2

    rows = []
    with open(manifest_in, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames) + ["preproc_path"]
        for r in reader:
            file_path = r.get("file_path") or r.get("audio_path")
            if not file_path:
                print("Skipping row, missing file path:", r)
                continue
    
            preproc = run_preprocess(file_path, out_dir="preprocessed", target_sr=16000, min_segment_s=0.25)
            if preproc is None:
                r["preproc_path"] = ""
            else:
                r["preproc_path"] = preproc
            rows.append(r)

    # write new manifest
    manifest_out = PROJECT_ROOT / manifest_out
    with open(manifest_out, "w", newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Wrote preprocessed manifest:", manifest_out)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="manifest_in", default="data/manifest.csv")
    parser.add_argument("--out", dest="manifest_out", default="data/manifest_preproc.csv")
    args = parser.parse_args()
    sys.exit(preprocess_manifest(manifest_in=args.manifest_in, manifest_out=args.manifest_out))