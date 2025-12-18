# src/playback_select.py
import csv
from pathlib import Path
from datetime import datetime, date
import argparse
import sys

def iso_to_date(s):
    return datetime.fromisoformat(s).date()

def compute_age(dob_date, when_date):
    years = when_date.year - dob_date.year
    if (when_date.month, when_date.day) < (dob_date.month, dob_date.day):
        years -= 1
    return years

def load_versions(meta_path):
    rows = []
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return rows
    with open(meta_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            # ensure types
            try:
                r['version_id'] = int(r['version_id'])
                r['age_at_recording'] = int(r['age_at_recording'])
            except Exception:
                continue
            rows.append(r)
    return rows

def choose_closest(versions, target_age):
    best = None
    best_diff = None
    for v in versions:
        diff = abs(v['age_at_recording'] - target_age)
        if best is None or diff < best_diff:
            best = v
            best_diff = diff
    return best, best_diff

def main(meta_file="versions/versions.csv", age=None, years_from_now=None, dob=None):
    versions = load_versions(meta_file)
    if not versions:
        print("No versions found (versions/versions.csv missing or empty).")
        return 1

    if years_from_now is not None:
        if dob is None:
            print("Error: --dob required when using --years_from_now")
            return 2
        dob_date = iso_to_date(dob)
        today = datetime.utcnow().date()
        current_age = compute_age(dob_date, today)
        target_age = current_age + int(years_from_now)
        print(f"Computed target age from DOB {dob}: current age {current_age} -> target {target_age}")
    elif age is not None:
        target_age = int(age)
    else:
        print("Error: provide --age or --years_from_now (with --dob).")
        return 3

    chosen, diff = choose_closest(versions, target_age)
    if chosen is None:
        print("No suitable version found.")
        return 4

    versions_dir = Path(meta_file).parent
    audio_path = versions_dir / chosen['audio_file']
    print("Selected version:")
    print("  version_id:", chosen['version_id'])
    print("  age_at_recording:", chosen['age_at_recording'])
    print("  best_similarity:", chosen.get('best_similarity', 'N/A'))
    print("  audio_file:", str(audio_path))
    print("  notes:", chosen.get('notes',''))
    print(f"Target age: {target_age} (difference {diff} years)")

    # For the MVP we just print the audio path to play.
    # Example: frontend / API can return this path and stream the file.
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="versions/versions.csv", help="Versions metadata CSV")
    parser.add_argument("--age", type=int, help="Target age in years (e.g. 8)")
    parser.add_argument("--years_from_now", type=int, help="e.g. 10 -> target age = current_age + 10 (requires --dob)")
    parser.add_argument("--dob", type=str, help="User DOB in YYYY-MM-DD (required for years_from_now)")
    args = parser.parse_args()
    sys.exit(main(meta_file=args.meta, age=args.age, years_from_now=args.years_from_now, dob=args.dob))