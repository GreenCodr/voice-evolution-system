from pathlib import Path
import csv
from typing import Optional, Dict, Any

def _load_versions_from_csv(meta_path: Path):
    rows = []
    if not meta_path.exists():
        return rows
    with open(meta_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                r["version_id"] = int(r.get("version_id", "") or 0)
            except Exception:
                # skip malformed row
                continue
            # normalize types and missing fields
            try:
                r["age_at_recording"] = (
                    int(r.get("age_at_recording"))
                    if r.get("age_at_recording") not in (None, "", "None")
                    else None
                )
            except Exception:
                r["age_at_recording"] = None
            r["audio_file"] = r.get("audio_file", "") or ""
            rows.append(r)
    return rows

def _choose_closest_version(versions, target_age: int):
    best = None
    best_diff = None
    for v in versions:
        age = v.get("age_at_recording")
        if age is None:
            continue
        diff = abs(age - target_age)
        if best is None or diff < best_diff:
            best = v
            best_diff = diff
    return best, best_diff

def select_version_by_age(age: int, meta_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Select the stored version closest to `age`.
    Returns a dict: {"audio_path": <str>, "meta": <row dict>} on success,
    or {"error": <str>, ...} on failure.
    """
    # meta_path can be passed in or we fallback to global VERSIONS_CSV
    if meta_path is None:
        try:
            meta_path = str(VERSIONS_CSV)
        except NameError:
            return {"error": "no_versions_defined", "message": "VERSIONS_CSV is not defined in the module."}

    meta_path = Path(meta_path)
    versions = _load_versions_from_csv(meta_path)
    if not versions:
        return {"error": "no_versions", "message": "versions CSV missing or empty"}

    chosen, diff = _choose_closest_version(versions, age)
    if chosen is None:
        # fallback: pick last entry (if any)
        chosen = versions[-1]

    # Resolve audio path robustly:
    audio_field = chosen.get("audio_file", "")
    audio_candidate = Path(audio_field)

    # If relative (not absolute) => make it relative to project root
    if not audio_candidate.is_absolute():
        audio_candidate = BASE_DIR / audio_candidate

    if audio_candidate.exists():
        return {"audio_path": str(audio_candidate), "meta": chosen}

    # Fallback: try versions/audio/<filename>
    fallback = BASE_DIR / "versions" / "audio" / Path(audio_field).name
    if fallback.exists():
        return {"audio_path": str(fallback), "meta": chosen}

    return {"error": "audio_not_found", "audio_path": str(audio_candidate), "meta": chosen}