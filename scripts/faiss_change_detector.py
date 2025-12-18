import sys
import csv
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
import faiss

# ------------------ PATH SETUP ------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ------------------ IMPORTS ------------------

from audio_quality import audio_quality_gate
from speaker_verification import speaker_verification_gate
from device_fingerprint import extract_device_fingerprint, device_match_score
from confidence_engine import compute_confidence
from version_decision import decide_voice_version

# 🔷 NEW: User Registry
from user_registry import UserRegistry

try:
    from register_version import register_version
except Exception:
    register_version = None

# ------------------ CONSTANTS ------------------

EMB_DIR = PROJECT_ROOT / "embeddings" / "ecapa"
VERSIONS_DIR = PROJECT_ROOT / "versions"
VERSIONS_AUDIO_DIR = VERSIONS_DIR / "audio"

MANIFEST_EMB = PROJECT_ROOT / "data" / "librispeech_manifest_small_emb.csv"

THRESHOLD = 0.75

# ------------------ HELPERS ------------------

def load_embedding(p: Path) -> np.ndarray:
    return np.load(p).astype("float32")

def normalize(e: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(e)
    return e if n == 0 else e / n

def read_versions(target_dim: int):
    names, embs = [], []

    if not VERSIONS_DIR.exists():
        return names, embs

    for p in VERSIONS_DIR.glob("*.npy"):
        try:
            e = normalize(load_embedding(p))
            if e.shape[0] != target_dim:
                continue
            embs.append(e)
            names.append(p.name)
        except Exception:
            pass

    return names, embs

def build_index(embs):
    if not embs:
        return None
    d = embs[0].shape[0]
    index = faiss.IndexFlatIP(d)
    index.add(np.vstack(embs))
    return index

def read_manifest():
    rows = {}
    if not MANIFEST_EMB.exists():
        print("❌ Manifest not found:", MANIFEST_EMB)
        return rows

    with open(MANIFEST_EMB, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[Path(r["emb_path"]).name] = r

    return rows

def compute_age(dob: str | None, rec: str | None):
    if not dob or not rec:
        return None
    try:
        dob = datetime.fromisoformat(dob).date()
        rec = datetime.fromisoformat(rec).date()
        age = rec.year - dob.year
        if (rec.month, rec.day) < (dob.month, dob.day):
            age -= 1
        return age
    except Exception:
        return None

# ------------------ MAIN ------------------

def main(threshold=THRESHOLD):
    print("FAISS change detector running | threshold =", threshold)

    # 🔷 NEW: Initialize User Registry
    user = UserRegistry("user_001")

    manifest = read_manifest()

    candidates = list(EMB_DIR.glob("*.npy"))
    if not candidates:
        print("❌ No embeddings found")
        return 1

    target_dim = load_embedding(candidates[0]).shape[0]

    version_names, version_embs = read_versions(target_dim)
    index = build_index(version_embs)

    print(f"Loaded {len(version_embs)} existing versions")
    print(f"Embedding dimension = {target_dim}")
    print(f"Found {len(candidates)} candidate embeddings")

    for emb_path in candidates:
        if emb_path.name in version_names:
            continue

        emb = normalize(load_embedding(emb_path))
        if emb.shape[0] != target_dim:
            print(f"⚠️ Skipping {emb_path.name} (dimension mismatch)")
            continue

        best_sim = -1.0
        if index is not None:
            D, _ = index.search(np.expand_dims(emb, 0), 1)
            best_sim = float(D[0][0])

        print(f"\nChecking {emb_path.name} | similarity={best_sim:.4f}")

        row = manifest.get(emb_path.name)
        if not row:
            print("⚠️ Skipping non-dataset embedding:", emb_path.name)
            continue

        audio_rel = row.get("file_path") or row.get("preproc_path")
        if not audio_rel:
            print("❌ Missing audio path in manifest")
            continue

        audio_path = PROJECT_ROOT / audio_rel
        if not audio_path.exists():
            print("❌ Audio file missing:", audio_path)
            continue

        # -------- Gate 1: Audio Quality --------
        quality = audio_quality_gate(str(audio_path), dev_mode=True)
        if not quality["accepted"]:
            print("❌ Audio rejected:", quality["reason"])
            continue

        # -------- Gate 2: Speaker Verification --------
        speaker = speaker_verification_gate(emb, version_embs)
        if not speaker["accepted"]:
            print("❌ Speaker mismatch")
            continue

        # -------- Gate 3: Device Fingerprint --------
        device_score = 1.0
        if VERSIONS_AUDIO_DIR.exists():
            try:
                ref_audio = next(VERSIONS_AUDIO_DIR.glob("*.wav"))
                fp_ref = extract_device_fingerprint(str(ref_audio))
                fp_new = extract_device_fingerprint(str(audio_path))
                device_score = device_match_score(fp_new, fp_ref)
            except StopIteration:
                pass

        # -------- Confidence --------
        confidence = compute_confidence(
            duration_s=quality["duration"],
            snr_db=quality["snr_db"],
            speaker_similarity=speaker["best_similarity"],
            device_match=device_score,
            history_count=len(version_embs)
        )

        decision = decide_voice_version(
            similarity=best_sim,
            confidence=confidence,
            speaker_ok=True,
            device_match=device_score,
            embedding_path=str(emb_path.relative_to(PROJECT_ROOT)),
            audio_path=str(audio_path.relative_to(PROJECT_ROOT)),
            user_dob=row.get("dob"),
        )

        print("➡️ Decision:", decision["action"])

        if decision["action"] != "CREATE_VERSION":
            continue

        # -------- Persist embedding --------
        VERSIONS_DIR.mkdir(exist_ok=True)
        final_emb_path = VERSIONS_DIR / emb_path.name
        shutil.move(str(emb_path), str(final_emb_path))

        # 🔷 NEW: Register in User Registry
        user.add_voice_version(
            version_id=emb_path.stem,
            embedding_path=str(final_emb_path.relative_to(PROJECT_ROOT)),
            audio_path=str(audio_path.relative_to(PROJECT_ROOT)),
            confidence=confidence,
            voice_type="RECORDED"
        )

        if register_version:
            register_version(
                str(final_emb_path),
                str(audio_path),
                row.get("dob", ""),
                versions_dir=str(VERSIONS_DIR),
                notes="auto-created"
            )

        version_embs.append(emb)
        if index is None:
            index = build_index([emb])
        else:
            index.add(np.expand_dims(emb, 0))

        print("✅ Version created & registered")

    print("\nFAISS detection complete")
    return 0

# ------------------ ENTRY ------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()
    sys.exit(main(args.threshold))