# scripts/embed_manifest.py
import csv
import sys
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2Processor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "facebook/wav2vec2-base-960h"


def load_audio(path, target_sr=16000):
    try:
        audio, sr = sf.read(path)
    except Exception as e:
        print(f"❌ ERROR reading audio file {path}: {e}")
        return None, None

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        import librosa
        try:
            audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=target_sr)
        except Exception as e:
            print(f"❌ ERROR resampling {path}: {e}")
            return None, None
        sr = target_sr

    return audio, sr


def compute_wav2vec_embedding(audio, sr, model, processor, device):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values, return_dict=True)
        hidden = outputs.last_hidden_state.squeeze(0)
        emb = hidden.mean(dim=0).cpu().numpy()

    return emb


def main(manifest_in="data/manifest_preproc.csv",
         manifest_out="data/manifest_emb.csv",
         device=None):

    manifest_in = PROJECT_ROOT / manifest_in
    if not manifest_in.exists():
        print("❌ Manifest not found:", manifest_in)
        return 2

    if device is None:
        device = "mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else "cpu"
    print("Using device:", device)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)

    emb_dir = PROJECT_ROOT / "embeddings"
    emb_dir.mkdir(exist_ok=True)

    out_rows = []

    with open(manifest_in, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)

        # ✅ ADD audio_path column
        fieldnames = list(reader.fieldnames) + ["emb_path", "audio_path"]

        for r in reader:
            preproc = r.get("preproc_path") or r.get("file_path")
            if not preproc:
                r["emb_path"] = ""
                r["audio_path"] = ""
                out_rows.append(r)
                continue

            candidates = []
            p = str(preproc)

            candidates.append(PROJECT_ROOT / p)

            if "_preproc_preproc" in p:
                candidates.append(PROJECT_ROOT / p.replace("_preproc_preproc", "_preproc"))

            if p.endswith("_preproc.wav"):
                candidates.append(PROJECT_ROOT / p.replace("_preproc.wav", ".wav"))

            if r.get("file_path"):
                candidates.append(PROJECT_ROOT / r["file_path"])

            src = None
            for c in candidates:
                if Path(c).exists():
                    src = Path(c)
                    break

            if src is None:
                print("❌ Missing audio file for row:", r)
                r["emb_path"] = ""
                r["audio_path"] = ""
                out_rows.append(r)
                continue

            print("➡️ Processing:", src)

            audio, sr = load_audio(str(src), target_sr=processor.feature_extractor.sampling_rate)
            if audio is None:
                r["emb_path"] = ""
                r["audio_path"] = ""
                out_rows.append(r)
                continue

            emb = compute_wav2vec_embedding(audio, sr, model, processor, device)

            emb_name = src.stem + "_wav2vec_emb.npy"
            emb_path = emb_dir / emb_name
            np.save(emb_path, emb)

            # ✅ CRITICAL FIX
            r["emb_path"] = str(emb_path.relative_to(PROJECT_ROOT))
            r["audio_path"] = str(src.relative_to(PROJECT_ROOT))

            out_rows.append(r)

    manifest_out = PROJECT_ROOT / manifest_out
    with open(manifest_out, "w", newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print("\n✅ Done! Embedding manifest saved to:", manifest_out)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="manifest_in", default="data/manifest_preproc.csv")
    parser.add_argument("--out", dest="manifest_out", default="data/manifest_emb.csv")
    parser.add_argument("--device", dest="device", default=None)
    args = parser.parse_args()

    sys.exit(main(
        manifest_in=args.manifest_in,
        manifest_out=args.manifest_out,
        device=args.device
    ))