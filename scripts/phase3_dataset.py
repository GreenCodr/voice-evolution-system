# scripts/phase3_dataset.py

import csv
import json
import torch
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UnifiedVoiceDataset(Dataset):
    """
    Unified dataset for Phase 3
    Returns:
      {
        audio: Tensor (T,)
        speaker_idx: int
        age: float | None
        path: str
      }
    """

    def __init__(self, manifest_paths, max_samples=None):
        self.samples = []

        # -------- Load speaker map --------
        speaker_map_path = PROJECT_ROOT / "data" / "speaker_map.json"
        with open(speaker_map_path, "r") as f:
            self.speaker_map = json.load(f)

        for mp in manifest_paths:
            path = PROJECT_ROOT / mp
            if not path.exists():
                continue

            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    speaker_id = row.get("speaker_id")
                    if speaker_id is None:
                        continue

                    speaker_id = str(speaker_id)
                    if speaker_id not in self.speaker_map:
                        continue

                    audio_path = row.get("file_path") or row.get("preproc_path")
                    if not audio_path:
                        continue

                    full_audio_path = PROJECT_ROOT / audio_path
                    if not full_audio_path.exists():
                        continue

                    age = row.get("age")
                    age = float(age) if age not in (None, "", "None") else None

                    self.samples.append({
                        "path": str(full_audio_path),
                        "speaker_idx": self.speaker_map[speaker_id],
                        "age": age
                    })

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"✅ Unified dataset ready | Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        audio, sr = sf.read(item["path"])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = torch.tensor(audio, dtype=torch.float32)

        return {
            "audio": audio,
            "speaker_idx": item["speaker_idx"],
            "age": item["age"],
            "path": item["path"]
        }