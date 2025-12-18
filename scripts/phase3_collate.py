# scripts/phase3_collate.py

import torch

def voice_collate_fn(batch):
    """
    Collate function for Phase-3
    Handles variable-length audio and speaker indices
    """

    audios = []
    speaker_idxs = []
    ages = []
    paths = []

    for item in batch:
        audios.append(item["audio"])
        speaker_idxs.append(item["speaker_idx"])
        ages.append(item.get("age"))
        paths.append(item.get("path"))

    # Pad audio to max length in batch
    audios = torch.nn.utils.rnn.pad_sequence(
        audios,
        batch_first=True
    )

    speaker_idxs = torch.tensor(speaker_idxs, dtype=torch.long)

    return {
        "audio": audios,
        "speaker_idx": speaker_idxs,
        "age": ages,
        "path": paths
    }