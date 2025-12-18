# scripts/phase3_model.py

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base-960h"


class AgeSpeakerModel(nn.Module):
    """
    Speaker-aware embedding model (Phase 3)
    """

    def __init__(self, num_speakers: int, device="cpu"):
        super().__init__()

        # -------- Backbone --------
        self.backbone = Wav2Vec2Model.from_pretrained(MODEL_NAME)

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        hidden = self.backbone.config.hidden_size  # 768

        # -------- Embedding Head (DEFINED HERE ✅) --------
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128)
        )

        # -------- Speaker Classifier --------
        self.classifier = nn.Linear(128, num_speakers)

        self.device = device
        self.to(device)

    def forward(self, audio):
        """
        audio: (B, T)
        """
        outputs = self.backbone(audio, return_dict=True)
        hidden = outputs.last_hidden_state           # (B, T, H)

        pooled = hidden.mean(dim=1)
        pooled = pooled.contiguous()                 # 🔥 MPS-safe

        emb = self.embedding_head(pooled)            # (B, 128)
        logits = self.classifier(emb)                # (B, num_speakers)

        return emb, logits