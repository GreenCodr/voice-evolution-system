import os
import numpy as np
import torch
from pathlib import Path
import soundfile as sf

# Patch huggingface_hub to ignore unexpected 'use_auth_token' kwarg if present
try:
    import huggingface_hub as hf_hub
    _orig_hf_hub_download = getattr(hf_hub, "hf_hub_download", None)
    if _orig_hf_hub_download is not None:
        def _patched_hf_hub_download(*args, **kwargs):
            # remove deprecated/unsupported kwargs that some callers may pass
            kwargs.pop("use_auth_token", None)
            kwargs.pop("token", None)
            return _orig_hf_hub_download(*args, **kwargs)
        hf_hub.hf_hub_download = _patched_hf_hub_download
except Exception:
    # if huggingface_hub not present or patch fails, fallback to normal behavior
    pass

# Use speechbrain to compute ECAPA-TDNN speaker embedding
# speechbrain.pretrained is deprecated but still works; we keep it for compatibility
from speechbrain.pretrained import EncoderClassifier

def compute_embedding(wav_path, device="cpu"):
    # load pretrained ECAPA-TDNN speaker encoder (VoxCeleb-trained)
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                               run_opts={"device": device},
                                               savedir="tmp_sbp_spkrec")
    embeddings = classifier.encode_batch(wav_path)  # returns tensor shape (1, emb_dim)
    emb = embeddings.squeeze(0).detach().cpu().numpy()
    return emb

if __name__ == "__main__":
    inp = "preprocessed/test_tone_preproc.wav"
    out_dir = "embeddings"
    os.makedirs(out_dir, exist_ok=True)
    print("Computing embedding for:", inp)
    device = "mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else "cpu"
    print("Using device:", device)
    emb = compute_embedding(inp, device=device)
    print("Embedding shape:", emb.shape)
    print("First 8 values:", emb[:8].tolist())
    out_path = os.path.join(out_dir, Path(inp).stem + "_emb.npy")
    np.save(out_path, emb)
    print("Saved embedding to:", out_path)
