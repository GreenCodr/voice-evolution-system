import numpy as np
from pathlib import Path
import tempfile
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def synthesize_aged_voice(
    text: str,
    aged_embedding: np.ndarray,
    reference_wav: str,
    out_wav: str,
):
    """
    Synthesize voice using aged embedding but real speaker timbre.
    """

    # 1️⃣ Save aged embedding temporarily
    tmp_emb = Path(tempfile.gettempdir()) / "aged_embedding.npy"
    np.save(tmp_emb, aged_embedding)

    # 2️⃣ Call existing synthesis pipeline
    subprocess.run(
        [
            sys.executable,
            "scripts/synthesize_from_embedding.py",
            text,
            out_wav,
            "--speaker_wav",
            reference_wav,
        ],
        check=True,
    )

    return out_wav