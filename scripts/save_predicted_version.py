from pathlib import Path
import numpy as np
from datetime import datetime, timezone
from scripts.user_registry import UserRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def save_predicted_version(
    user_id: str,
    aged_embedding: np.ndarray,
    target_age: int,
    base_version: dict,
    confidence: float = 0.9
):
    user = UserRegistry(user_id)

    version_id = str(int(datetime.now(timezone.utc).timestamp()))
    emb_dir = PROJECT_ROOT / "versions" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    emb_path = emb_dir / f"{user_id}_predicted_{target_age}_{version_id}.npy"
    np.save(emb_path, aged_embedding)

    user.add_voice_version(
        version_id=version_id,
        embedding_path=str(emb_path.relative_to(PROJECT_ROOT)),
        audio_path=None,
        confidence=confidence,
        voice_type="PREDICTED",
        age_at_recording=target_age
    )

    return {
        "version_id": version_id,
        "embedding_path": str(emb_path),
        "age": target_age
    }