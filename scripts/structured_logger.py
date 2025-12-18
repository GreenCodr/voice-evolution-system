# scripts/structured_logger.py

import json
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "voice_evolution.log"


def log_event(event_type: str, payload: dict):
    """
    Write structured JSON log for every important system decision.
    """

    record = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "payload": payload,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")