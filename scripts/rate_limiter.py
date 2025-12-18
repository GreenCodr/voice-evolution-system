# scripts/rate_limiter.py

import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
RATE_LIMIT_FILE = RUNTIME_DIR / "rate_limits.json"

# ------------------ CONFIG (PRODUCTION SAFE) ------------------

MAX_REQUESTS = 5          # max synth requests
WINDOW_SECONDS = 600      # 10 minutes

# -------------------------------------------------------------


def _load_state() -> dict:
    if not RATE_LIMIT_FILE.exists():
        return {}
    try:
        with open(RATE_LIMIT_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: dict):
    RUNTIME_DIR.mkdir(exist_ok=True)
    with open(RATE_LIMIT_FILE, "w") as f:
        json.dump(state, f, indent=2)


def check_rate_limit(user_id: str) -> dict:
    """
    Enforces per-user rate limiting.

    Returns:
    {
        allowed: bool,
        remaining: int,
        reset_in_sec: int
    }
    """

    now = int(time.time())
    state = _load_state()

    user = state.get(user_id, {
        "window_start": now,
        "count": 0
    })

    elapsed = now - user["window_start"]

    # Reset window
    if elapsed >= WINDOW_SECONDS:
        user = {
            "window_start": now,
            "count": 0
        }

    # Reject if limit exceeded
    if user["count"] >= MAX_REQUESTS:
        return {
            "allowed": False,
            "remaining": 0,
            "reset_in_sec": max(0, WINDOW_SECONDS - elapsed)
        }

    # Accept request
    user["count"] += 1
    state[user_id] = user
    _save_state(state)

    return {
        "allowed": True,
        "remaining": MAX_REQUESTS - user["count"],
        "reset_in_sec": max(0, WINDOW_SECONDS - elapsed)
    }