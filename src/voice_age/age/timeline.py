"""
src/voice_age/age/timeline.py

Voice Timeline — maps stored voice versions to ages and resolves
age / date / future queries to the correct version(s).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup so this module works whether imported from project root or src/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[3]          # …/voice-evolution-system

import sys
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from user_registry import UserRegistry   # Part 1 — read-only access

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimelineEntry:
    """A single stored voice version with its age metadata."""
    version_id: str
    recorded_utc: str
    age_at_recording: Optional[float]
    audio_path: str
    embedding_path: str
    confidence: float
    voice_type: str

    @property
    def recorded_date(self) -> date:
        return datetime.fromisoformat(self.recorded_utc.replace("Z", "")).date()

    @property
    def audio_abs_path(self) -> Path:
        p = Path(self.audio_path)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p

    @property
    def embedding_abs_path(self) -> Optional[Path]:
        if not self.embedding_path:
            return None
        p = Path(self.embedding_path)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p


@dataclass
class TimelineResult:
    """
    Result of a version lookup.  Exactly one of the flags below is True.

    Flags
    -----
    is_exact          – target age matches a stored version exactly (±0.5 yr)
    is_interpolated   – target age falls between two stored versions
    is_extrapolate    – target age is before the earliest recording
    is_future         – target age is after the latest recording
    is_empty          – no versions stored at all
    """
    target_age: float

    # populated for most cases
    nearest: Optional[TimelineEntry] = None

    # populated for interpolation
    lower: Optional[TimelineEntry] = None
    upper: Optional[TimelineEntry] = None
    interp_weight: float = 0.0          # 0.0 = all lower, 1.0 = all upper

    # flags (exactly one True)
    is_exact: bool = False
    is_interpolated: bool = False
    is_extrapolate: bool = False
    is_future: bool = False
    is_empty: bool = False

    # current age of the user (set for future queries)
    current_age: Optional[float] = None


# ---------------------------------------------------------------------------
# VoiceTimeline
# ---------------------------------------------------------------------------

class VoiceTimeline:
    """
    Reads stored voice versions for a user and resolves queries of the form:
      - "What recording is closest to age X?"
      - "What two recordings bracket age X?"
      - "What is the latest recording + how far in the future is X?"
    """

    def __init__(self, user_id: str, date_of_birth: Optional[date] = None):
        self.user_id = user_id
        self._registry = UserRegistry(user_id)

        # DOB from argument takes priority; fall back to stored value
        stored_dob = self._registry.data.get("date_of_birth")
        if date_of_birth is not None:
            self.dob: Optional[date] = date_of_birth
        elif stored_dob:
            self.dob = datetime.strptime(stored_dob, "%Y-%m-%d").date()
        else:
            self.dob = None

        self._entries: List[TimelineEntry] = []
        self.build_timeline()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_timeline(self) -> None:
        """Reload all voice versions from the user registry and compute ages."""
        self._entries = []
        versions = self._registry.get_versions()

        for v in versions:
            age: Optional[float] = v.get("age_at_recording")

            # If age is missing but we have DOB, recompute it
            if age is None and self.dob:
                rec_utc = v.get("recorded_utc", "")
                try:
                    rec_date = datetime.fromisoformat(
                        rec_utc.replace("Z", "")
                    ).date()
                    age = float(_compute_age(self.dob, rec_date))
                except Exception:
                    age = None

            if age is not None:
                age = float(age)

            entry = TimelineEntry(
                version_id=v.get("version_id", ""),
                recorded_utc=v.get("recorded_utc", ""),
                age_at_recording=age,
                audio_path=v.get("audio_path", ""),
                embedding_path=v.get("embedding_path", ""),
                confidence=float(v.get("confidence", 1.0)),
                voice_type=v.get("type", "RECORDED"),
            )
            self._entries.append(entry)

        # Sort by age (None ages go last)
        self._entries.sort(
            key=lambda e: (e.age_at_recording is None, e.age_at_recording or 0.0)
        )
        logger.debug(
            "Timeline built for %s: %d versions", self.user_id, len(self._entries)
        )

    # ------------------------------------------------------------------
    # Current age
    # ------------------------------------------------------------------

    @property
    def current_age(self) -> Optional[float]:
        if self.dob is None:
            return None
        return float(_compute_age(self.dob, date.today()))

    # ------------------------------------------------------------------
    # Core resolver
    # ------------------------------------------------------------------

    def get_version_at_age(self, target_age: float) -> TimelineResult:
        """
        Returns the best TimelineResult for the requested target_age.

        Logic (in order):
        1. No versions → is_empty
        2. Only one version → treat as nearest (exact or close)
        3. Exact match (within ±0.5 yr) → is_exact
        4. Between two versions → is_interpolated
        5. Before earliest → is_extrapolate  (use earliest version)
        6. After latest → is_future          (use latest version)
        """
        self.build_timeline()   # refresh from disk

        dated = [e for e in self._entries if e.age_at_recording is not None]

        if not dated:
            return TimelineResult(
                target_age=target_age,
                is_empty=True,
                current_age=self.current_age,
            )

        if len(dated) == 1:
            only = dated[0]
            age_gap = abs(target_age - (only.age_at_recording or target_age))
            if target_age > (only.age_at_recording or 0) + 0.5:
                return TimelineResult(
                    target_age=target_age,
                    nearest=only,
                    is_future=True,
                    current_age=self.current_age,
                )
            if target_age < (only.age_at_recording or 0) - 0.5:
                return TimelineResult(
                    target_age=target_age,
                    nearest=only,
                    is_extrapolate=True,
                    current_age=self.current_age,
                )
            return TimelineResult(
                target_age=target_age,
                nearest=only,
                is_exact=True,
                current_age=self.current_age,
            )

        ages = [e.age_at_recording for e in dated]
        min_age = min(ages)
        max_age = max(ages)

        # --- Before earliest ---
        if target_age < min_age:
            return TimelineResult(
                target_age=target_age,
                nearest=dated[0],
                is_extrapolate=True,
                current_age=self.current_age,
            )

        # --- After latest ---
        if target_age > max_age:
            return TimelineResult(
                target_age=target_age,
                nearest=dated[-1],
                is_future=True,
                current_age=self.current_age,
            )

        # --- Exact match (within 0.5 yr) ---
        closest_idx = min(range(len(dated)), key=lambda i: abs(dated[i].age_at_recording - target_age))
        if abs(dated[closest_idx].age_at_recording - target_age) <= 0.5:
            return TimelineResult(
                target_age=target_age,
                nearest=dated[closest_idx],
                is_exact=True,
                current_age=self.current_age,
            )

        # --- Interpolation: find bracketing pair ---
        lower_entry, upper_entry, weight = self._bracket(dated, target_age)
        # nearest = whichever bracket is closer
        nearest = lower_entry if weight <= 0.5 else upper_entry
        return TimelineResult(
            target_age=target_age,
            nearest=nearest,
            lower=lower_entry,
            upper=upper_entry,
            interp_weight=weight,
            is_interpolated=True,
            current_age=self.current_age,
        )

    def get_version_at_date(self, target_date: date) -> TimelineResult:
        """Convert a calendar date to an age and delegate."""
        if self.dob is None:
            raise ValueError(
                f"User {self.user_id} has no date_of_birth set — "
                "cannot convert date to age."
            )
        target_age = float(_compute_age_fractional(self.dob, target_date))
        return self.get_version_at_age(target_age)

    def get_version_at_year(self, year: int) -> TimelineResult:
        """Convert calendar year to age (uses Jan 1 of that year)."""
        return self.get_version_at_date(date(year, 1, 1))

    def get_version_in_future(self, years_from_now: float) -> TimelineResult:
        """Look up the version for (current_age + years_from_now)."""
        if self.current_age is None:
            raise ValueError(
                f"User {self.user_id} has no date_of_birth set — "
                "cannot compute future age."
            )
        target_age = self.current_age + float(years_from_now)
        return self.get_version_at_age(target_age)

    def get_full_timeline(self) -> List[TimelineEntry]:
        """Return all stored versions sorted by age (for UI visualisation)."""
        self.build_timeline()
        return list(self._entries)

    def get_available_range(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Returns (earliest_age, latest_age, predicted_max_age).
        predicted_max_age is capped at latest_age + 40 years.
        """
        dated = [e for e in self._entries if e.age_at_recording is not None]
        if not dated:
            return None, None, None
        ages = [e.age_at_recording for e in dated]
        earliest = min(ages)
        latest = max(ages)
        predicted_max = latest + 40.0
        return earliest, latest, predicted_max

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bracket(
        self,
        dated: List[TimelineEntry],
        target_age: float,
    ) -> Tuple[TimelineEntry, TimelineEntry, float]:
        """
        Find lower, upper bracketing entries and the interpolation weight.
        weight = 0.0 → entirely lower,  weight = 1.0 → entirely upper.
        """
        lower = dated[0]
        upper = dated[-1]
        for entry in dated:
            if entry.age_at_recording <= target_age:
                lower = entry
            else:
                upper = entry
                break

        age_lo = lower.age_at_recording
        age_hi = upper.age_at_recording
        span = age_hi - age_lo
        weight = 0.0 if span < 1e-6 else (target_age - age_lo) / span
        return lower, upper, float(weight)


# ---------------------------------------------------------------------------
# Age computation utilities
# ---------------------------------------------------------------------------

def _compute_age(dob: date, ref: date) -> int:
    age = ref.year - dob.year
    if (ref.month, ref.day) < (dob.month, dob.day):
        age -= 1
    return max(0, age)


def _compute_age_fractional(dob: date, ref: date) -> float:
    """Return age as a float (e.g. 23.75 years)."""
    from datetime import timedelta
    total_days = (ref - dob).days
    return max(0.0, total_days / 365.25)
