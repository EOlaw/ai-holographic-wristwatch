"""Mood monitoring via behavioural and physiological signal fusion."""
from __future__ import annotations

import threading
import time
import random
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Deque, List, Optional
from collections import deque

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class MoodState(Enum):
    JOYFUL = "joyful"
    CONTENT = "content"
    NEUTRAL = "neutral"
    ANXIOUS = "anxious"
    FATIGUED = "fatigued"
    IRRITABLE = "irritable"
    DEPRESSED = "depressed"
    EXCITED = "excited"


class MoodValence(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


MOOD_VALENCE_MAP: Dict[MoodState, MoodValence] = {
    MoodState.JOYFUL: MoodValence.POSITIVE,
    MoodState.CONTENT: MoodValence.POSITIVE,
    MoodState.EXCITED: MoodValence.POSITIVE,
    MoodState.NEUTRAL: MoodValence.NEUTRAL,
    MoodState.ANXIOUS: MoodValence.NEGATIVE,
    MoodState.FATIGUED: MoodValence.NEGATIVE,
    MoodState.IRRITABLE: MoodValence.NEGATIVE,
    MoodState.DEPRESSED: MoodValence.NEGATIVE,
}


@dataclass
class MoodReading:
    """Computed mood state with confidence and signals."""
    state: MoodState
    valence: MoodValence
    arousal: float          # 0-1, low=calm, high=energized
    confidence: float       # 0-1
    signals: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    note: str = ""

    def to_dict(self) -> Dict:
        return {
            "state": self.state.value,
            "valence": self.valence.value,
            "arousal": round(self.arousal, 3),
            "confidence": round(self.confidence, 3),
            "signals": {k: round(v, 3) for k, v in self.signals.items()},
            "timestamp": self.timestamp,
            "note": self.note,
        }


class MoodMonitor:
    """Tracks user mood using physiological and behavioural signals."""

    # Arousal bands: (low_arousal_signals, high_arousal_signals)
    MOOD_CRITERIA: Dict[MoodState, Dict] = {
        MoodState.JOYFUL:    {"min_valence": 0.6, "arousal_range": (0.4, 1.0)},
        MoodState.CONTENT:   {"min_valence": 0.3, "arousal_range": (0.0, 0.5)},
        MoodState.EXCITED:   {"min_valence": 0.5, "arousal_range": (0.7, 1.0)},
        MoodState.NEUTRAL:   {"min_valence": -0.2, "arousal_range": (0.2, 0.6)},
        MoodState.ANXIOUS:   {"min_valence": -0.9, "arousal_range": (0.5, 1.0)},
        MoodState.FATIGUED:  {"min_valence": -0.3, "arousal_range": (0.0, 0.3)},
        MoodState.IRRITABLE: {"min_valence": -0.6, "arousal_range": (0.4, 0.8)},
        MoodState.DEPRESSED: {"min_valence": -0.9, "arousal_range": (0.0, 0.4)},
    }

    def __init__(self, history_size: int = 200) -> None:
        self._lock = threading.RLock()
        self._history: Deque[MoodReading] = deque(maxlen=history_size)
        self._current: Optional[MoodReading] = None
        self._mood_duration: Dict[MoodState, float] = {m: 0.0 for m in MoodState}
        self._last_update: float = time.time()
        logger.info("MoodMonitor initialised")

    def _compute_valence(self, signals: Dict[str, float]) -> float:
        """Map biometric signals to a valence score [-1, 1]."""
        score = 0.0
        count = 0
        hr = signals.get("heart_rate")
        hrv = signals.get("hrv")
        stress = signals.get("stress_score")
        sleep_quality = signals.get("sleep_quality")
        activity = signals.get("activity_level")

        if hr is not None:
            # Moderate HR is positive
            if 60 <= hr <= 90:
                score += 0.2
            elif hr > 110:
                score -= 0.3
            count += 1
        if hrv is not None:
            hrv_norm = (hrv - 20) / 60  # normalise 20-80 ms range
            score += hrv_norm * 0.4
            count += 1
        if stress is not None:
            score -= (stress / 100.0) * 0.5
            count += 1
        if sleep_quality is not None:
            score += (sleep_quality - 0.5) * 0.3
            count += 1
        if activity is not None:
            score += min(activity / 10.0, 0.2)
            count += 1
        return max(-1.0, min(1.0, score / max(count, 1)))

    def _compute_arousal(self, signals: Dict[str, float]) -> float:
        """Map signals to arousal level [0, 1]."""
        hr = signals.get("heart_rate", 72)
        rr = signals.get("respiratory_rate", 16)
        activity = signals.get("activity_level", 0.0)
        hr_norm = min(1.0, max(0.0, (hr - 50) / 100))
        rr_norm = min(1.0, max(0.0, (rr - 10) / 20))
        act_norm = min(1.0, activity / 10.0)
        return (hr_norm * 0.4 + rr_norm * 0.3 + act_norm * 0.3)

    def _classify_mood(self, valence: float, arousal: float) -> tuple[MoodState, float]:
        """Return (MoodState, confidence)."""
        best_state = MoodState.NEUTRAL
        best_conf = 0.0
        for state, criteria in self.MOOD_CRITERIA.items():
            v_match = max(0.0, valence - criteria["min_valence"])
            ar_lo, ar_hi = criteria["arousal_range"]
            ar_match = 1.0 if ar_lo <= arousal <= ar_hi else max(0.0, 1 - min(
                abs(arousal - ar_lo), abs(arousal - ar_hi)
            ) * 2)
            conf = (v_match * 0.6 + ar_match * 0.4)
            if conf > best_conf:
                best_conf = conf
                best_state = state
        return best_state, min(1.0, best_conf)

    def update(self, signals: Dict[str, float]) -> MoodReading:
        """Compute and record current mood from biometric signals."""
        with self._lock:
            now = time.time()
            valence = self._compute_valence(signals)
            arousal = self._compute_arousal(signals)
            state, confidence = self._classify_mood(valence, arousal)

            # Update duration tracking
            if self._current:
                elapsed = now - self._last_update
                self._mood_duration[self._current.state] += elapsed

            reading = MoodReading(
                state=state,
                valence=MOOD_VALENCE_MAP[state],
                arousal=arousal,
                confidence=confidence,
                signals=signals.copy(),
                timestamp=now,
            )
            self._current = reading
            self._history.append(reading)
            self._last_update = now
            logger.debug("Mood: %s (confidence=%.2f, valence=%.2f)", state.value, confidence, valence)
            return reading

    def current_mood(self) -> Optional[MoodReading]:
        with self._lock:
            return self._current

    def mood_distribution(self, window: int = 100) -> Dict[str, float]:
        """Fraction of recent readings spent in each mood state."""
        with self._lock:
            readings = list(self._history)[-window:]
            if not readings:
                return {}
            counts: Dict[str, int] = {}
            for r in readings:
                counts[r.state.value] = counts.get(r.state.value, 0) + 1
            return {k: round(v / len(readings), 3) for k, v in counts.items()}

    def dominant_mood(self, window: int = 100) -> Optional[MoodState]:
        dist = self.mood_distribution(window)
        if not dist:
            return None
        best = max(dist, key=dist.get)
        return MoodState(best)


_MOOD_MONITOR: Optional["MoodMonitor"] = None
_MOOD_MONITOR_LOCK = threading.Lock()


def get_mood_monitor() -> MoodMonitor:
    global _MOOD_MONITOR
    with _MOOD_MONITOR_LOCK:
        if _MOOD_MONITOR is None:
            _MOOD_MONITOR = MoodMonitor()
        return _MOOD_MONITOR


def run_mood_monitor_tests() -> bool:
    logger.info("Running MoodMonitor tests...")
    monitor = MoodMonitor()

    # Happy/calm signals
    r = monitor.update({"heart_rate": 68.0, "hrv": 60.0, "stress_score": 10.0, "sleep_quality": 0.9, "activity_level": 3.0})
    assert r.state is not None
    assert r.valence == MoodValence.POSITIVE, f"Expected positive, got {r.valence}"

    # Stressed signals
    r2 = monitor.update({"heart_rate": 110.0, "hrv": 15.0, "stress_score": 80.0, "sleep_quality": 0.3})
    assert r2.valence == MoodValence.NEGATIVE, f"Expected negative, got {r2.valence}"

    dist = monitor.mood_distribution()
    assert sum(dist.values()) > 0

    logger.info("MoodMonitor tests passed")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = run_mood_monitor_tests()
    print("Tests passed:", ok)
    mm = get_mood_monitor()
    r = mm.update({"heart_rate": 72.0, "hrv": 50.0, "stress_score": 20.0, "sleep_quality": 0.75, "activity_level": 4.0})
    print(r.to_dict())
