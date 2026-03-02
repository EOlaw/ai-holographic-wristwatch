"""Voice personalisation module for the AI Holographic Wristwatch.

Learns individual user voice preferences from explicit feedback, adapts
synthesis parameters to environmental conditions (noise, time of day),
and can run simple A/B variant tests to discover which voice settings
a user prefers.

Thread-safe singleton available via :func:`get_voice_personalizer`.
"""
from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------

@dataclass
class _FeedbackRecord:
    """A single feedback event recorded for a user."""
    user_id: str
    timestamp: float
    param_key: str      # e.g. "rate", "pitch", "volume"
    old_value: float
    new_value: float
    rating: float       # User satisfaction 1–5 (or 0 if implicit)
    source: str         # "explicit", "implicit", "ab_test"


@dataclass
class _UserPreferences:
    """Persisted preference state for a single user."""
    user_id: str
    preferred_rate: float = 1.0
    preferred_pitch: float = 0.0
    preferred_volume: float = 1.0
    preferred_emphasis: float = 0.5
    preferred_voice_id: str = "voice_aria"
    feedback_count: int = 0
    ab_test_history: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.monotonic)

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "preferred_rate": self.preferred_rate,
            "preferred_pitch": self.preferred_pitch,
            "preferred_volume": self.preferred_volume,
            "preferred_emphasis": self.preferred_emphasis,
            "preferred_voice_id": self.preferred_voice_id,
            "feedback_count": self.feedback_count,
            "last_updated": round(self.last_updated, 2),
        }


# ---------------------------------------------------------------------------
# Environmental adaptation constants
# ---------------------------------------------------------------------------

# Noise level (dB SPL) → volume multiplier adjustments.
_NOISE_VOLUME_CURVE: List[Tuple[float, float]] = [
    (0.0,  0.70),   # Silent environment — quieter voice
    (30.0, 0.85),   # Library / quiet office
    (50.0, 1.00),   # Normal office
    (65.0, 1.20),   # Busy office / café
    (80.0, 1.45),   # Street / gym
    (90.0, 1.70),   # Very noisy environment
]

# Context string → rate adjustment (quieter / slower at night, etc.)
_CONTEXT_RATE_ADJUSTMENTS: Dict[str, float] = {
    "night":      -0.12,
    "sleep":      -0.18,
    "morning":    +0.00,
    "commute":    +0.05,
    "workout":    +0.10,
    "driving":    +0.08,
    "meeting":    -0.08,
    "focus":      -0.05,
    "relaxing":   -0.10,
}

# Minimum feedback events before personalisation is trusted.
_MIN_FEEDBACK_FOR_TRUST = 3

# Learning rate for exponential moving average of preferred values.
_LEARNING_RATE = 0.25


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VoicePersonalizer:
    """Learns and applies per-user voice synthesis preferences.

    Thread-safe; a shared singleton is available via
    :func:`get_voice_personalizer`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._preferences: Dict[str, _UserPreferences] = {}
        self._feedback_history: Deque[_FeedbackRecord] = deque(maxlen=100)
        self._start_time: float = time.monotonic()
        logger.debug("VoicePersonalizer initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn_preferences(self, user_id: str, feedback: Dict) -> None:
        """Update voice preferences for *user_id* based on *feedback*.

        The feedback dict may contain any subset of the adjustable parameters
        along with a ``"rating"`` key (float, 1–5).

        Supported feedback keys:
          ``"rate"``, ``"pitch"``, ``"volume"``, ``"emphasis"``,
          ``"voice_id"``, ``"rating"``.

        Updates are applied as an exponential moving average so that recent
        feedback has more influence than older observations.

        Args:
            user_id: Unique identifier for the user.
            feedback: Dict of parameter names to their newly preferred values.
        """
        with self._lock:
            prefs = self._get_or_create_prefs(user_id)
            rating = float(feedback.get("rating", 3.0))

            # Adaptive learning rate: trust increases with more feedback.
            effective_lr = _LEARNING_RATE if prefs.feedback_count >= _MIN_FEEDBACK_FOR_TRUST \
                else _LEARNING_RATE * 1.5

            if "rate" in feedback:
                new_val = max(0.5, min(2.0, float(feedback["rate"])))
                old = prefs.preferred_rate
                prefs.preferred_rate = old + effective_lr * (new_val - old)
                self._record_feedback(user_id, "rate", old, prefs.preferred_rate, rating)

            if "pitch" in feedback:
                new_val = max(-12.0, min(12.0, float(feedback["pitch"])))
                old = prefs.preferred_pitch
                prefs.preferred_pitch = old + effective_lr * (new_val - old)
                self._record_feedback(user_id, "pitch", old, prefs.preferred_pitch, rating)

            if "volume" in feedback:
                new_val = max(0.0, min(2.0, float(feedback["volume"])))
                old = prefs.preferred_volume
                prefs.preferred_volume = old + effective_lr * (new_val - old)
                self._record_feedback(user_id, "volume", old, prefs.preferred_volume, rating)

            if "emphasis" in feedback:
                new_val = max(0.0, min(1.0, float(feedback["emphasis"])))
                old = prefs.preferred_emphasis
                prefs.preferred_emphasis = old + effective_lr * (new_val - old)
                self._record_feedback(user_id, "emphasis", old, prefs.preferred_emphasis, rating)

            if "voice_id" in feedback:
                prefs.preferred_voice_id = str(feedback["voice_id"])

            prefs.feedback_count += 1
            prefs.last_updated = time.monotonic()

        logger.debug(
            "learn_preferences: user=%r feedback_count=%d",
            user_id,
            prefs.feedback_count,
        )

    def get_personalized_params(self, user_id: str) -> Dict:
        """Return the current personalised voice parameters for *user_id*.

        If the user has insufficient feedback history, returns sensible
        defaults with a flag indicating the personalisation confidence is low.

        Returns:
            A dict with keys: ``"rate"``, ``"pitch"``, ``"volume"``,
            ``"emphasis"``, ``"voice_id"``, ``"confidence"``.
        """
        with self._lock:
            prefs = self._get_or_create_prefs(user_id)
            confidence = min(
                1.0,
                prefs.feedback_count / max(1, _MIN_FEEDBACK_FOR_TRUST * 2),
            )
            return {
                "rate":     round(prefs.preferred_rate, 4),
                "pitch":    round(prefs.preferred_pitch, 4),
                "volume":   round(prefs.preferred_volume, 4),
                "emphasis": round(prefs.preferred_emphasis, 4),
                "voice_id": prefs.preferred_voice_id,
                "confidence": round(confidence, 3),
            }

    def adapt_to_environment(
        self,
        noise_level_db: float,
        context: str,
    ) -> Dict:
        """Compute environment-adaptive voice parameter adjustments.

        Raises the volume in noisy conditions and slows the rate at night or
        in quiet contexts (driving, meeting).

        Args:
            noise_level_db: Measured ambient noise level in dB SPL.
            context: Contextual label such as ``"night"``, ``"workout"``,
                ``"commute"``, etc.

        Returns:
            A dict of parameter adjustments: ``"volume_multiplier"``,
            ``"rate_delta"``, ``"emphasis_boost"``.
        """
        # Volume from noise curve (linear interpolation).
        volume_mult = self._interpolate_noise_volume(noise_level_db)

        # Rate delta from context map.
        context_lower = context.lower().strip()
        rate_delta = _CONTEXT_RATE_ADJUSTMENTS.get(context_lower, 0.0)

        # Emphasis boost: louder environments need more stress to be intelligible.
        emphasis_boost = min(0.3, (noise_level_db - 50.0) / 100.0) if noise_level_db > 50 else 0.0

        result = {
            "volume_multiplier": round(volume_mult, 3),
            "rate_delta": round(rate_delta, 3),
            "emphasis_boost": round(max(0.0, emphasis_boost), 3),
            "noise_level_db": noise_level_db,
            "context": context,
        }
        logger.debug("adapt_to_environment: %s", result)
        return result

    def run_ab_test(
        self,
        user_id: str,
        variant_a: Dict,
        variant_b: Dict,
    ) -> str:
        """Simulate an A/B test for voice parameters and return the preferred variant.

        In a real deployment this would present both variants to the user and
        wait for implicit or explicit feedback. Here we use the user's
        existing preference vector to predict which variant is closer.

        Args:
            user_id: The user being tested.
            variant_a: A dict of synthesis params for variant A.
            variant_b: A dict of synthesis params for variant B.

        Returns:
            ``"a"`` or ``"b"`` indicating the preferred variant.
        """
        with self._lock:
            prefs = self._get_or_create_prefs(user_id)

        def _distance(variant: Dict) -> float:
            """Manhattan distance between variant and user's preferred params."""
            dist = 0.0
            dist += abs(variant.get("rate", 1.0) - prefs.preferred_rate)
            dist += abs(variant.get("pitch", 0.0) - prefs.preferred_pitch) / 12.0
            dist += abs(variant.get("volume", 1.0) - prefs.preferred_volume)
            dist += abs(variant.get("emphasis", 0.5) - prefs.preferred_emphasis)
            return dist

        dist_a = _distance(variant_a)
        dist_b = _distance(variant_b)

        # Add a small random tiebreaker to avoid deterministic ties.
        if abs(dist_a - dist_b) < 0.01:
            winner = random.choice(["a", "b"])
        else:
            winner = "a" if dist_a <= dist_b else "b"

        with self._lock:
            prefs.ab_test_history.append(winner)

        logger.debug(
            "run_ab_test: user=%r dist_a=%.4f dist_b=%.4f winner=%s",
            user_id,
            dist_a,
            dist_b,
            winner,
        )
        return winner

    def get_user_preference_summary(self, user_id: str) -> Dict:
        """Return a human-readable summary of *user_id*'s preference profile."""
        with self._lock:
            prefs = self._get_or_create_prefs(user_id)
            # Recent feedback for this user.
            recent = [
                r for r in self._feedback_history if r.user_id == user_id
            ][-5:]  # last 5 events
            return {
                "preferences": prefs.to_dict(),
                "recent_feedback_count": len(recent),
                "ab_test_history": list(prefs.ab_test_history[-10:]),
                "trusted": prefs.feedback_count >= _MIN_FEEDBACK_FOR_TRUST,
            }

    def get_stats(self) -> Dict:
        """Return runtime statistics."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            return {
                "user_count": len(self._preferences),
                "total_feedback_events": len(self._feedback_history),
                "uptime_seconds": round(elapsed, 2),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_prefs(self, user_id: str) -> _UserPreferences:
        """Return existing prefs or create defaults (must be called under lock)."""
        if user_id not in self._preferences:
            self._preferences[user_id] = _UserPreferences(user_id=user_id)
            logger.debug("Created new preference profile for user %r.", user_id)
        return self._preferences[user_id]

    def _record_feedback(
        self,
        user_id: str,
        param_key: str,
        old_value: float,
        new_value: float,
        rating: float,
        source: str = "explicit",
    ) -> None:
        """Append a feedback record to the history deque (must be called under lock)."""
        self._feedback_history.append(
            _FeedbackRecord(
                user_id=user_id,
                timestamp=time.monotonic(),
                param_key=param_key,
                old_value=old_value,
                new_value=new_value,
                rating=rating,
                source=source,
            )
        )

    def _interpolate_noise_volume(self, noise_db: float) -> float:
        """Linearly interpolate the volume multiplier from the noise curve."""
        noise_db = max(0.0, min(100.0, noise_db))
        curve = _NOISE_VOLUME_CURVE
        # Below the lowest point.
        if noise_db <= curve[0][0]:
            return curve[0][1]
        # Above the highest point.
        if noise_db >= curve[-1][0]:
            return curve[-1][1]
        # Find bracketing segment.
        for i in range(len(curve) - 1):
            lo_db, lo_vol = curve[i]
            hi_db, hi_vol = curve[i + 1]
            if lo_db <= noise_db <= hi_db:
                t = (noise_db - lo_db) / (hi_db - lo_db)
                return lo_vol + t * (hi_vol - lo_vol)
        return 1.0  # fallback


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_personalizer_instance: Optional[VoicePersonalizer] = None
_personalizer_lock = threading.Lock()


def get_voice_personalizer() -> VoicePersonalizer:
    """Return the module-level :class:`VoicePersonalizer` singleton.

    Thread-safe; the instance is created lazily on first call.
    """
    global _personalizer_instance
    if _personalizer_instance is None:
        with _personalizer_lock:
            if _personalizer_instance is None:
                _personalizer_instance = VoicePersonalizer()
    return _personalizer_instance


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    vp = get_voice_personalizer()

    user = "user_42"

    # Teach preferences gradually.
    feedback_events = [
        {"rate": 1.15, "volume": 1.1,  "rating": 4},
        {"rate": 1.10, "pitch": 1.0,   "rating": 5},
        {"rate": 1.12, "emphasis": 0.6, "volume": 1.05, "rating": 4},
        {"voice_id": "voice_nova",     "rating": 5},
    ]

    print("=== VoicePersonalizer Demo ===\n")
    for i, fb in enumerate(feedback_events, 1):
        vp.learn_preferences(user, fb)
        params = vp.get_personalized_params(user)
        print(f"After feedback #{i}: {params}")

    # Environmental adaptation.
    print("\nEnvironmental adaptation:")
    envs = [
        (20.0,  "night"),
        (55.0,  "normal"),
        (75.0,  "commute"),
        (85.0,  "workout"),
    ]
    for noise, ctx in envs:
        adj = vp.adapt_to_environment(noise, ctx)
        print(f"  noise={noise:5.1f}dB  ctx={ctx:<10}  {adj}")

    # A/B test.
    variant_a = {"rate": 1.0, "pitch": 0.0, "volume": 1.0, "emphasis": 0.5}
    variant_b = {"rate": 1.1, "pitch": 1.0, "volume": 1.1, "emphasis": 0.6}
    winner = vp.run_ab_test(user, variant_a, variant_b)
    print(f"\nA/B test winner for {user}: variant {winner.upper()}")

    print("\nPreference summary:", vp.get_user_preference_summary(user))
    print("\nStats:", vp.get_stats())
