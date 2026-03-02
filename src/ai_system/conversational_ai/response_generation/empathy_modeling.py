"""Empathy modeling module for the AI Holographic Wristwatch.

Determines the appropriate empathetic register for a response based on the
user's detected emotion, sentiment, and health context. Generates
acknowledgment phrases and modifies response text to reflect the chosen
empathy level.

Thread-safe singleton available via :func:`get_empathy_modeler`.
"""
from __future__ import annotations

import threading
import time
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EmpathyLevel(Enum):
    """Degree of empathetic expression to apply to a response."""
    NEUTRAL = "neutral"
    SUPPORTIVE = "supportive"
    CONCERNED = "concerned"
    VERY_CONCERNED = "very_concerned"
    URGENT = "urgent"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmpathyResponse:
    """Encapsulates the computed empathy recommendation for a given context."""
    level: EmpathyLevel
    acknowledgment: str
    response_modifier: str
    should_offer_help: bool
    empathy_score: float          # 0.0 = fully neutral, 1.0 = maximum empathy

    def __repr__(self) -> str:
        return (
            f"EmpathyResponse(level={self.level.value}, "
            f"score={self.empathy_score:.2f}, "
            f"offer_help={self.should_offer_help})"
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps emotion label → list of acknowledgment phrases.
_ACKNOWLEDGMENTS: Dict[str, List[str]] = {
    "happy": [
        "That's wonderful to hear!",
        "Great — you seem to be in good spirits!",
        "It's lovely to see you doing well.",
    ],
    "sad": [
        "I'm sorry to hear that.",
        "That sounds tough — I'm here with you.",
        "I understand this can be difficult.",
        "I'm sorry you're feeling this way.",
    ],
    "angry": [
        "I can understand your frustration.",
        "I hear you — that sounds really frustrating.",
        "I'm here to help work through this.",
    ],
    "anxious": [
        "It's okay to feel anxious sometimes.",
        "I'm right here — let's take this one step at a time.",
        "Take a breath — I'm with you.",
    ],
    "stressed": [
        "It sounds like you have a lot on your plate.",
        "I can see you're under pressure right now.",
        "Let's see what we can do to ease things a bit.",
    ],
    "tired": [
        "You sound exhausted — make sure to rest.",
        "Rest is important. Let me handle what I can.",
        "You deserve a break.",
    ],
    "confused": [
        "No worries — let me help clarify that.",
        "That's understandable — let me explain.",
        "I'll do my best to make that clearer.",
    ],
    "excited": [
        "I love the enthusiasm!",
        "That sounds exciting!",
        "Your energy is contagious!",
    ],
    "neutral": [
        "Understood.",
        "Got it.",
        "Of course.",
        "",
    ],
    "pain": [
        "I'm concerned — are you okay?",
        "I noticed something that worries me. Please take care.",
        "Your wellbeing is my priority — let's check on this.",
    ],
    "fear": [
        "There's no need to worry — I'm here.",
        "I've got you. Let's work through this together.",
        "Whatever it is, we can handle it.",
    ],
}

# Health metric thresholds that trigger heightened concern.
# Format: {metric: (low_critical, low_warn, high_warn, high_critical)}
_CONCERN_THRESHOLDS: Dict[str, tuple] = {
    "heart_rate":      (40,  55,  100, 140),   # bpm
    "spo2":            (88,  93,  100, 100),   # % (no high critical for SpO2)
    "systolic_bp":     (80,  90,  140, 180),   # mmHg
    "diastolic_bp":    (50,  60,   90, 110),   # mmHg
    "temperature":     (35.0, 36.0, 37.5, 38.5),  # °C
    "blood_glucose":   (3.0, 4.0,  10.0, 14.0),   # mmol/L
    "stress_index":    (0,    0,    70,   90),  # 0–100
    "respiratory_rate": (8,  12,   20,   25),  # breaths/min
    "steps_hourly_gap": (0,   0,  9999, 9999), # large gap = sedentary alert (unused)
}

# Modifier phrases prepended / appended for each empathy level.
_LEVEL_MODIFIERS: Dict[EmpathyLevel, str] = {
    EmpathyLevel.NEUTRAL:       "",
    EmpathyLevel.SUPPORTIVE:    "I'm here for you. ",
    EmpathyLevel.CONCERNED:     "I want to make sure you're okay. ",
    EmpathyLevel.VERY_CONCERNED: "I'm genuinely concerned about you. ",
    EmpathyLevel.URGENT:        "This needs your immediate attention. ",
}

# Offer-help thresholds: levels at or above which we offer assistance.
_OFFER_HELP_LEVELS = {EmpathyLevel.SUPPORTIVE, EmpathyLevel.CONCERNED,
                      EmpathyLevel.VERY_CONCERNED, EmpathyLevel.URGENT}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EmpathyModeler:
    """Computes the appropriate empathy level and modifies response text.

    Thread-safe; a shared singleton is available via
    :func:`get_empathy_modeler`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model_count: int = 0
        self._start_time: float = time.monotonic()
        logger.debug("EmpathyModeler initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def model_empathy(
        self,
        sentiment_valence: float,
        emotion: str,
        health_context: Optional[Dict] = None,
    ) -> EmpathyResponse:
        """Determine the empathy level and craft acknowledgment text.

        Args:
            sentiment_valence: Floating-point sentiment score in [-1.0, 1.0]
                where -1 is very negative, 0 is neutral, +1 is very positive.
            emotion: Detected emotion label (e.g. ``"sad"``, ``"anxious"``).
            health_context: Optional dict of current health metrics
                (e.g. ``{"heart_rate": 45, "spo2": 91}``).

        Returns:
            An :class:`EmpathyResponse` with the recommended level,
            acknowledgment text, response modifier, and other metadata.
        """
        with self._lock:
            self._model_count += 1

        health_context = health_context or {}
        emotion_lower = emotion.lower().strip()

        # Determine base empathy score from sentiment valence.
        # Negative sentiment → higher empathy score.
        valence_clamped = max(-1.0, min(1.0, sentiment_valence))
        base_score = (1.0 - valence_clamped) / 2.0  # maps [-1,1] → [1.0, 0.0]

        # Emotion-based adjustment.
        emotion_boost = self._emotion_boost(emotion_lower)
        empathy_score = min(1.0, base_score + emotion_boost)

        # Health context can escalate the level independently.
        health_concern = self._health_concern_level(health_context)

        # Map empathy_score → EmpathyLevel.
        if health_concern == EmpathyLevel.URGENT or empathy_score >= 0.9:
            level = EmpathyLevel.URGENT
        elif health_concern == EmpathyLevel.VERY_CONCERNED or empathy_score >= 0.70:
            level = EmpathyLevel.VERY_CONCERNED
        elif health_concern == EmpathyLevel.CONCERNED or empathy_score >= 0.45:
            level = EmpathyLevel.CONCERNED
        elif empathy_score >= 0.20:
            level = EmpathyLevel.SUPPORTIVE
        else:
            level = EmpathyLevel.NEUTRAL

        acknowledgment = self.generate_empathetic_acknowledgment(emotion_lower)
        modifier = _LEVEL_MODIFIERS[level]
        should_offer = level in _OFFER_HELP_LEVELS

        logger.debug(
            "model_empathy: valence=%.2f emotion=%r score=%.2f level=%s",
            sentiment_valence,
            emotion,
            empathy_score,
            level.value,
        )

        return EmpathyResponse(
            level=level,
            acknowledgment=acknowledgment,
            response_modifier=modifier,
            should_offer_help=should_offer,
            empathy_score=round(empathy_score, 3),
        )

    def generate_empathetic_acknowledgment(self, emotion: str) -> str:
        """Return a context-appropriate acknowledgment phrase for *emotion*."""
        emotion = emotion.lower().strip()
        phrases = _ACKNOWLEDGMENTS.get(emotion, _ACKNOWLEDGMENTS["neutral"])
        chosen = random.choice(phrases)
        return chosen

    def should_express_concern(self, health_metrics: Dict) -> bool:
        """Return ``True`` if any metric in *health_metrics* is outside the safe range.

        Uses the warning (not critical) thresholds for a cautious approach.
        """
        level = self._health_concern_level(health_metrics)
        return level != EmpathyLevel.NEUTRAL

    def apply_empathy(self, text: str, empathy_response: EmpathyResponse) -> str:
        """Prepend the acknowledgment and modifier to *text*.

        If the empathy level is NEUTRAL and there is no acknowledgment, the
        text is returned unchanged.

        Args:
            text: The raw or partially-processed response string.
            empathy_response: The :class:`EmpathyResponse` from
                :meth:`model_empathy`.

        Returns:
            The empathy-augmented response string.
        """
        parts: List[str] = []

        ack = empathy_response.acknowledgment.strip()
        mod = empathy_response.response_modifier.strip()

        if ack:
            parts.append(ack)
        if mod:
            parts.append(mod)

        if parts:
            prefix = " ".join(parts) + " "
            # Lower-case the first character of the original text when joining.
            if text:
                text = prefix + text[0].lower() + text[1:]
            else:
                text = prefix.strip()

        if empathy_response.should_offer_help:
            if not text.endswith((".", "!", "?")):
                text += "."
            if empathy_response.level == EmpathyLevel.URGENT:
                text += " Please seek assistance if needed."
            else:
                text += " I'm here if you need anything."

        return text.strip()

    def get_stats(self) -> Dict:
        """Return runtime statistics."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            return {
                "model_count": self._model_count,
                "uptime_seconds": round(elapsed, 2),
                "known_emotions": list(_ACKNOWLEDGMENTS.keys()),
                "monitored_metrics": list(_CONCERN_THRESHOLDS.keys()),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emotion_boost(self, emotion: str) -> float:
        """Return an additive empathy boost [0.0, 0.5] based on emotion label."""
        boosts = {
            "pain":    0.50,
            "fear":    0.45,
            "sad":     0.40,
            "angry":   0.30,
            "anxious": 0.35,
            "stressed": 0.30,
            "tired":   0.20,
            "confused": 0.10,
            "neutral": 0.00,
            "happy":   0.00,
            "excited": 0.00,
        }
        return boosts.get(emotion, 0.15)

    def _health_concern_level(self, metrics: Dict) -> EmpathyLevel:
        """Return the most severe concern level implied by *metrics*."""
        worst = EmpathyLevel.NEUTRAL
        severity_order = [
            EmpathyLevel.NEUTRAL,
            EmpathyLevel.SUPPORTIVE,
            EmpathyLevel.CONCERNED,
            EmpathyLevel.VERY_CONCERNED,
            EmpathyLevel.URGENT,
        ]

        for metric, value in metrics.items():
            thresholds = _CONCERN_THRESHOLDS.get(metric)
            if thresholds is None or value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            lo_crit, lo_warn, hi_warn, hi_crit = thresholds

            if v <= lo_crit or v >= hi_crit:
                candidate = EmpathyLevel.URGENT
            elif v <= lo_warn or v >= hi_warn:
                candidate = EmpathyLevel.VERY_CONCERNED
            else:
                candidate = EmpathyLevel.NEUTRAL

            if severity_order.index(candidate) > severity_order.index(worst):
                worst = candidate

        return worst


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_modeler_instance: Optional[EmpathyModeler] = None
_modeler_lock = threading.Lock()


def get_empathy_modeler() -> EmpathyModeler:
    """Return the module-level :class:`EmpathyModeler` singleton.

    Thread-safe; the instance is created lazily on first call.
    """
    global _modeler_instance
    if _modeler_instance is None:
        with _modeler_lock:
            if _modeler_instance is None:
                _modeler_instance = EmpathyModeler()
    return _modeler_instance


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    modeler = get_empathy_modeler()

    scenarios = [
        # (valence, emotion, health_context)
        (0.8,   "happy",   {}),
        (-0.3,  "stressed", {"heart_rate": 105}),
        (-0.7,  "sad",      {}),
        (-0.9,  "pain",     {"heart_rate": 38, "spo2": 89}),
        (0.1,   "neutral",  {"stress_index": 85}),
        (0.0,   "anxious",  {}),
    ]

    base_response = "Your heart rate is 38 bpm."

    print("=== EmpathyModeler Demo ===\n")
    for valence, emotion, health in scenarios:
        er = modeler.model_empathy(valence, emotion, health)
        modified = modeler.apply_empathy(base_response, er)
        print(f"Emotion={emotion:<10} valence={valence:+.1f}  health={health}")
        print(f"  Level     : {er.level.value}")
        print(f"  Score     : {er.empathy_score:.3f}")
        print(f"  Offer help: {er.should_offer_help}")
        print(f"  Modified  : {modified!r}")
        print()

    print("Stats:", modeler.get_stats())
