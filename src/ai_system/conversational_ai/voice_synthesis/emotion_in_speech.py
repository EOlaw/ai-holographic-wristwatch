"""Emotion-in-speech module for the AI Holographic Wristwatch.

Maps discrete emotional states to low-level voice parameter adjustments.
Supports blending multiple emotional states and applying intensity scaling.

Thread-safe singleton available via :func:`get_emotional_speech`.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmotionProfile:
    """Voice parameter deltas and modifiers for a single emotional state.

    All numeric fields are *deltas* relative to neutral baseline unless
    explicitly noted.
    """
    pitch_shift: float      # Semitone offset from neutral (positive = higher)
    rate_factor: float      # Speaking rate multiplier (1.0 = no change)
    breathiness: float      # Amount of breathy noise added [0.0, 1.0]
    energy: float           # Overall signal energy / loudness factor [0.0, 2.0]
    tremor: float           # Amplitude tremor amount [0.0, 1.0]

    def __repr__(self) -> str:
        return (
            f"EmotionProfile(pitch_shift={self.pitch_shift:+.1f}st, "
            f"rate={self.rate_factor:.2f}x, "
            f"breathiness={self.breathiness:.2f}, "
            f"energy={self.energy:.2f}, "
            f"tremor={self.tremor:.2f})"
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "pitch_shift": self.pitch_shift,
            "rate_factor": self.rate_factor,
            "breathiness": self.breathiness,
            "energy": self.energy,
            "tremor": self.tremor,
        }


# ---------------------------------------------------------------------------
# Emotion profile constants
# ---------------------------------------------------------------------------

# Neutral reference point — all values represent deltas from this baseline.
_NEUTRAL = EmotionProfile(
    pitch_shift=0.0,
    rate_factor=1.0,
    breathiness=0.05,
    energy=1.0,
    tremor=0.0,
)

_EMOTION_PROFILES: Dict[str, EmotionProfile] = {
    "neutral": EmotionProfile(
        pitch_shift=0.0,
        rate_factor=1.00,
        breathiness=0.05,
        energy=1.00,
        tremor=0.00,
    ),
    "joy": EmotionProfile(
        pitch_shift=+2.5,
        rate_factor=1.12,
        breathiness=0.10,
        energy=1.15,
        tremor=0.00,
    ),
    "sadness": EmotionProfile(
        pitch_shift=-2.0,
        rate_factor=0.82,
        breathiness=0.30,
        energy=0.75,
        tremor=0.05,
    ),
    "anger": EmotionProfile(
        pitch_shift=+1.5,
        rate_factor=1.08,
        breathiness=0.05,
        energy=1.35,
        tremor=0.10,
    ),
    "fear": EmotionProfile(
        pitch_shift=+3.0,
        rate_factor=1.18,
        breathiness=0.40,
        energy=0.88,
        tremor=0.25,
    ),
    "excitement": EmotionProfile(
        pitch_shift=+4.0,
        rate_factor=1.22,
        breathiness=0.12,
        energy=1.25,
        tremor=0.00,
    ),
    "calm": EmotionProfile(
        pitch_shift=-1.0,
        rate_factor=0.88,
        breathiness=0.08,
        energy=0.90,
        tremor=0.00,
    ),
    "concerned": EmotionProfile(
        pitch_shift=-0.5,
        rate_factor=0.92,
        breathiness=0.12,
        energy=0.95,
        tremor=0.02,
    ),
    "surprise": EmotionProfile(
        pitch_shift=+3.5,
        rate_factor=1.10,
        breathiness=0.15,
        energy=1.10,
        tremor=0.02,
    ),
    "disgust": EmotionProfile(
        pitch_shift=-1.5,
        rate_factor=0.95,
        breathiness=0.20,
        energy=0.95,
        tremor=0.05,
    ),
    "empathetic": EmotionProfile(
        pitch_shift=-0.5,
        rate_factor=0.90,
        breathiness=0.18,
        energy=0.92,
        tremor=0.00,
    ),
    "urgent": EmotionProfile(
        pitch_shift=+2.0,
        rate_factor=1.20,
        breathiness=0.05,
        energy=1.30,
        tremor=0.00,
    ),
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EmotionalSpeech:
    """Applies emotional colouring to voice synthesis parameters.

    Thread-safe; a shared singleton is available via
    :func:`get_emotional_speech`.
    """

    _EMOTION_PROFILES: Dict[str, EmotionProfile] = _EMOTION_PROFILES

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._apply_count: int = 0
        self._start_time: float = time.monotonic()
        logger.debug("EmotionalSpeech initialised with %d profiles.",
                     len(self._EMOTION_PROFILES))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_emotion(
        self,
        base_params: Dict,
        emotion: str,
        intensity: float = 1.0,
    ) -> Dict:
        """Apply emotional voice parameters on top of *base_params*.

        Args:
            base_params: A dict of baseline voice parameters. Expected keys
                are a subset of ``{"pitch", "rate", "volume", "breathiness",
                "tremor"}``. Unknown keys are passed through unchanged.
            emotion: The target emotion label (e.g. ``"joy"``, ``"sadness"``).
                Falls back to ``"neutral"`` if unknown.
            intensity: How strongly to apply the emotional offset. 0.0 = no
                change, 1.0 = full emotional profile. Clamped to [0.0, 1.0].

        Returns:
            A new dict with the same keys as *base_params* but with emotional
            deltas blended in.
        """
        intensity = max(0.0, min(1.0, intensity))

        with self._lock:
            self._apply_count += 1

        profile = self.map_emotion_to_voice_params(emotion)

        # Blend: result = base + intensity * delta
        result = dict(base_params)

        # pitch (semitones)
        result["pitch"] = (
            float(result.get("pitch", 0.0)) + intensity * profile.pitch_shift
        )
        # rate multiplier
        base_rate = float(result.get("rate", 1.0))
        emotion_rate_delta = profile.rate_factor - 1.0
        result["rate"] = base_rate + intensity * emotion_rate_delta

        # volume (energy proxy)
        base_volume = float(result.get("volume", 1.0))
        energy_delta = profile.energy - 1.0
        result["volume"] = max(0.0, min(2.0, base_volume + intensity * energy_delta))

        # breathiness
        result["breathiness"] = max(
            0.0,
            min(1.0, float(result.get("breathiness", 0.05))
                + intensity * (profile.breathiness - 0.05)),
        )

        # tremor
        result["tremor"] = max(
            0.0,
            min(1.0, float(result.get("tremor", 0.0)) + intensity * profile.tremor),
        )

        logger.debug(
            "apply_emotion: emotion=%r intensity=%.2f -> pitch=%+.1f rate=%.2f",
            emotion,
            intensity,
            result["pitch"],
            result["rate"],
        )
        return result

    def map_emotion_to_voice_params(self, emotion: str) -> EmotionProfile:
        """Return the :class:`EmotionProfile` for *emotion*.

        Falls back to ``"neutral"`` for unknown labels.
        """
        key = emotion.lower().strip()
        profile = self._EMOTION_PROFILES.get(key)
        if profile is None:
            logger.warning("Unknown emotion %r; using neutral profile.", emotion)
            profile = self._EMOTION_PROFILES["neutral"]
        return profile

    def blend_emotional_states(
        self, emotions: Dict[str, float]
    ) -> EmotionProfile:
        """Compute a weighted blend of multiple emotional states.

        Args:
            emotions: A mapping of emotion label → weight (unnormalised).
                e.g. ``{"joy": 0.7, "excited": 0.3}``.

        Returns:
            A blended :class:`EmotionProfile`.

        Raises:
            ValueError: If *emotions* is empty or all weights are zero.
        """
        if not emotions:
            raise ValueError("emotions dict must not be empty.")

        total_weight = sum(emotions.values())
        if total_weight <= 0.0:
            raise ValueError("Sum of emotion weights must be > 0.")

        # Weighted average over each parameter.
        pitch_shift = 0.0
        rate_factor = 0.0
        breathiness = 0.0
        energy = 0.0
        tremor = 0.0

        for label, weight in emotions.items():
            p = self.map_emotion_to_voice_params(label)
            norm_w = weight / total_weight
            pitch_shift += norm_w * p.pitch_shift
            rate_factor += norm_w * p.rate_factor
            breathiness += norm_w * p.breathiness
            energy += norm_w * p.energy
            tremor += norm_w * p.tremor

        blended = EmotionProfile(
            pitch_shift=round(pitch_shift, 3),
            rate_factor=round(rate_factor, 4),
            breathiness=round(max(0.0, min(1.0, breathiness)), 3),
            energy=round(max(0.0, min(2.0, energy)), 3),
            tremor=round(max(0.0, min(1.0, tremor)), 3),
        )
        logger.debug("blend_emotional_states(%r) -> %s", emotions, blended)
        return blended

    def get_available_emotions(self) -> List[str]:
        """Return the list of supported emotion labels."""
        return sorted(self._EMOTION_PROFILES.keys())

    def get_stats(self) -> Dict:
        """Return runtime statistics."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            return {
                "apply_count": self._apply_count,
                "uptime_seconds": round(elapsed, 2),
                "emotion_profiles": len(self._EMOTION_PROFILES),
                "available_emotions": self.get_available_emotions(),
            }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_emotional_speech_instance: Optional[EmotionalSpeech] = None
_emotional_speech_lock = threading.Lock()


def get_emotional_speech() -> EmotionalSpeech:
    """Return the module-level :class:`EmotionalSpeech` singleton.

    Thread-safe; the instance is created lazily on first call.
    """
    global _emotional_speech_instance
    if _emotional_speech_instance is None:
        with _emotional_speech_lock:
            if _emotional_speech_instance is None:
                _emotional_speech_instance = EmotionalSpeech()
    return _emotional_speech_instance


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    es = get_emotional_speech()

    print("=== EmotionalSpeech Demo ===\n")
    print("Available emotions:", es.get_available_emotions())
    print()

    base = {"pitch": 0.0, "rate": 1.0, "volume": 1.0, "breathiness": 0.05, "tremor": 0.0}
    for emotion in ["joy", "sadness", "anger", "fear", "excitement", "calm", "urgent"]:
        result = es.apply_emotion(dict(base), emotion, intensity=1.0)
        profile = es.map_emotion_to_voice_params(emotion)
        print(f"Emotion={emotion:<12} profile={profile}")
        print(f"           applied={result}")
        print()

    # Blending.
    blend = es.blend_emotional_states({"joy": 0.6, "excitement": 0.3, "calm": 0.1})
    print("Blended(joy=0.6, excitement=0.3, calm=0.1):", blend)

    print("\nStats:", es.get_stats())
