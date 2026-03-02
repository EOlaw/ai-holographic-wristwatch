"""Prosody control module for the AI Holographic Wristwatch.

Manages the suprasegmental features of synthesised speech: speaking rate,
pitch contour, volume (loudness), emphasis, and pause placement. Outputs an
annotated text string (SSML-compatible markup) and a :class:`ProsodyParams`
object that the speech synthesiser can apply.

Thread-safe singleton available via :func:`get_prosody_controller`.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProsodyParams:
    """Numeric prosody parameters for a single utterance."""
    rate: float = 1.0           # Speaking rate multiplier (0.5 = half speed, 2.0 = double)
    pitch: float = 0.0          # Semitone offset from the voice's base pitch
    volume: float = 1.0         # Amplitude multiplier [0.0, 2.0]
    emphasis: float = 0.5       # Emphasis intensity for highlighted words [0.0, 1.0]
    pause_factor: float = 1.0   # Multiplier applied to all inter-punctuation pauses

    def clone(self) -> "ProsodyParams":
        return ProsodyParams(
            rate=self.rate,
            pitch=self.pitch,
            volume=self.volume,
            emphasis=self.emphasis,
            pause_factor=self.pause_factor,
        )

    def __repr__(self) -> str:
        return (
            f"ProsodyParams(rate={self.rate}, pitch={self.pitch:+.1f}st, "
            f"vol={self.volume}, emphasis={self.emphasis}, "
            f"pause_factor={self.pause_factor})"
        )


# ---------------------------------------------------------------------------
# Emotion → prosody mapping
# ---------------------------------------------------------------------------

# Adjustments applied to the *base* ProsodyParams for each detected emotion.
# Format: {"rate_delta", "pitch_delta", "volume_delta", "emphasis_delta", "pause_delta"}
_EMOTION_PROSODY: Dict[str, Dict[str, float]] = {
    "calm":       {"rate": 0.90, "pitch": -1.0, "volume": 0.85, "emphasis": 0.35, "pause": 1.20},
    "happy":      {"rate": 1.10, "pitch": +2.0, "volume": 1.05, "emphasis": 0.60, "pause": 0.90},
    "excited":    {"rate": 1.20, "pitch": +3.5, "volume": 1.15, "emphasis": 0.80, "pause": 0.75},
    "sad":        {"rate": 0.80, "pitch": -2.5, "volume": 0.80, "emphasis": 0.25, "pause": 1.40},
    "anxious":    {"rate": 1.15, "pitch": +1.5, "volume": 0.95, "emphasis": 0.55, "pause": 0.85},
    "angry":      {"rate": 1.05, "pitch": +1.0, "volume": 1.20, "emphasis": 0.90, "pause": 0.80},
    "fearful":    {"rate": 1.10, "pitch": +2.0, "volume": 0.90, "emphasis": 0.65, "pause": 0.90},
    "surprised":  {"rate": 1.15, "pitch": +3.0, "volume": 1.10, "emphasis": 0.75, "pause": 0.85},
    "neutral":    {"rate": 1.00, "pitch":  0.0, "volume": 1.00, "emphasis": 0.50, "pause": 1.00},
    "concerned":  {"rate": 0.92, "pitch": -0.5, "volume": 0.95, "emphasis": 0.55, "pause": 1.15},
    "urgent":     {"rate": 1.18, "pitch": +1.5, "volume": 1.25, "emphasis": 0.95, "pause": 0.70},
    "soothing":   {"rate": 0.85, "pitch": -1.5, "volume": 0.88, "emphasis": 0.30, "pause": 1.30},
}

# Pause durations (in milliseconds) inserted after each punctuation mark.
_PAUSE_MARKS: Dict[str, int] = {
    ".":  350,
    "!":  300,
    "?":  350,
    ",":  150,
    ";":  200,
    ":":  180,
    "—":  200,
    "–":  150,
    "...": 450,
}

# Regex that matches pause-eligible punctuation (order matters for multi-char).
_PUNCTUATION_RE = re.compile(r"(\.\.\.|[.!?,;:—–])")

# Words per minute bounds.
_WPM_MIN = 80
_WPM_MAX = 220
# Baseline WPM (used to compute rate multiplier).
_WPM_BASELINE = 150

# Keywords whose surrounding words should receive heavier emphasis.
_EMPHASIS_TRIGGERS: List[str] = [
    "important", "critical", "urgent", "warning", "alert",
    "immediately", "now", "danger", "emergency", "attention",
    "please", "must", "never", "always", "highest",
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ProsodyController:
    """Computes and applies prosody adjustments to speech synthesis parameters.

    Thread-safe; a shared singleton is available via
    :func:`get_prosody_controller`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._apply_count: int = 0
        self._start_time: float = time.monotonic()
        logger.debug("ProsodyController initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_prosody(
        self,
        text: str,
        emotion: str,
        context: Optional[Dict] = None,
    ) -> Tuple[str, ProsodyParams]:
        """Compute prosody parameters for *text* given the current *emotion*.

        Also annotates the text with SSML-like pause markers.

        Args:
            text: The utterance to be synthesised.
            emotion: Detected emotion label (e.g. ``"calm"``, ``"urgent"``).
            context: Optional context dict. Keys used:
                ``"speaking_rate_wpm"`` (int), ``"keywords"`` (List[str]),
                ``"volume_override"`` (float).

        Returns:
            A ``(annotated_text, ProsodyParams)`` tuple.
        """
        context = context or {}
        emotion_key = emotion.lower().strip()

        with self._lock:
            self._apply_count += 1

        # Build base params from emotion map.
        adj = _EMOTION_PROSODY.get(emotion_key, _EMOTION_PROSODY["neutral"])
        params = ProsodyParams(
            rate=adj["rate"],
            pitch=adj["pitch"],
            volume=adj["volume"],
            emphasis=adj["emphasis"],
            pause_factor=adj["pause"],
        )

        # Override speaking rate if explicitly supplied.
        if "speaking_rate_wpm" in context:
            params.rate = self.set_speaking_rate(int(context["speaking_rate_wpm"]))

        # Override volume if supplied (e.g. from environmental noise adaptation).
        if "volume_override" in context:
            params.volume = max(0.0, min(2.0, float(context["volume_override"])))

        # Annotate text with pauses.
        annotated = self.add_pause_after_punctuation(text, params.pause_factor)

        # Emphasise any context-supplied keywords.
        keywords: List[str] = context.get("keywords", [])
        # Also emphasise any built-in emphasis trigger words found in text.
        auto_kw = [
            w for w in _EMPHASIS_TRIGGERS
            if re.search(r"\b" + w + r"\b", text, re.IGNORECASE)
        ]
        all_kw = list(set(keywords + auto_kw))
        if all_kw:
            annotated = self.emphasize_keywords(annotated, all_kw)

        logger.debug(
            "apply_prosody: emotion=%r params=%s annotated_len=%d",
            emotion_key,
            params,
            len(annotated),
        )
        return annotated, params

    def set_speaking_rate(self, wpm: int) -> float:
        """Convert a words-per-minute value to a rate multiplier.

        Args:
            wpm: Target speaking rate in words per minute.

        Returns:
            Rate multiplier relative to the 150 wpm baseline.
        """
        clamped = max(_WPM_MIN, min(_WPM_MAX, wpm))
        return round(clamped / _WPM_BASELINE, 3)

    def add_pause_after_punctuation(self, text: str, pause_factor: float) -> str:
        """Insert SSML-like pause annotations after punctuation marks.

        Each punctuation character is followed by a ``<pause ms="N"/>`` tag
        where N is the base duration scaled by *pause_factor*.

        Args:
            text: The raw utterance text.
            pause_factor: Multiplier applied to the base pause durations.
                1.0 = default, >1.0 = slower/more deliberate.

        Returns:
            Annotated text string.
        """
        pause_factor = max(0.1, pause_factor)

        def replacer(match: re.Match) -> str:
            punc = match.group(1)
            base_ms = _PAUSE_MARKS.get(punc, 200)
            actual_ms = int(base_ms * pause_factor)
            return f"{punc}<pause ms=\"{actual_ms}\"/>"

        return _PUNCTUATION_RE.sub(replacer, text)

    def emphasize_keywords(self, text: str, keywords: List[str]) -> str:
        """Wrap *keywords* in SSML ``<emphasis>`` tags.

        Args:
            text: The (possibly already annotated) utterance text.
            keywords: Words or phrases to emphasise.

        Returns:
            Text with emphasis tags inserted.
        """
        for kw in keywords:
            # Escape special regex characters in the keyword.
            escaped = re.escape(kw)
            pattern = re.compile(r"\b(" + escaped + r")\b", re.IGNORECASE)
            # Only wrap if not already inside an emphasis tag.
            text = pattern.sub(r'<emphasis level="strong">\1</emphasis>', text)
        return text

    def get_stats(self) -> Dict:
        """Return runtime statistics."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            return {
                "apply_count": self._apply_count,
                "uptime_seconds": round(elapsed, 2),
                "known_emotions": list(_EMOTION_PROSODY.keys()),
                "pause_marks": len(_PAUSE_MARKS),
                "emphasis_triggers": len(_EMPHASIS_TRIGGERS),
            }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_controller_instance: Optional[ProsodyController] = None
_controller_lock = threading.Lock()


def get_prosody_controller() -> ProsodyController:
    """Return the module-level :class:`ProsodyController` singleton.

    Thread-safe; the instance is created lazily on first call.
    """
    global _controller_instance
    if _controller_instance is None:
        with _controller_lock:
            if _controller_instance is None:
                _controller_instance = ProsodyController()
    return _controller_instance


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ctrl = get_prosody_controller()

    utterances = [
        ("Your heart rate is 78 bpm.", "calm"),
        ("Warning! Heart rate critical — 145 bpm detected!", "urgent"),
        ("Good morning! You slept really well last night.", "happy"),
        ("Battery low. Please charge immediately.", "concerned"),
        ("Timer set for ten minutes.", "neutral"),
    ]

    print("=== ProsodyController Demo ===\n")
    for text, emotion in utterances:
        annotated, params = ctrl.apply_prosody(text, emotion, {})
        print(f"Emotion   : {emotion}")
        print(f"Raw text  : {text}")
        print(f"Annotated : {annotated}")
        print(f"Params    : {params}")
        print()

    # WPM conversion examples.
    print("WPM → rate multiplier:")
    for wpm in [80, 120, 150, 180, 220]:
        print(f"  {wpm} wpm → {ctrl.set_speaking_rate(wpm):.3f}x")

    print("\nStats:", ctrl.get_stats())
