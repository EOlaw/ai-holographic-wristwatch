"""Personality injection module for the AI Holographic Wristwatch.

Applies personality traits to generated response text. Supports warmth,
humour, formality, casual register, conciseness, and verbosity adjustment.
A configurable intensity parameter controls how strongly each trait is applied.

Thread-safe singleton available via :func:`get_personality_injector`.
"""
from __future__ import annotations

import re
import random
import threading
import time
from enum import Enum
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PersonalityTrait(Enum):
    """Supported personality dimensions."""
    FORMAL = "formal"
    CASUAL = "casual"
    WARM = "warm"
    PROFESSIONAL = "professional"
    PLAYFUL = "playful"
    EMPATHETIC = "empathetic"
    CONCISE = "concise"


# ---------------------------------------------------------------------------
# String constants used by the injector
# ---------------------------------------------------------------------------

_WARM_INTROS: List[str] = [
    "I'm happy to help! ",
    "Great question! ",
    "Of course! ",
    "Absolutely! ",
    "Sure thing! ",
    "Glad you asked! ",
    "Happy to share — ",
]

_WARM_CLOSINGS: List[str] = [
    " Let me know if there's anything else I can do.",
    " Feel free to ask if you need more help.",
    " I'm always here if you need me.",
    " Hope that helps!",
    " Take care!",
]

_PLAYFUL_ADDITIONS: List[str] = [
    " (Yes, I noticed — I'm a smartwatch, not a mind reader!)",
    " Fun fact: your wristwatch knows more about you than most people do.",
    " ...which, for a device on your wrist, I find quite impressive.",
    " As they say: time flies — and I literally measure that!",
    " Not bad for a device that used to just show the time, right?",
]

_WRISTWATCH_PUNS: List[str] = [
    "I've been keeping a close watch on that.",
    "Time to get to the point!",
    "Second to none in accuracy.",
    "I'm at your service around the clock.",
    "That's a timely observation.",
    "Minute by minute, I've got you covered.",
]

_EMPATHETIC_BRIDGES: List[str] = [
    "I understand — ",
    "That makes sense — ",
    "I hear you — ",
    "I get it — ",
    "I can see why — ",
]

# Filler words that are stripped in CONCISE mode.
_FILLER_WORDS: List[str] = [
    r"\bactually\b",
    r"\bbasically\b",
    r"\bsimply\b",
    r"\bjust\b",
    r"\bvery\b",
    r"\breally\b",
    r"\bquite\b",
    r"\brather\b",
    r"\bsomewhat\b",
    r"\bliterally\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\ba bit\b",
    r"\byou know\b",
    r"\bI mean\b",
    r"\bwell\b,\s*",
    r"\bso\b,\s*",
]

# Contractions applied in CASUAL mode (formal → casual).
_CASUAL_REPLACEMENTS: Dict[str, str] = {
    r"\bI am\b": "I'm",
    r"\bI have\b": "I've",
    r"\bI will\b": "I'll",
    r"\bdo not\b": "don't",
    r"\bcannot\b": "can't",
    r"\bwill not\b": "won't",
    r"\bis not\b": "isn't",
    r"\bare not\b": "aren't",
    r"\bwas not\b": "wasn't",
    r"\bwere not\b": "weren't",
    r"\bshould not\b": "shouldn't",
    r"\bwould not\b": "wouldn't",
    r"\bcould not\b": "couldn't",
    r"\byou are\b": "you're",
    r"\bwe are\b": "we're",
    r"\bthey are\b": "they're",
    r"\bit is\b": "it's",
    r"\bthat is\b": "that's",
    r"\bthere is\b": "there's",
    r"\bHello\b": "Hey",
    r"\bGoodbye\b": "Bye",
}

# Formal replacements applied in FORMAL mode (casual → formal).
_FORMAL_REPLACEMENTS: Dict[str, str] = {
    r"\bI'm\b": "I am",
    r"\bI've\b": "I have",
    r"\bI'll\b": "I will",
    r"\bdon't\b": "do not",
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bshouldn't\b": "should not",
    r"\bwouldn't\b": "would not",
    r"\bcouldn't\b": "could not",
    r"\byou're\b": "you are",
    r"\bwe're\b": "we are",
    r"\bthey're\b": "they are",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bthere's\b": "there is",
    r"\bHey\b": "Hello",
    r"\bBye\b": "Goodbye",
    r"\bOk\b": "Understood",
    r"\byep\b": "yes",
    r"\bnope\b": "no",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PersonalityInjector:
    """Applies personality traits to raw response text.

    Thread-safe; a shared singleton is available via
    :func:`get_personality_injector`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._inject_count: int = 0
        self._start_time: float = time.monotonic()
        logger.debug("PersonalityInjector initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject(
        self,
        text: str,
        traits: List[PersonalityTrait],
        intensity: float = 0.5,
    ) -> str:
        """Apply a set of personality *traits* to *text*.

        Traits are applied in a defined order so that conflicting adjustments
        (e.g. FORMAL + CASUAL) are resolved predictably — later traits in the
        list override earlier ones.

        Args:
            text: The raw generated response string.
            traits: List of :class:`PersonalityTrait` values to apply.
            intensity: How strongly to apply each trait. 0.0 = minimal effect,
                1.0 = maximum effect. Values outside [0, 1] are clamped.

        Returns:
            The modified response string.
        """
        intensity = max(0.0, min(1.0, intensity))

        with self._lock:
            self._inject_count += 1

        logger.debug(
            "inject() #%d — traits=%r intensity=%.2f",
            self._inject_count,
            [t.value for t in traits],
            intensity,
        )

        # Apply traits in a sensible order.
        trait_set = set(traits)

        # 1. Register adjustments.
        if PersonalityTrait.FORMAL in trait_set:
            text = self.add_formality(text)
        if PersonalityTrait.CASUAL in trait_set:
            text = self.make_casual(text)

        # 2. Warmth / empathy additions.
        if PersonalityTrait.WARM in trait_set and random.random() < intensity:
            text = self.add_warmth(text)
        if PersonalityTrait.EMPATHETIC in trait_set and random.random() < intensity:
            bridge = random.choice(_EMPATHETIC_BRIDGES)
            if not text.lower().startswith(bridge.lower().strip()):
                text = bridge + text[0].lower() + text[1:]

        # 3. Playful / humour.
        if PersonalityTrait.PLAYFUL in trait_set and random.random() < intensity * 0.6:
            text = self.add_humor(text)

        # 4. Professional: strip playful additions; ensure clean sentence end.
        if PersonalityTrait.PROFESSIONAL in trait_set:
            text = self._make_professional(text)

        # 5. Concise: remove fillers.
        if PersonalityTrait.CONCISE in trait_set:
            text = self.make_concise(text)

        return text.strip()

    def add_warmth(self, text: str) -> str:
        """Prepend a warm introductory phrase to *text*."""
        intro = random.choice(_WARM_INTROS)
        # Avoid double-prefixing if text already starts with a warm phrase.
        for phrase in _WARM_INTROS:
            if text.startswith(phrase.strip()):
                return text
        # Lowercase the first character of the original text when we prepend.
        if text:
            text = intro + text[0].lower() + text[1:]
        else:
            text = intro.strip()
        return text

    def add_humor(self, text: str) -> str:
        """Append a light wristwatch-themed witty remark to *text*."""
        # Only append if the text doesn't already end in a pun.
        for pun in _WRISTWATCH_PUNS:
            if pun.lower() in text.lower():
                return text
        addition = random.choice(_WRISTWATCH_PUNS + _PLAYFUL_ADDITIONS)
        if not text.endswith((".", "!", "?")):
            text += "."
        return text + " " + addition

    def add_formality(self, text: str) -> str:
        """Convert contractions and casual language to formal equivalents."""
        for pattern, replacement in _FORMAL_REPLACEMENTS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def make_casual(self, text: str) -> str:
        """Apply contractions and colloquial language to *text*."""
        for pattern, replacement in _CASUAL_REPLACEMENTS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def make_concise(self, text: str) -> str:
        """Remove common filler words from *text*."""
        for pattern in _FILLER_WORDS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        # Clean up double spaces left behind.
        text = re.sub(r"  +", " ", text).strip()
        return text

    def adjust_verbosity(self, text: str, level: float) -> str:
        """Adjust response verbosity.

        Args:
            text: The response string.
            level: 0.0 = very brief (truncates at first sentence),
                   1.0 = verbose (appends a closing phrase).

        Returns:
            Adjusted text.
        """
        level = max(0.0, min(1.0, level))

        if level < 0.25:
            # Keep only the first sentence.
            match = re.search(r"[.!?]", text)
            if match:
                text = text[: match.end()]
        elif level > 0.75:
            # Append a warm closing.
            closing = random.choice(_WARM_CLOSINGS)
            if not any(text.endswith(c.strip()) for c in _WARM_CLOSINGS):
                text = text.rstrip(".!?") + "." + closing
        return text.strip()

    def get_stats(self) -> Dict:
        """Return runtime statistics."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            return {
                "inject_count": self._inject_count,
                "uptime_seconds": round(elapsed, 2),
                "available_traits": [t.value for t in PersonalityTrait],
                "warm_intros": len(_WARM_INTROS),
                "filler_patterns": len(_FILLER_WORDS),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_professional(self, text: str) -> str:
        """Strip informal additions and ensure the text ends cleanly."""
        # Remove playful additions that were injected.
        for pun in _WRISTWATCH_PUNS + _PLAYFUL_ADDITIONS:
            text = text.replace(" " + pun, "").replace(pun, "")
        text = text.strip()
        if text and not text[-1] in ".!?":
            text += "."
        return text


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_injector_instance: Optional[PersonalityInjector] = None
_injector_lock = threading.Lock()


def get_personality_injector() -> PersonalityInjector:
    """Return the module-level :class:`PersonalityInjector` singleton.

    Thread-safe; the instance is created lazily on first call.
    """
    global _injector_instance
    if _injector_instance is None:
        with _injector_lock:
            if _injector_instance is None:
                _injector_instance = PersonalityInjector()
    return _injector_instance


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    inj = get_personality_injector()
    base = "Your heart rate is 78 bpm."

    configs = [
        ([PersonalityTrait.WARM], 0.9),
        ([PersonalityTrait.PLAYFUL], 0.9),
        ([PersonalityTrait.FORMAL], 1.0),
        ([PersonalityTrait.CASUAL], 1.0),
        ([PersonalityTrait.CONCISE], 1.0),
        ([PersonalityTrait.WARM, PersonalityTrait.CASUAL], 0.8),
        ([PersonalityTrait.PROFESSIONAL], 1.0),
        ([PersonalityTrait.EMPATHETIC], 0.9),
    ]

    print("=== PersonalityInjector Demo ===\n")
    print(f"Base text: {base!r}\n")
    for traits, intensity in configs:
        result = inj.inject(base, traits, intensity)
        print(f"Traits={[t.value for t in traits]}  intensity={intensity}")
        print(f"  -> {result!r}\n")

    print("Stats:", inj.get_stats())
