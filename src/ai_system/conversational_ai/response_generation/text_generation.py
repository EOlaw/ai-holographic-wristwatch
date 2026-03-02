"""Text generation module for the AI Holographic Wristwatch.

Responsible for producing natural language responses by selecting and filling
intent-specific templates. Supports multiple response types, weighted template
selection, style adjustment (verbosity & formality), and a thread-safe singleton
interface.
"""
from __future__ import annotations

import re
import threading
import time
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ResponseType(Enum):
    """High-level category of a generated response."""
    ANSWER = "answer"
    CONFIRMATION = "confirmation"
    QUESTION = "question"
    GREETING = "greeting"
    FAREWELL = "farewell"
    HEALTH_REPORT = "health_report"
    NOTIFICATION = "notification"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GeneratedText:
    """Container for a single generated response."""
    text: str
    response_type: ResponseType
    confidence: float
    alternatives: List[str] = field(default_factory=list)
    intent_name: str = ""
    slot_count: int = 0

    def __repr__(self) -> str:
        return (
            f"GeneratedText(intent={self.intent_name!r}, "
            f"type={self.response_type.value}, "
            f"confidence={self.confidence:.2f}, "
            f"text={self.text[:60]!r})"
        )


# ---------------------------------------------------------------------------
# Template library
# ---------------------------------------------------------------------------

# Maps intent names → list of (template_string, weight) tuples.
# A weight > 1 makes a template more likely to be selected.
RESPONSE_TEMPLATES: Dict[str, List[tuple]] = {
    # Health & fitness
    "query_heart_rate": [
        ("Your heart rate is {heart_rate} bpm.", 2),
        ("Currently {heart_rate} beats per minute.", 1),
        ("Heart rate check: {heart_rate} bpm right now.", 1),
        ("I'm reading {heart_rate} bpm on your heart rate monitor.", 1),
    ],
    "query_steps": [
        ("You've taken {steps} steps today.", 2),
        ("Step count: {steps} so far today.", 1),
        ("You're at {steps} steps — keep it up!", 1),
        ("Today's step total is {steps}.", 1),
    ],
    "query_calories": [
        ("You've burned approximately {calories} calories today.", 2),
        ("Calorie burn so far: {calories} kcal.", 1),
        ("About {calories} calories burned since midnight.", 1),
    ],
    "query_sleep": [
        ("Last night you slept {hours} hours with {quality} quality.", 2),
        ("Sleep summary: {hours} hours, quality rated {quality}.", 1),
        ("You got {hours} hours of {quality}-quality sleep.", 1),
    ],
    "query_blood_oxygen": [
        ("Your SpO2 reading is {spo2}%.", 2),
        ("Blood oxygen saturation: {spo2}%.", 1),
        ("Current oxygen level is {spo2}% — within normal range.", 1),
    ],
    "query_stress": [
        ("Your stress level is currently {level}.", 2),
        ("Stress index: {level}. {suggestion}", 1),
        ("I detect a {level} stress level right now.", 1),
    ],
    # Time & timers
    "query_time": [
        ("It's {time} right now.", 2),
        ("The current time is {time}.", 1),
        ("Right now it's {time}.", 1),
    ],
    "query_date": [
        ("Today is {date}.", 2),
        ("The date is {date}.", 1),
    ],
    "set_timer": [
        ("Timer set for {duration}.", 2),
        ("I've started a {duration} timer.", 1),
        ("Got it — your {duration} timer is running.", 1),
        ("A {duration} timer has been set. I'll alert you when it's done.", 1),
    ],
    "cancel_timer": [
        ("Your {duration} timer has been cancelled.", 2),
        ("Timer cancelled.", 1),
        ("I've stopped the {duration} timer.", 1),
    ],
    "set_alarm": [
        ("Alarm set for {time}.", 2),
        ("I'll wake you at {time}.", 1),
        ("Your alarm is scheduled for {time}.", 1),
    ],
    # Weather
    "query_weather": [
        ("It's {condition} outside at {temp}°C.", 2),
        ("Current weather: {condition}, {temp}°C.", 1),
        ("Outside it's {temp}°C and {condition}.", 1),
        ("Weather update: {condition} with a temperature of {temp}°C.", 1),
    ],
    "query_forecast": [
        ("Tomorrow looks {condition} with a high of {high}°C and low of {low}°C.", 2),
        ("Forecast: {condition}, {high}°C high, {low}°C low.", 1),
    ],
    # Navigation
    "start_navigation": [
        ("Starting navigation to {destination}.", 2),
        ("Got it, navigating to {destination} now.", 1),
        ("Route to {destination} found — turn-by-turn guidance starting.", 1),
    ],
    "query_eta": [
        ("You'll arrive at {destination} in about {eta}.", 2),
        ("ETA to {destination}: {eta}.", 1),
    ],
    # Messaging & communication
    "send_message": [
        ("Message sent to {contact}.", 2),
        ("Your message to {contact} has been delivered.", 1),
        ("Done — {contact} has been notified.", 1),
    ],
    "read_message": [
        ("{sender} says: {message}", 2),
        ("New message from {sender}: {message}", 1),
    ],
    # General interaction
    "greeting": [
        ("Hello! How can I help you today?", 2),
        ("Hi there! What can I do for you?", 1),
        ("Hey! Ready to assist.", 1),
        ("Good to see you! What do you need?", 1),
    ],
    "farewell": [
        ("Goodbye! Have a great day.", 2),
        ("See you later! Take care.", 1),
        ("Bye! Stay healthy.", 1),
    ],
    "general_conversation": [
        ("That's interesting! Tell me more.", 2),
        ("I'm here to help!", 1),
        ("Good point!", 1),
        ("I hadn't thought of it that way.", 1),
        ("Let me know if there's anything I can do.", 1),
    ],
    "confirm_action": [
        ("Done! {action} completed.", 2),
        ("All set — {action} is taken care of.", 1),
        ("Confirmed: {action}.", 1),
    ],
    "error_fallback": [
        ("Sorry, I didn't quite catch that. Could you rephrase?", 2),
        ("I'm not sure I understood. Can you say that again?", 1),
        ("Hmm, I couldn't process that request. Please try again.", 1),
    ],
    "low_battery": [
        ("Battery is at {level}%. Please charge your watch soon.", 2),
        ("Low battery warning: {level}% remaining.", 1),
        ("You're running low — only {level}% battery left.", 1),
    ],
    "unknown": [
        ("I'm not sure about that, but I'm learning every day.", 2),
        ("That's outside my current knowledge. Want me to search for it?", 1),
        ("I don't have information on that yet.", 1),
    ],
}

# Maps intent names to their ResponseType category.
_INTENT_RESPONSE_TYPE: Dict[str, ResponseType] = {
    "query_heart_rate": ResponseType.HEALTH_REPORT,
    "query_steps": ResponseType.HEALTH_REPORT,
    "query_calories": ResponseType.HEALTH_REPORT,
    "query_sleep": ResponseType.HEALTH_REPORT,
    "query_blood_oxygen": ResponseType.HEALTH_REPORT,
    "query_stress": ResponseType.HEALTH_REPORT,
    "query_time": ResponseType.ANSWER,
    "query_date": ResponseType.ANSWER,
    "set_timer": ResponseType.CONFIRMATION,
    "cancel_timer": ResponseType.CONFIRMATION,
    "set_alarm": ResponseType.CONFIRMATION,
    "query_weather": ResponseType.ANSWER,
    "query_forecast": ResponseType.ANSWER,
    "start_navigation": ResponseType.CONFIRMATION,
    "query_eta": ResponseType.ANSWER,
    "send_message": ResponseType.CONFIRMATION,
    "read_message": ResponseType.NOTIFICATION,
    "greeting": ResponseType.GREETING,
    "farewell": ResponseType.FAREWELL,
    "general_conversation": ResponseType.ANSWER,
    "confirm_action": ResponseType.CONFIRMATION,
    "error_fallback": ResponseType.ERROR,
    "low_battery": ResponseType.NOTIFICATION,
    "unknown": ResponseType.ANSWER,
}

# Filler words and phrases added to verbose responses.
_VERBOSE_ADDITIONS: List[str] = [
    " Just so you know, ",
    " By the way, ",
    " For your reference, ",
    " Worth noting that ",
    " As a heads-up, ",
]

# Formal language mappings applied when formality > 0.7.
_FORMALITY_MAP: Dict[str, str] = {
    r"\bdon't\b": "do not",
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwe're\b": "we are",
    r"\byou're\b": "you are",
    r"\bI'm\b": "I am",
    r"\bI've\b": "I have",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bthere's\b": "there is",
    r"\bHey\b": "Hello",
    r"\bhi\b": "hello",
    r"\bGot it\b": "Understood",
    r"\byeah\b": "yes",
    r"\bnope\b": "no",
}

# Informal replacements applied when formality < 0.3.
_CASUAL_MAP: Dict[str, str] = {
    r"\bdo not\b": "don't",
    r"\bcannot\b": "can't",
    r"\bwill not\b": "won't",
    r"\bis not\b": "isn't",
    r"\bHello\b": "Hey",
    r"\bUnderstood\b": "Got it",
    r"\byes\b": "yep",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TextGenerator:
    """Generates natural language responses from intent + slot data.

    Thread-safe; a shared singleton is available via :func:`get_text_generator`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._generate_count: int = 0
        self._start_time: float = time.monotonic()
        logger.debug("TextGenerator initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        intent_name: str,
        slots: Dict,
        context: Optional[Dict] = None,
    ) -> GeneratedText:
        """Generate a response for *intent_name* using the provided *slots*.

        Args:
            intent_name: The detected intent (e.g. ``"query_heart_rate"``).
            slots: A mapping of slot names to values that will be interpolated
                into the template (e.g. ``{"heart_rate": 72}``).
            context: Optional dialogue/session context (not used for template
                filling but logged for debugging).

        Returns:
            A :class:`GeneratedText` instance with the primary response and
            a list of alternative formulations.
        """
        context = context or {}
        with self._lock:
            self._generate_count += 1
            count = self._generate_count

        logger.debug(
            "generate() call #%d — intent=%r slots=%r",
            count,
            intent_name,
            slots,
        )

        # Resolve the template pool.
        resolved_intent = intent_name if intent_name in RESPONSE_TEMPLATES else "unknown"
        pool = RESPONSE_TEMPLATES[resolved_intent]

        # Pick primary template and fill it.
        primary_template = self._select_template(resolved_intent)
        primary_text = self._fill_template(primary_template, slots)

        # Build alternatives from the remaining templates (up to 3).
        alternatives: List[str] = []
        seen = {primary_template}
        for tpl, _w in pool:
            if tpl not in seen:
                alternatives.append(self._fill_template(tpl, slots))
                seen.add(tpl)
            if len(alternatives) >= 3:
                break

        # Determine response type.
        response_type = _INTENT_RESPONSE_TYPE.get(
            resolved_intent, ResponseType.ANSWER
        )

        # Confidence is reduced when we fall back to the "unknown" intent.
        confidence = 0.95 if resolved_intent == intent_name else 0.40

        return GeneratedText(
            text=primary_text,
            response_type=response_type,
            confidence=confidence,
            alternatives=alternatives,
            intent_name=intent_name,
            slot_count=len(slots),
        )

    def apply_style(self, text: str, verbosity: float, formality: float) -> str:
        """Adjust the style of *text* based on verbosity and formality scores.

        Args:
            text: The raw generated text.
            verbosity: 0.0 = very brief, 1.0 = verbose. Values outside [0, 1]
                are clamped.
            formality: 0.0 = very casual, 1.0 = very formal. Values outside
                [0, 1] are clamped.

        Returns:
            The style-adjusted text.
        """
        verbosity = max(0.0, min(1.0, verbosity))
        formality = max(0.0, min(1.0, formality))

        # Verbosity: verbose mode prepends an attention phrase.
        if verbosity > 0.75 and text and not text.startswith(tuple(_VERBOSE_ADDITIONS)):
            prefix = random.choice(_VERBOSE_ADDITIONS).strip()
            text = prefix + " " + text[0].lower() + text[1:]

        # Formality adjustments.
        if formality >= 0.7:
            for pattern, replacement in _FORMALITY_MAP.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        elif formality <= 0.3:
            for pattern, replacement in _CASUAL_MAP.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def get_stats(self) -> Dict:
        """Return runtime statistics for this generator instance."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            return {
                "generate_count": self._generate_count,
                "uptime_seconds": round(elapsed, 2),
                "known_intents": len(RESPONSE_TEMPLATES),
                "total_templates": sum(
                    len(v) for v in RESPONSE_TEMPLATES.values()
                ),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_template(self, template: str, slots: Dict) -> str:
        """Replace ``{slot_name}`` placeholders in *template* with slot values.

        Missing slots are left as their placeholder text rather than raising.
        """
        try:
            return template.format_map(_DefaultDict(str, slots))
        except Exception as exc:  # pragma: no cover
            logger.warning("Template fill error for %r: %s", template, exc)
            return template

    def _select_template(self, intent_name: str) -> str:
        """Randomly select a template from the pool using weighted sampling."""
        pool = RESPONSE_TEMPLATES.get(intent_name, RESPONSE_TEMPLATES["unknown"])
        templates, weights = zip(*pool)
        (chosen,) = random.choices(list(templates), weights=list(weights), k=1)
        return chosen

    def _generate_fallback(self, intent_name: str) -> str:
        """Generate a generic fallback string when no template matches."""
        return (
            f"I received your request about '{intent_name}', "
            "but I don't have a specific response template for that yet."
        )


class _DefaultDict(dict):
    """A dict subclass that returns the key placeholder when a key is missing."""

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return "{" + key + "}"


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_generator_instance: Optional[TextGenerator] = None
_generator_lock = threading.Lock()


def get_text_generator() -> TextGenerator:
    """Return the module-level :class:`TextGenerator` singleton.

    Thread-safe; the instance is created lazily on first call.
    """
    global _generator_instance
    if _generator_instance is None:
        with _generator_lock:
            if _generator_instance is None:
                _generator_instance = TextGenerator()
    return _generator_instance


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gen = get_text_generator()

    demos = [
        ("query_heart_rate", {"heart_rate": 72}),
        ("query_steps", {"steps": 8_432}),
        ("set_timer", {"duration": "10 minutes"}),
        ("query_weather", {"condition": "partly cloudy", "temp": 18}),
        ("greeting", {}),
        ("query_sleep", {"hours": 7.5, "quality": "good"}),
        ("general_conversation", {}),
    ]

    print("=== TextGenerator Demo ===\n")
    for intent, slots in demos:
        result = gen.generate(intent, slots)
        styled = gen.apply_style(result.text, verbosity=0.3, formality=0.5)
        print(f"Intent    : {intent}")
        print(f"Primary   : {result.text}")
        print(f"Styled    : {styled}")
        print(f"Type      : {result.response_type.value}")
        print(f"Confidence: {result.confidence:.2f}")
        if result.alternatives:
            print(f"Alt 1     : {result.alternatives[0]}")
        print()

    print("Stats:", gen.get_stats())
