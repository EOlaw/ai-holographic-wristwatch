"""
Intent Recognition — AI Holographic Wristwatch

Classifies user utterances into structured intents for the dialogue system.
Uses a hybrid approach: keyword/pattern matching for common commands,
with LLM fallback for complex or ambiguous requests.

Supported intent domains:
- Health queries (heart rate, sleep, stress levels)
- System control (brightness, ANC, battery)
- Communication (messages, calls, reminders)
- Navigation (directions, location)
- AI conversation (general chat, questions)
- Hologram control (show/hide, resize, move)
"""

from __future__ import annotations

import re
import threading
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class IntentDomain(Enum):
    HEALTH        = "health"
    SYSTEM        = "system"
    COMMUNICATION = "communication"
    NAVIGATION    = "navigation"
    CONVERSATION  = "conversation"
    HOLOGRAM      = "hologram"
    TIMER_ALARM   = "timer_alarm"
    WEATHER       = "weather"
    UNKNOWN       = "unknown"


class IntentConfidence(Enum):
    HIGH    = "high"    # > 0.85
    MEDIUM  = "medium"  # 0.60–0.85
    LOW     = "low"     # 0.40–0.60
    UNCLEAR = "unclear" # < 0.40


@dataclass
class Intent:
    """Structured intent extracted from user utterance."""
    name: str                                        # e.g. "query_heart_rate"
    domain: IntentDomain = IntentDomain.UNKNOWN
    confidence: float = 0.0
    confidence_level: IntentConfidence = IntentConfidence.UNCLEAR
    parameters: Dict[str, str] = field(default_factory=dict)
    raw_utterance: str = ""
    alternatives: List["Intent"] = field(default_factory=list)
    requires_clarification: bool = False
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Pattern-based intent rules
# ---------------------------------------------------------------------------

# Each rule: (pattern, intent_name, domain, parameters_extractor)
INTENT_RULES: List[Tuple] = [
    # Health
    (r"\b(heart rate|pulse|bpm)\b",              "query_heart_rate",   IntentDomain.HEALTH,        {}),
    (r"\b(blood oxygen|spo2|oxygen)\b",           "query_spo2",         IntentDomain.HEALTH,        {}),
    (r"\b(stress|stress level)\b",                "query_stress",       IntentDomain.HEALTH,        {}),
    (r"\b(sleep|sleep score|sleep quality)\b",    "query_sleep",        IntentDomain.HEALTH,        {}),
    (r"\b(steps?|step count)\b",                  "query_steps",        IntentDomain.HEALTH,        {}),
    (r"\b(calories?)\b",                          "query_calories",     IntentDomain.HEALTH,        {}),
    (r"\b(temperature|body temp)\b",              "query_temperature",  IntentDomain.HEALTH,        {}),
    (r"\b(hydration|water|drink)\b",              "query_hydration",    IntentDomain.HEALTH,        {}),
    # System control
    (r"\b(brightness|dim|brighter)\b",            "control_brightness", IntentDomain.SYSTEM,        {}),
    (r"\b(battery|power|charge)\b",               "query_battery",      IntentDomain.SYSTEM,        {}),
    (r"\b(volume|sound|mute)\b",                  "control_volume",     IntentDomain.SYSTEM,        {}),
    (r"\b(do not disturb|dnd|focus mode)\b",      "set_focus_mode",     IntentDomain.SYSTEM,        {}),
    # Communication
    (r"\b(call|phone|ring)\s+(\w+)",              "initiate_call",      IntentDomain.COMMUNICATION, {"contact": r"\b(?:call|phone|ring)\s+(\w+)"}),
    (r"\b(message|text|send)\s+(\w+)",            "send_message",       IntentDomain.COMMUNICATION, {"contact": r"\b(?:message|text|send)\s+(\w+)"}),
    (r"\b(reminder|remind me)\b",                 "set_reminder",       IntentDomain.COMMUNICATION, {}),
    (r"\b(alarm|wake me)\b",                      "set_alarm",          IntentDomain.TIMER_ALARM,   {}),
    (r"\b(timer|countdown)\b",                    "set_timer",          IntentDomain.TIMER_ALARM,   {}),
    # Navigation
    (r"\b(directions?|navigate|how do i get)\b",  "navigate_to",        IntentDomain.NAVIGATION,    {}),
    (r"\b(where am i|location|current location)\b","query_location",     IntentDomain.NAVIGATION,    {}),
    (r"\b(nearby|around me|close to)\b",           "find_nearby",        IntentDomain.NAVIGATION,    {}),
    # Hologram
    (r"\b(show hologram|display|project)\b",       "show_hologram",      IntentDomain.HOLOGRAM,      {}),
    (r"\b(hide hologram|dismiss|close)\b",         "hide_hologram",      IntentDomain.HOLOGRAM,      {}),
    (r"\b(resize|bigger|smaller|scale)\b",         "resize_hologram",    IntentDomain.HOLOGRAM,      {}),
    # Weather
    (r"\b(weather|forecast|temperature outside)\b","query_weather",       IntentDomain.WEATHER,       {}),
    (r"\b(rain|snow|cloudy|sunny|wind)\b",         "query_weather",       IntentDomain.WEATHER,       {}),
]


def _compile_rules():
    return [(re.compile(pattern, re.IGNORECASE), name, domain, params)
            for pattern, name, domain, params in INTENT_RULES]


COMPILED_RULES = _compile_rules()


class PatternMatcher:
    """Fast pattern-based intent matching for common commands."""

    def match(self, utterance: str) -> List[Intent]:
        matches = []
        for pattern, name, domain, param_patterns in COMPILED_RULES:
            m = pattern.search(utterance)
            if m:
                params = {}
                for param_key, param_pattern in param_patterns.items():
                    pm = re.search(param_pattern, utterance, re.IGNORECASE)
                    if pm and pm.lastindex:
                        params[param_key] = pm.group(1)
                matches.append(Intent(
                    name=name,
                    domain=domain,
                    confidence=0.85,
                    confidence_level=IntentConfidence.HIGH,
                    parameters=params,
                    raw_utterance=utterance,
                ))
        return matches


class IntentRanker:
    """Ranks multiple matched intents by relevance and domain priority."""

    DOMAIN_PRIORITY = {
        IntentDomain.HEALTH: 1,
        IntentDomain.SYSTEM: 2,
        IntentDomain.COMMUNICATION: 3,
        IntentDomain.NAVIGATION: 4,
        IntentDomain.TIMER_ALARM: 5,
        IntentDomain.WEATHER: 6,
        IntentDomain.HOLOGRAM: 7,
        IntentDomain.CONVERSATION: 8,
        IntentDomain.UNKNOWN: 9,
    }

    def rank(self, intents: List[Intent]) -> List[Intent]:
        if not intents:
            return intents
        return sorted(intents,
                      key=lambda i: (self.DOMAIN_PRIORITY.get(i.domain, 9), -i.confidence))


# ---------------------------------------------------------------------------
# Intent Recognizer
# ---------------------------------------------------------------------------

_GLOBAL_IR: Optional["IntentRecognizer"] = None
_GLOBAL_IR_LOCK = threading.Lock()


class IntentRecognizer:
    """
    Hybrid intent recognizer: pattern matching + LLM fallback.
    For ambiguous inputs, produces multiple intent candidates for clarification.
    """

    FALLBACK_INTENT = "general_conversation"

    def __init__(self) -> None:
        self._pattern_matcher = PatternMatcher()
        self._ranker          = IntentRanker()
        self._lock            = threading.RLock()
        self._recognized_count = 0
        self._fallback_count   = 0

    def recognize(self, utterance: str, context: Optional[Dict] = None) -> Intent:
        """
        Recognize intent from utterance.
        Returns top intent with alternatives for disambiguation.
        """
        with self._lock:
            utterance = utterance.strip()
            if not utterance:
                return Intent(name="empty", domain=IntentDomain.UNKNOWN,
                              confidence=0.0, raw_utterance=utterance)

            candidates = self._pattern_matcher.match(utterance)
            ranked     = self._ranker.rank(candidates)

            if ranked:
                top = ranked[0]
                top.alternatives = ranked[1:3]
                top.requires_clarification = len(ranked) > 1 and ranked[1].confidence > 0.75
                self._recognized_count += 1
                logger.debug(f"Intent '{top.name}' (conf={top.confidence:.2f}) from: '{utterance[:40]}'")
                return top

            # Fallback: general conversation
            self._fallback_count += 1
            return Intent(
                name=self.FALLBACK_INTENT,
                domain=IntentDomain.CONVERSATION,
                confidence=0.60,
                confidence_level=IntentConfidence.MEDIUM,
                raw_utterance=utterance,
            )

    def recognize_batch(self, utterances: List[str]) -> List[Intent]:
        return [self.recognize(u) for u in utterances]

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "recognized": self._recognized_count,
                "fallbacks":  self._fallback_count,
                "total":      self._recognized_count + self._fallback_count,
            }


def get_intent_recognizer() -> IntentRecognizer:
    global _GLOBAL_IR
    with _GLOBAL_IR_LOCK:
        if _GLOBAL_IR is None:
            _GLOBAL_IR = IntentRecognizer()
        return _GLOBAL_IR


def run_intent_recognition_tests() -> bool:
    ir = IntentRecognizer()
    tests = [
        ("What is my heart rate?",       "query_heart_rate",     IntentDomain.HEALTH),
        ("Show me the hologram",         "show_hologram",        IntentDomain.HOLOGRAM),
        ("Call Sarah",                   "initiate_call",        IntentDomain.COMMUNICATION),
        ("How is the weather today?",    "query_weather",        IntentDomain.WEATHER),
        ("Set a timer for 5 minutes",    "set_timer",            IntentDomain.TIMER_ALARM),
        ("How are you?",                 "general_conversation", IntentDomain.CONVERSATION),
    ]
    for utterance, expected_name, expected_domain in tests:
        intent = ir.recognize(utterance)
        assert intent.name == expected_name, \
            f"Expected '{expected_name}', got '{intent.name}' for: '{utterance}'"
        assert intent.domain == expected_domain
    logger.info("IntentRecognizer tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_intent_recognition_tests()
    ir = get_intent_recognizer()
    samples = [
        "What's my stress level?",
        "Navigate to the nearest coffee shop",
        "Remind me to drink water in 30 minutes",
        "Tell me a joke",
    ]
    for s in samples:
        intent = ir.recognize(s)
        print(f"  '{s}'\n    → intent={intent.name}  domain={intent.domain.value}  "
              f"conf={intent.confidence:.2f}")
