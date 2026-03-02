"""Command Parsing — AI Holographic Wristwatch

Transforms raw user utterances into structured ``ParsedCommand`` objects by
normalising text, extracting the primary action verb, resolving the target
entity, collecting modifiers, and classifying the command type.
"""
from __future__ import annotations

import re
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    from src.core.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CommandType(Enum):
    CONTROL = "control"
    QUERY = "query"
    NAVIGATION = "navigation"
    COMMUNICATION = "communication"
    HEALTH = "health"
    HOLOGRAM = "hologram"
    TIMER = "timer"
    MEDIA = "media"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParsedCommand:
    """Structured representation of a user command."""
    raw_text: str
    command_type: CommandType
    action: str
    target: str
    modifiers: Dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "raw_text": self.raw_text,
            "command_type": self.command_type.value,
            "action": self.action,
            "target": self.target,
            "modifiers": dict(self.modifiers),
            "confidence": round(self.confidence, 3),
        }

    def __repr__(self) -> str:
        return (
            f"ParsedCommand(type={self.command_type.value}, action={self.action!r}, "
            f"target={self.target!r}, modifiers={self.modifiers}, "
            f"confidence={self.confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

# Maps canonical action name -> list of synonym verbs / phrases
ACTION_PATTERNS: Dict[str, List[str]] = {
    # Media
    "play":          ["play", "start playing", "put on", "listen to", "stream"],
    "pause":         ["pause", "hold", "freeze"],
    "stop":          ["stop", "end", "terminate", "quit", "abort", "finish"],
    "resume":        ["resume", "continue", "unpause", "keep playing"],
    "skip":          ["skip", "next", "forward"],
    "previous":      ["previous", "back", "last", "rewind"],
    "volume_up":     ["volume up", "louder", "increase volume", "turn up"],
    "volume_down":   ["volume down", "quieter", "decrease volume", "turn down"],
    "set_volume":    ["set volume", "volume to", "change volume"],
    # Timer
    "set_timer":     ["set a timer", "set timer", "timer for", "start a timer", "countdown for"],
    "set_alarm":     ["set an alarm", "set alarm", "alarm at", "wake me at", "alarm for"],
    "cancel_timer":  ["cancel timer", "stop timer", "delete timer", "remove timer"],
    "cancel_alarm":  ["cancel alarm", "stop alarm", "delete alarm", "remove alarm"],
    # Navigation
    "navigate":      ["navigate", "directions to", "take me to", "get directions", "route to",
                      "how do i get to", "drive to", "walk to"],
    "search_nearby": ["find nearby", "near me", "closest", "nearest"],
    # Communication
    "call":          ["call", "phone", "dial", "ring"],
    "send_message":  ["send a message", "send message", "text", "message", "sms to"],
    "send_email":    ["send email", "email", "compose"],
    "read_messages": ["read messages", "check messages", "show messages", "any messages"],
    # Health
    "check_health":  ["check health", "health status", "how am i doing", "my health"],
    "check_steps":   ["how many steps", "step count", "steps today"],
    "check_heart_rate": ["heart rate", "pulse", "bpm", "my heart"],
    "start_workout": ["start workout", "begin workout", "start exercise", "track run"],
    "stop_workout":  ["stop workout", "end workout", "finish exercise"],
    # Hologram
    "show_hologram": ["show", "display", "project", "render", "visualise", "visualize",
                      "bring up", "open"],
    "hide_hologram": ["hide", "close", "dismiss", "remove", "turn off display"],
    "set_brightness":["brightness", "set brightness", "brighter", "dimmer", "dim"],
    # General control
    "query":         ["what", "when", "where", "who", "why", "how", "tell me",
                      "show me", "find", "search for", "look up", "get"],
    "confirm":       ["yes", "yeah", "yep", "sure", "okay", "confirm", "correct", "right"],
    "deny":          ["no", "nope", "cancel", "never mind", "wrong"],
    "help":          ["help", "assist", "support", "what can you do"],
}

# Maps target-domain keywords -> normalised target category
TARGET_PATTERNS: Dict[str, List[str]] = {
    "music":         ["music", "song", "track", "album", "artist", "playlist", "spotify"],
    "podcast":       ["podcast", "episode", "show"],
    "timer":         ["timer", "countdown"],
    "alarm":         ["alarm", "wake"],
    "contact":       ["contact", "person", "friend", "family", "colleague"],
    "navigation":    ["map", "route", "directions", "location", "address", "place"],
    "health":        ["health", "heart", "steps", "calories", "sleep", "workout",
                      "fitness", "pulse", "bpm"],
    "hologram":      ["hologram", "display", "projection", "screen", "image"],
    "settings":      ["settings", "preferences", "configuration", "options", "brightness",
                      "volume", "wifi", "bluetooth"],
    "messages":      ["message", "text", "sms", "email", "mail"],
    "weather":       ["weather", "temperature", "forecast", "rain", "sun"],
    "news":          ["news", "headlines", "update"],
    "reminder":      ["reminder", "remind", "note"],
    "calendar":      ["calendar", "event", "appointment", "meeting", "schedule"],
}

# Action -> CommandType classification
_ACTION_TYPE_MAP: Dict[str, CommandType] = {
    "play":             CommandType.MEDIA,
    "pause":            CommandType.MEDIA,
    "stop":             CommandType.CONTROL,
    "resume":           CommandType.MEDIA,
    "skip":             CommandType.MEDIA,
    "previous":         CommandType.MEDIA,
    "volume_up":        CommandType.MEDIA,
    "volume_down":      CommandType.MEDIA,
    "set_volume":       CommandType.MEDIA,
    "set_timer":        CommandType.TIMER,
    "set_alarm":        CommandType.TIMER,
    "cancel_timer":     CommandType.TIMER,
    "cancel_alarm":     CommandType.TIMER,
    "navigate":         CommandType.NAVIGATION,
    "search_nearby":    CommandType.NAVIGATION,
    "call":             CommandType.COMMUNICATION,
    "send_message":     CommandType.COMMUNICATION,
    "send_email":       CommandType.COMMUNICATION,
    "read_messages":    CommandType.COMMUNICATION,
    "check_health":     CommandType.HEALTH,
    "check_steps":      CommandType.HEALTH,
    "check_heart_rate": CommandType.HEALTH,
    "start_workout":    CommandType.HEALTH,
    "stop_workout":     CommandType.HEALTH,
    "show_hologram":    CommandType.HOLOGRAM,
    "hide_hologram":    CommandType.HOLOGRAM,
    "set_brightness":   CommandType.HOLOGRAM,
    "query":            CommandType.QUERY,
    "confirm":          CommandType.CONTROL,
    "deny":             CommandType.CONTROL,
    "help":             CommandType.CONTROL,
}

# Common filler words to strip before action extraction
_FILLER_WORDS = re.compile(
    r"\b(please|hey|hi|um|uh|can you|could you|would you|i want|i'd like|i need|"
    r"i want you to|kindly|just)\b",
    re.IGNORECASE,
)

# Duration / time modifiers
_DURATION_PATTERN = re.compile(
    r"(\d+)\s*(second|sec|minute|min|hour|hr)s?", re.IGNORECASE
)
_TIME_PATTERN = re.compile(
    r"\b(\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm)|noon|midnight)\b",
    re.IGNORECASE,
)
# Level modifiers (volume, brightness)
_LEVEL_PATTERN = re.compile(
    r"\b(\d{1,3})\s*(?:percent|%|level)?\b"
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class CommandParser:
    """Parses raw utterances into structured ``ParsedCommand`` objects.

    Processing pipeline:
      1. ``normalize_command`` — lowercase, strip fillers, collapse whitespace
      2. ``extract_action`` — match against ``ACTION_PATTERNS``
      3. ``extract_target`` — match against ``TARGET_PATTERNS``
      4. ``extract_modifiers`` — extract duration, time, level modifiers
      5. ``_classify_type`` — map action to ``CommandType``
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._parse_count: int = 0
        self._unknown_count: int = 0
        # Pre-build reverse lookup: phrase -> canonical action
        self._phrase_to_action: Dict[str, str] = {}
        for action, phrases in ACTION_PATTERNS.items():
            for phrase in phrases:
                self._phrase_to_action[phrase.lower()] = action
        # Sort by length (longest first) for greedy matching
        self._sorted_phrases: List[Tuple[str, str]] = sorted(
            self._phrase_to_action.items(),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )
        logger.debug(
            "CommandParser initialised with %d action phrases.", len(self._sorted_phrases)
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def parse(self, text: str) -> ParsedCommand:
        """Parse *text* into a ``ParsedCommand``.

        Args:
            text: Raw user utterance.

        Returns:
            A ``ParsedCommand`` with the extracted fields.
        """
        with self._lock:
            self._parse_count += 1
            normalised = self.normalize_command(text)
            action = self.extract_action(normalised)
            target = self.extract_target(normalised, action)
            modifiers = self.extract_modifiers(normalised)
            command_type = self._classify_type(action, target)
            confidence = self._compute_confidence(action, target, normalised)

            if command_type == CommandType.UNKNOWN:
                self._unknown_count += 1

            cmd = ParsedCommand(
                raw_text=text,
                command_type=command_type,
                action=action,
                target=target,
                modifiers=modifiers,
                confidence=confidence,
            )
            logger.debug("Parsed: %r", cmd)
            return cmd

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def normalize_command(self, text: str) -> str:
        """Lowercase, strip filler words, and collapse whitespace."""
        lower = text.lower().strip()
        cleaned = _FILLER_WORDS.sub(" ", lower)
        # Remove punctuation except apostrophes
        cleaned = re.sub(r"[^\w\s']", " ", cleaned)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def extract_action(self, text: str) -> str:
        """Return the canonical action name from *text*, or 'unknown'."""
        for phrase, action in self._sorted_phrases:
            if phrase in text:
                logger.debug("Action matched: phrase=%r -> action=%s", phrase, action)
                return action
        # Try first-token heuristic
        first_token = text.split()[0] if text.split() else ""
        if first_token in self._phrase_to_action:
            return self._phrase_to_action[first_token]
        # Fall through to query if WH-word present
        wh_words = {"what", "when", "where", "who", "why", "how"}
        if any(w in text.split() for w in wh_words):
            return "query"
        return "unknown"

    def extract_target(self, text: str, action: str) -> str:
        """Extract the primary target entity from *text*.

        Checks ``TARGET_PATTERNS`` keyword lists and returns the matched
        target category, or attempts to extract a free-form noun phrase.
        """
        # Check keyword-based targets
        for target_cat, keywords in TARGET_PATTERNS.items():
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", text):
                    return target_cat

        # Action-based defaults when no keyword found
        action_defaults: Dict[str, str] = {
            "play":          "music",
            "pause":         "media",
            "stop":          "current",
            "navigate":      "location",
            "call":          "contact",
            "send_message":  "contact",
            "set_timer":     "timer",
            "set_alarm":     "alarm",
            "check_health":  "health",
            "show_hologram": "hologram",
            "query":         "general",
        }
        if action in action_defaults:
            return action_defaults[action]

        # Last resort: extract first noun after the action phrase
        tokens = text.split()
        if len(tokens) > 1:
            return tokens[-1]  # last token as crude fallback

        return "unknown"

    def extract_modifiers(self, text: str) -> Dict[str, str]:
        """Extract structured modifiers (duration, time, level, etc.) from *text*."""
        modifiers: Dict[str, str] = {}

        # Duration (e.g. "5 minutes", "30 seconds")
        duration_match = _DURATION_PATTERN.search(text)
        if duration_match:
            amount = duration_match.group(1)
            unit = duration_match.group(2).lower()
            # Normalise unit
            unit_map = {
                "second": "seconds", "sec": "seconds",
                "minute": "minutes", "min": "minutes",
                "hour": "hours", "hr": "hours",
            }
            normalised_unit = unit_map.get(unit, unit + "s")
            modifiers["duration"] = f"{amount} {normalised_unit}"

        # Time of day
        time_match = _TIME_PATTERN.search(text)
        if time_match:
            modifiers["time"] = time_match.group(1).strip()

        # Numeric level (volume/brightness percentage)
        level_matches = _LEVEL_PATTERN.findall(text)
        if level_matches:
            # Pick the largest numeric value found (e.g. "set volume to 75%")
            nums = [int(n) for n in level_matches if int(n) <= 100]
            if nums:
                modifiers["level"] = str(max(nums))

        # Direction modifiers
        if re.search(r"\b(left|right|straight|north|south|east|west)\b", text):
            direction = re.search(
                r"\b(left|right|straight|north|south|east|west)\b", text
            )
            if direction:
                modifiers["direction"] = direction.group(1)

        # Contact name after "call"/"text"/"message"
        contact_match = re.search(
            r"\b(?:call|text|message|phone|email)\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)\b",
            text,
        )
        if contact_match:
            modifiers["contact"] = contact_match.group(1).title()

        return modifiers

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_type(self, action: str, target: str) -> CommandType:
        """Map (action, target) to a ``CommandType``."""
        if action in _ACTION_TYPE_MAP:
            return _ACTION_TYPE_MAP[action]
        # Fallback: infer from target
        target_type_map: Dict[str, CommandType] = {
            "music":       CommandType.MEDIA,
            "podcast":     CommandType.MEDIA,
            "timer":       CommandType.TIMER,
            "alarm":       CommandType.TIMER,
            "contact":     CommandType.COMMUNICATION,
            "messages":    CommandType.COMMUNICATION,
            "navigation":  CommandType.NAVIGATION,
            "health":      CommandType.HEALTH,
            "hologram":    CommandType.HOLOGRAM,
            "settings":    CommandType.CONTROL,
        }
        if target in target_type_map:
            return target_type_map[target]
        return CommandType.UNKNOWN

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def _compute_confidence(self, action: str, target: str, text: str) -> float:
        """Heuristic confidence score (0–1) for the parse result."""
        score = 1.0
        if action == "unknown":
            score -= 0.5
        if target == "unknown":
            score -= 0.3
        # Penalise very short inputs
        if len(text.split()) < 2:
            score -= 0.1
        # Penalise very long inputs that may indicate confusion
        if len(text.split()) > 20:
            score -= 0.1
        return round(max(0.0, min(1.0, score)), 3)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_parses": self._parse_count,
                "unknown_count": self._unknown_count,
                "known_phrase_count": len(self._sorted_phrases),
                "unknown_rate": round(
                    self._unknown_count / self._parse_count if self._parse_count > 0 else 0.0,
                    3,
                ),
            }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[CommandParser] = None
_instance_lock: threading.Lock = threading.Lock()


def get_command_parser() -> CommandParser:
    """Return the process-wide singleton ``CommandParser``."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CommandParser()
    return _instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """Run basic assertion-based tests on the ``CommandParser``."""
    parser = CommandParser()

    def assert_parse(text: str, expected_action: str, expected_type: CommandType):
        result = parser.parse(text)
        assert result.action == expected_action, (
            f"FAIL: '{text}' -> action={result.action!r}, expected={expected_action!r}"
        )
        assert result.command_type == expected_type, (
            f"FAIL: '{text}' -> type={result.command_type}, expected={expected_type}"
        )
        print(f"  PASS: '{text}' -> action={result.action}, type={result.command_type.value}")

    print("Running CommandParser tests...")
    assert_parse("Play some jazz music", "play", CommandType.MEDIA)
    assert_parse("Set a timer for 10 minutes", "set_timer", CommandType.TIMER)
    assert_parse("Navigate to the nearest coffee shop", "navigate", CommandType.NAVIGATION)
    assert_parse("Call Sarah please", "call", CommandType.COMMUNICATION)
    assert_parse("What is my heart rate?", "query", CommandType.QUERY)
    assert_parse("Show me the hologram", "show_hologram", CommandType.HOLOGRAM)
    assert_parse("Set alarm for 7:30 am", "set_alarm", CommandType.TIMER)
    assert_parse("Volume up", "volume_up", CommandType.MEDIA)
    assert_parse("Check my steps today", "check_steps", CommandType.HEALTH)
    assert_parse("Stop", "stop", CommandType.CONTROL)
    print("All tests passed.\n")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    run_tests()

    parser = get_command_parser()
    print("=== Command Parser Demo ===\n")

    test_inputs = [
        "Hey, please play Bohemian Rhapsody",
        "Can you set a timer for 5 minutes",
        "Take me to the nearest hospital",
        "Call mom",
        "How many steps have I walked today?",
        "Set the brightness to 80 percent",
        "Send a message to John saying I'll be late",
        "Set an alarm for 6:30 am tomorrow",
        "Skip to the next track",
        "What time is it?",
        "Show the calendar for today",
        "Volume down please",
    ]

    for text in test_inputs:
        cmd = parser.parse(text)
        print(f"Input    : {text!r}")
        print(f"Parsed   : {cmd}")
        print()

    print(f"Stats: {parser.get_stats()}")
    print("Demo complete.")
