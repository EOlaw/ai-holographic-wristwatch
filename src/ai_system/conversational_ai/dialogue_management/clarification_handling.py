"""Clarification Handling — AI Holographic Wristwatch

Manages the lifecycle of slot-filling clarification requests during
dialogue — detecting when required information is missing, generating
natural clarification questions, and processing user responses to fill
those slots.
"""
from __future__ import annotations

import re
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

try:
    from src.core.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ClarificationState(Enum):
    NONE = "none"
    PENDING = "pending"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClarificationRequest:
    """Tracks the lifecycle of a single clarification exchange."""
    intent_name: str
    missing_slots: List[str]
    question: str
    attempts: int = 0
    state: ClarificationState = ClarificationState.PENDING
    filled_slots: Dict[str, str] = field(default_factory=dict)

    def is_fully_resolved(self) -> bool:
        return not self.missing_slots

    def to_dict(self) -> Dict:
        return {
            "intent_name": self.intent_name,
            "missing_slots": list(self.missing_slots),
            "question": self.question,
            "attempts": self.attempts,
            "state": self.state.value,
            "filled_slots": dict(self.filled_slots),
        }


# ---------------------------------------------------------------------------
# Slot-filling patterns
# ---------------------------------------------------------------------------

# Regex patterns used to extract slot values from a free-text response.
# Each entry maps slot_name -> (regex_pattern, value_group_index)
_SLOT_EXTRACTION_PATTERNS: Dict[str, List[tuple]] = {
    "duration": [
        (r"(\d+)\s*(minute|min|second|sec|hour|hr)s?", 0),
        (r"(a\s+minute|a\s+second|a\s+hour)", 0),
    ],
    "contact_name": [
        (r"(?:call|message|text|email)\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)", 1),
        (r"^([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)$", 1),
    ],
    "location": [
        (r"(?:to|at|near|around|in)\s+([A-Za-z0-9 ,]+?)(?:\.|$)", 1),
        (r"^([A-Za-z0-9 ,]+(?:street|avenue|road|blvd|lane|drive|place|square)[A-Za-z0-9 ,]*)$", 1),
    ],
    "time": [
        (r"(\d{1,2}:\d{2}\s*(?:am|pm)?)", 1),
        (r"(\d{1,2}\s*(?:am|pm))", 1),
        (r"(noon|midnight|morning|afternoon|evening|night)", 1),
    ],
    "date": [
        (r"(today|tomorrow|yesterday)", 1),
        (r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", 1),
        (r"(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)", 1),
    ],
    "song_name": [
        (r"(?:play|put on)\s+(.+?)(?:\s+by\s+|\s*$)", 1),
        (r"^(.+)$", 1),
    ],
    "brightness_level": [
        (r"(\d{1,3})\s*(?:percent|%)?", 1),
        (r"(low|medium|high|full|off|dim|bright)", 1),
    ],
    "volume_level": [
        (r"(\d{1,3})\s*(?:percent|%)?", 1),
        (r"(low|medium|high|full|off|quiet|loud|mute)", 1),
    ],
    "number": [
        (r"(\d+)", 1),
        (r"(one|two|three|four|five|six|seven|eight|nine|ten)", 1),
    ],
    "search_query": [
        (r"(?:search|find|look up|look for)\s+(.+)", 1),
        (r"^(.+)$", 1),
    ],
}

# Question templates for each slot name
_SLOT_QUESTIONS: Dict[str, str] = {
    "duration":        "For how long? (e.g. '5 minutes', '30 seconds')",
    "contact_name":    "Who would you like to contact?",
    "location":        "Where would you like to go?",
    "time":            "At what time? (e.g. '3:30 pm', 'noon')",
    "date":            "Which date? (e.g. 'today', 'tomorrow', 'Friday')",
    "song_name":       "What song or artist would you like to play?",
    "brightness_level":"What brightness level? (e.g. '50%', 'low', 'high')",
    "volume_level":    "What volume level? (e.g. '30%', 'quiet', 'loud')",
    "number":          "What number would you like?",
    "search_query":    "What would you like me to search for?",
}

# Generic fallback question if the slot has no template
_GENERIC_QUESTION = "Could you provide more details about '{slot}'?"

# Per-intent required slots
_INTENT_REQUIRED_SLOTS: Dict[str, List[str]] = {
    "set_timer":          ["duration"],
    "set_alarm":          ["time", "date"],
    "call_contact":       ["contact_name"],
    "send_message":       ["contact_name"],
    "navigate_to":        ["location"],
    "play_music":         ["song_name"],
    "search_web":         ["search_query"],
    "set_brightness":     ["brightness_level"],
    "set_volume":         ["volume_level"],
    "set_reminder":       ["duration", "search_query"],
}

# Max clarification attempts before abandoning
_MAX_ATTEMPTS: int = 3


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class ClarificationHandler:
    """Manages multi-turn slot-filling clarification dialogues.

    Usage pattern::

        handler = get_clarification_handler()

        if handler.needs_clarification("set_timer", filled, required_map):
            question = handler.generate_clarification_question("set_timer", "duration")
            # ... ask user ...
            filled = handler.process_response(user_reply, handler.get_current_request())
    """

    def __init__(self) -> None:
        self._current_request: Optional[ClarificationRequest] = None
        self._lock: threading.RLock = threading.RLock()
        self._slot_questions: Dict[str, str] = _SLOT_QUESTIONS
        self._total_resolved: int = 0
        self._total_abandoned: int = 0
        logger.debug("ClarificationHandler initialised.")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def needs_clarification(
        self,
        intent_name: str,
        slots: Dict,
        required_slots: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        """Return True if any required slot for *intent_name* is missing.

        Args:
            intent_name:    The recognised intent.
            slots:          Slots already extracted (may be empty / partial).
            required_slots: Optional override mapping intent -> required slots.
                            Falls back to the built-in ``_INTENT_REQUIRED_SLOTS``.

        Returns:
            True if clarification is needed, False otherwise.
        """
        with self._lock:
            req_map = required_slots if required_slots is not None else _INTENT_REQUIRED_SLOTS
            required = req_map.get(intent_name, [])
            missing = [s for s in required if not slots.get(s)]
            if missing:
                logger.info(
                    "Clarification needed for intent='%s', missing slots: %s",
                    intent_name, missing,
                )
                # Initialise a request so generate_clarification_question has context
                question = self.generate_clarification_question(intent_name, missing[0])
                self._current_request = ClarificationRequest(
                    intent_name=intent_name,
                    missing_slots=missing,
                    question=question,
                )
                return True
            return False

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------

    def generate_clarification_question(
        self, intent_name: str, missing_slot: str
    ) -> str:
        """Return a natural-language question to elicit the *missing_slot* value.

        Uses ``_slot_questions`` templates, falling back to a generic phrasing.
        """
        template = self._slot_questions.get(missing_slot)
        if template:
            question = template
        else:
            question = _GENERIC_QUESTION.format(slot=missing_slot.replace("_", " "))

        logger.debug(
            "Clarification question for slot '%s': %s", missing_slot, question
        )
        return question

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    def process_response(
        self, response: str, pending: Optional[ClarificationRequest] = None
    ) -> Dict[str, str]:
        """Attempt to extract slot values from the user's *response*.

        Args:
            response: Raw text of the user's reply.
            pending:  The in-flight ``ClarificationRequest`` (defaults to
                      ``self._current_request`` if not supplied).

        Returns:
            A dict mapping slot names to extracted values for any slots
            successfully filled. Also updates ``pending.filled_slots`` and
            removes resolved slots from ``pending.missing_slots``.
        """
        with self._lock:
            req = pending if pending is not None else self._current_request
            if req is None:
                logger.warning("process_response called with no pending clarification.")
                return {}
            if req.state == ClarificationState.ABANDONED:
                logger.warning("process_response called on an abandoned clarification.")
                return {}

            req.attempts += 1
            newly_filled: Dict[str, str] = {}
            still_missing: List[str] = []

            for slot in list(req.missing_slots):
                value = self._extract_slot_value(response, slot)
                if value:
                    req.filled_slots[slot] = value
                    newly_filled[slot] = value
                    logger.info(
                        "Slot '%s' filled with value '%s'", slot, value
                    )
                else:
                    still_missing.append(slot)

            req.missing_slots = still_missing

            if not still_missing:
                req.state = ClarificationState.RESOLVED
                self._total_resolved += 1
                logger.info(
                    "Clarification resolved for intent '%s'. Filled: %s",
                    req.intent_name, req.filled_slots,
                )
            elif req.attempts >= _MAX_ATTEMPTS:
                req.state = ClarificationState.ABANDONED
                self._total_abandoned += 1
                logger.warning(
                    "Clarification abandoned for intent '%s' after %d attempts.",
                    req.intent_name, req.attempts,
                )

            return newly_filled

    # ------------------------------------------------------------------
    # Abandonment
    # ------------------------------------------------------------------

    def abandon_clarification(self) -> None:
        """Mark the current clarification request as abandoned."""
        with self._lock:
            if self._current_request is None:
                return
            self._current_request.state = ClarificationState.ABANDONED
            self._total_abandoned += 1
            logger.info(
                "Clarification abandoned for intent '%s'.",
                self._current_request.intent_name,
            )
            self._current_request = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_current_request(self) -> Optional[ClarificationRequest]:
        """Return the current (possibly resolved/pending) request, or None."""
        with self._lock:
            return self._current_request

    def get_next_question(self) -> Optional[str]:
        """If the current request is still pending, return the next question."""
        with self._lock:
            req = self._current_request
            if req is None or req.state != ClarificationState.PENDING:
                return None
            if not req.missing_slots:
                return None
            return self.generate_clarification_question(
                req.intent_name, req.missing_slots[0]
            )

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_resolved": self._total_resolved,
                "total_abandoned": self._total_abandoned,
                "has_pending": (
                    self._current_request is not None
                    and self._current_request.state == ClarificationState.PENDING
                ),
                "current_request": (
                    self._current_request.to_dict()
                    if self._current_request else None
                ),
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_slot_value(self, text: str, slot: str) -> Optional[str]:
        """Try to extract a value for *slot* from *text* using regex patterns."""
        patterns = _SLOT_EXTRACTION_PATTERNS.get(slot, [])
        lower = text.lower().strip()
        for pattern, group_idx in patterns:
            match = re.search(pattern, lower)
            if match:
                try:
                    value = match.group(group_idx) if group_idx > 0 else match.group(0)
                    return value.strip()
                except IndexError:
                    value = match.group(0)
                    return value.strip()
        # Last resort: if single-token response and slot is open-ended
        if slot in ("contact_name", "song_name", "search_query", "location"):
            cleaned = text.strip()
            if cleaned and len(cleaned.split()) <= 6:
                return cleaned
        return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[ClarificationHandler] = None
_instance_lock: threading.Lock = threading.Lock()


def get_clarification_handler() -> ClarificationHandler:
    """Return the process-wide singleton ``ClarificationHandler``."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ClarificationHandler()
    return _instance


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    handler = get_clarification_handler()
    print("=== Clarification Handler Demo ===\n")

    # Scenario 1: set_timer with no duration
    slots: Dict = {}
    needs = handler.needs_clarification("set_timer", slots)
    print(f"Needs clarification (set_timer, no slots): {needs}")
    print(f"Question: {handler.get_next_question()}\n")

    # User replies
    filled = handler.process_response("5 minutes")
    print(f"Filled slots: {filled}")
    print(f"Request state: {handler.get_current_request().state.value}\n")

    # Scenario 2: call_contact with no name
    handler2 = ClarificationHandler()
    slots2: Dict = {}
    handler2.needs_clarification("call_contact", slots2)
    print(f"Question (call_contact): {handler2.get_next_question()}")
    filled2 = handler2.process_response("call Sarah")
    print(f"Filled slots: {filled2}")
    print(f"Request state: {handler2.get_current_request().state.value}\n")

    print(f"Stats: {handler.get_stats()}")
    print("\nDemo complete.")
