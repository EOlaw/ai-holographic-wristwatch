"""Interruption Management — AI Holographic Wristwatch

Detects and handles conversational interruptions — including emergency
keywords, user overrides, urgent alerts, and mid-task clarification
requests — and manages save/restore of conversation state so that
interrupted dialogues can be resumed.
"""
from __future__ import annotations

import threading
import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

try:
    from src.core.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class InterruptionType(Enum):
    USER_OVERRIDE = "user_override"
    EMERGENCY = "emergency"
    CLARIFYING_QUESTION = "clarifying_question"
    TOPIC_CHANGE = "topic_change"
    URGENT_ALERT = "urgent_alert"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InterruptionResult:
    """The outcome of handling a conversational interruption."""
    interrupt_type: InterruptionType
    saved_state: Optional[Dict]
    new_utterance: str
    should_resume: bool = True
    save_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "interrupt_type": self.interrupt_type.value,
            "has_saved_state": self.saved_state is not None,
            "save_id": self.save_id,
            "new_utterance": self.new_utterance,
            "should_resume": self.should_resume,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

EMERGENCY_KEYWORDS: List[str] = [
    "help",
    "emergency",
    "call 911",
    "call 999",
    "call 112",
    "hurting",
    "hurt",
    "injured",
    "injury",
    "bleeding",
    "unconscious",
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "fire",
    "drowning",
    "overdose",
    "stroke",
    "heart attack",
    "seizure",
    "danger",
    "attack",
    "sos",
    "mayday",
    "accident",
    "crash",
]

# Phrases/words that strongly signal the user wants to override ongoing AI speech
USER_OVERRIDE_PHRASES: List[str] = [
    "stop",
    "wait",
    "hold on",
    "pause",
    "cancel",
    "never mind",
    "forget it",
    "shut up",
    "be quiet",
    "quiet",
    "silence",
    "abort",
    "enough",
    "no no",
    "stop talking",
    "stop it",
    "that's wrong",
    "that's not right",
    "wrong",
    "incorrect",
]

# Phrases that suggest an inline clarifying question rather than a full switch
CLARIFYING_PHRASES: List[str] = [
    "what do you mean",
    "can you explain",
    "i don't understand",
    "what does that mean",
    "clarify",
    "explain that",
    "say that again",
    "repeat that",
    "come again",
    "sorry",
    "pardon",
    "huh",
    "what",
]

# Topic-change signal phrases
TOPIC_CHANGE_PHRASES: List[str] = [
    "actually",
    "change of topic",
    "different question",
    "by the way",
    "also",
    "one more thing",
    "another thing",
    "something else",
    "forget the",
    "switch to",
    "move on to",
]

# Urgency phrases for system-generated alerts
URGENT_ALERT_PHRASES: List[str] = [
    "urgent",
    "critical",
    "important alert",
    "warning",
    "low battery",
    "battery critical",
    "connection lost",
    "fall detected",
    "high heart rate",
    "irregular heartbeat",
    "medication reminder",
]

# Minimum phrase similarity ratio for a partial phrase match (0-1)
_PHRASE_MATCH_THRESHOLD: float = 0.8


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class InterruptionManager:
    """Detects, classifies, and handles conversational interruptions.

    Maintains a store of saved states so that interrupted tasks can be
    resumed after the interruption has been dealt with.
    """

    def __init__(self) -> None:
        self._saved_states: Dict[str, Dict] = {}
        self._lock: threading.RLock = threading.RLock()
        self._interruption_count: int = 0
        self._resume_count: int = 0
        self._emergency_count: int = 0
        logger.debug("InterruptionManager initialised.")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_interruption(
        self,
        utterance: str,
        current_state=None,
    ) -> Optional[InterruptionType]:
        """Classify an utterance as an interruption type or return None.

        Priority order (highest first):
          1. EMERGENCY
          2. USER_OVERRIDE
          3. CLARIFYING_QUESTION
          4. TOPIC_CHANGE
          5. URGENT_ALERT

        Args:
            utterance:     The raw user utterance.
            current_state: Optional current dialogue/turn state (unused in the
                           base heuristic but available for subclass overrides).

        Returns:
            An ``InterruptionType`` if an interruption is detected, else None.
        """
        lower = utterance.lower().strip()

        # --- EMERGENCY (highest priority) --------------------------------
        for kw in EMERGENCY_KEYWORDS:
            if kw in lower:
                logger.warning(
                    "EMERGENCY interruption detected: keyword='%s' utterance='%s'",
                    kw, utterance,
                )
                self._emergency_count += 1
                return InterruptionType.EMERGENCY

        # --- USER_OVERRIDE -----------------------------------------------
        for phrase in USER_OVERRIDE_PHRASES:
            if self._phrase_in_text(phrase, lower):
                logger.info(
                    "USER_OVERRIDE interruption detected: phrase='%s'", phrase
                )
                return InterruptionType.USER_OVERRIDE

        # --- CLARIFYING_QUESTION -----------------------------------------
        for phrase in CLARIFYING_PHRASES:
            if self._phrase_in_text(phrase, lower):
                logger.info(
                    "CLARIFYING_QUESTION interruption detected: phrase='%s'", phrase
                )
                return InterruptionType.CLARIFYING_QUESTION

        # --- TOPIC_CHANGE ------------------------------------------------
        for phrase in TOPIC_CHANGE_PHRASES:
            if self._phrase_in_text(phrase, lower):
                logger.info(
                    "TOPIC_CHANGE interruption detected: phrase='%s'", phrase
                )
                return InterruptionType.TOPIC_CHANGE

        # --- URGENT_ALERT (system-originated but included for completeness) -
        for phrase in URGENT_ALERT_PHRASES:
            if self._phrase_in_text(phrase, lower):
                logger.info(
                    "URGENT_ALERT interruption detected: phrase='%s'", phrase
                )
                return InterruptionType.URGENT_ALERT

        return None

    # ------------------------------------------------------------------
    # Handling
    # ------------------------------------------------------------------

    def handle_interruption(
        self,
        current_state,
        utterance: str,
    ) -> InterruptionResult:
        """Process an interruption by saving state and classifying the event.

        Args:
            current_state: The current dialogue state (any serialisable object
                           or dataclass — will be serialised via ``vars()`` /
                           ``to_dict()`` if available).
            utterance:     The interrupting utterance.

        Returns:
            An ``InterruptionResult`` describing how to proceed.
        """
        with self._lock:
            interrupt_type = self.detect_interruption(utterance, current_state)

            # Default to USER_OVERRIDE if detection missed but we're called directly
            if interrupt_type is None:
                interrupt_type = InterruptionType.USER_OVERRIDE
                logger.debug(
                    "handle_interruption: no interrupt type detected, "
                    "defaulting to USER_OVERRIDE."
                )

            self._interruption_count += 1

            # Save current state so it can be resumed
            save_id: Optional[str] = None
            serialised_state: Optional[Dict] = None

            if current_state is not None:
                save_id = self.save_state(current_state)
                serialised_state = self._saved_states.get(save_id)

            # Decide whether the task should be resumed after handling
            should_resume = interrupt_type not in (
                InterruptionType.EMERGENCY,
                InterruptionType.USER_OVERRIDE,
            )

            result = InterruptionResult(
                interrupt_type=interrupt_type,
                saved_state=serialised_state,
                new_utterance=utterance,
                should_resume=should_resume,
                save_id=save_id,
            )

            logger.info(
                "Interruption handled: type=%s save_id=%s should_resume=%s",
                interrupt_type.value, save_id, should_resume,
            )
            return result

    # ------------------------------------------------------------------
    # State save / restore
    # ------------------------------------------------------------------

    def save_state(self, state) -> str:
        """Serialise *state* and store it, returning a unique save ID.

        Args:
            state: Any object with a ``to_dict()`` method, or a plain dict,
                   or an object whose ``__dict__`` can be copied.

        Returns:
            A UUID string that can later be passed to ``resume_saved_state()``.
        """
        with self._lock:
            save_id = str(uuid.uuid4())

            if isinstance(state, dict):
                serialised = dict(state)
            elif hasattr(state, "to_dict") and callable(state.to_dict):
                serialised = state.to_dict()
            elif hasattr(state, "__dict__"):
                serialised = dict(vars(state))
            else:
                serialised = {"raw_state": str(state)}

            serialised["_save_id"] = save_id
            serialised["_saved_at"] = time.time()

            self._saved_states[save_id] = serialised
            logger.debug("State saved with id=%s", save_id)
            return save_id

    def resume_saved_state(self, save_id: str) -> Optional[Dict]:
        """Retrieve and remove a previously saved state snapshot.

        Args:
            save_id: The ID returned by a previous call to ``save_state()``.

        Returns:
            The saved state dict, or None if the ID was not found.
        """
        with self._lock:
            state = self._saved_states.pop(save_id, None)
            if state is not None:
                self._resume_count += 1
                logger.info("State resumed: save_id=%s", save_id)
            else:
                logger.warning("resume_saved_state: save_id=%s not found.", save_id)
            return state

    def list_saved_states(self) -> List[str]:
        """Return a list of all currently stored save IDs."""
        with self._lock:
            return list(self._saved_states.keys())

    def discard_saved_state(self, save_id: str) -> bool:
        """Discard a saved state without resuming it.

        Returns True if the state was found and removed, False otherwise.
        """
        with self._lock:
            if save_id in self._saved_states:
                del self._saved_states[save_id]
                logger.debug("Saved state discarded: save_id=%s", save_id)
                return True
            return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return cumulative interruption statistics."""
        with self._lock:
            return {
                "total_interruptions": self._interruption_count,
                "emergency_count": self._emergency_count,
                "total_resumes": self._resume_count,
                "pending_saved_states": len(self._saved_states),
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _phrase_in_text(phrase: str, text: str) -> bool:
        """Return True if *phrase* is a substring of *text* (case-insensitive)."""
        return phrase in text


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[InterruptionManager] = None
_instance_lock: threading.Lock = threading.Lock()


def get_interruption_manager() -> InterruptionManager:
    """Return the process-wide singleton ``InterruptionManager``."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = InterruptionManager()
    return _instance


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    mgr = get_interruption_manager()
    print("=== Interruption Manager Demo ===\n")

    test_utterances = [
        "stop talking",
        "call 911, I need help!",
        "what do you mean by that?",
        "actually, by the way, let's talk about something else",
        "just a normal question about the weather",
    ]

    fake_state = {
        "intent": "set_timer",
        "duration": "5 minutes",
        "step": "confirming",
    }

    for utt in test_utterances:
        interrupt_type = mgr.detect_interruption(utt)
        print(f"Utterance : {utt!r}")
        print(f"  Detected: {interrupt_type.value if interrupt_type else 'None'}")

        if interrupt_type:
            result = mgr.handle_interruption(fake_state, utt)
            print(f"  Result   : type={result.interrupt_type.value} "
                  f"save_id={result.save_id} resume={result.should_resume}")
            if result.save_id:
                restored = mgr.resume_saved_state(result.save_id)
                print(f"  Restored : {restored}")
        print()

    print(f"Stats: {mgr.get_stats()}")
    print("\nDemo complete.")
