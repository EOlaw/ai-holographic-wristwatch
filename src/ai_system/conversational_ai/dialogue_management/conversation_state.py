"""Conversation State — AI Holographic Wristwatch

Manages the state machine for ongoing conversations, tracking transitions
between dialogue phases (IDLE, GREETING, ACTIVE, CLARIFYING, etc.) and
maintaining per-session history for context-aware responses.
"""
from __future__ import annotations

import threading
import time
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Deque

try:
    from src.core.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DialogueState(Enum):
    """High-level states of an ongoing dialogue session."""
    IDLE = "idle"
    GREETING = "greeting"
    ACTIVE = "active"
    CLARIFYING = "clarifying"
    CONFIRMING = "confirming"
    TASK_EXECUTING = "task_executing"
    ENDING = "ending"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    """Snapshot of conversation state at a given point in time."""
    state: DialogueState = DialogueState.IDLE
    turn_count: int = 0
    last_intent: str = ""
    user_id: str = ""
    session_id: str = ""
    start_time: float = field(default_factory=time.time)
    context: Dict = field(default_factory=dict)
    error_msg: Optional[str] = None

    def age_seconds(self) -> float:
        """Return how many seconds this session has been running."""
        return time.time() - self.start_time

    def to_dict(self) -> Dict:
        """Serialise to a plain dictionary (for logging / persistence)."""
        return {
            "state": self.state.value,
            "turn_count": self.turn_count,
            "last_intent": self.last_intent,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "age_seconds": self.age_seconds(),
            "context": dict(self.context),
            "error_msg": self.error_msg,
        }


# ---------------------------------------------------------------------------
# State-machine transitions
# ---------------------------------------------------------------------------

# Mapping:  current_state  ->  { event_name -> next_state }
_TRANSITIONS: Dict[DialogueState, Dict[str, DialogueState]] = {
    DialogueState.IDLE: {
        "session_start": DialogueState.GREETING,
        "error": DialogueState.ERROR,
    },
    DialogueState.GREETING: {
        "intent_received": DialogueState.ACTIVE,
        "goodbye": DialogueState.ENDING,
        "error": DialogueState.ERROR,
    },
    DialogueState.ACTIVE: {
        "needs_clarification": DialogueState.CLARIFYING,
        "needs_confirmation": DialogueState.CONFIRMING,
        "execute_task": DialogueState.TASK_EXECUTING,
        "goodbye": DialogueState.ENDING,
        "error": DialogueState.ERROR,
    },
    DialogueState.CLARIFYING: {
        "clarification_received": DialogueState.ACTIVE,
        "clarification_abandoned": DialogueState.ACTIVE,
        "goodbye": DialogueState.ENDING,
        "error": DialogueState.ERROR,
    },
    DialogueState.CONFIRMING: {
        "confirmed": DialogueState.TASK_EXECUTING,
        "denied": DialogueState.ACTIVE,
        "goodbye": DialogueState.ENDING,
        "error": DialogueState.ERROR,
    },
    DialogueState.TASK_EXECUTING: {
        "task_complete": DialogueState.ACTIVE,
        "task_failed": DialogueState.ERROR,
        "goodbye": DialogueState.ENDING,
        "error": DialogueState.ERROR,
    },
    DialogueState.ENDING: {
        "session_end": DialogueState.IDLE,
    },
    DialogueState.ERROR: {
        "recover": DialogueState.ACTIVE,
        "session_end": DialogueState.IDLE,
    },
}


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class ConversationStateManager:
    """Thread-safe state machine for tracking conversation lifecycle.

    Maintains a rolling history of ``ConversationState`` snapshots and
    exposes helpers for the rest of the dialogue system to query and
    advance state.
    """

    def __init__(self) -> None:
        self._state: Optional[ConversationState] = None
        self._history: Deque[ConversationState] = deque(maxlen=100)
        self._lock: threading.RLock = threading.RLock()
        self._transitions: Dict[DialogueState, Dict[str, DialogueState]] = _TRANSITIONS
        self._total_sessions: int = 0
        self._total_turns: int = 0
        logger.debug("ConversationStateManager initialised.")

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def begin_session(self, user_id: str = "") -> ConversationState:
        """Start a new conversation session and transition to GREETING."""
        with self._lock:
            if self._state is not None and self._state.state not in (
                DialogueState.IDLE, DialogueState.ENDING, DialogueState.ERROR
            ):
                logger.warning(
                    "begin_session called while session %s is still active; "
                    "archiving previous session.",
                    self._state.session_id,
                )
                self._history.append(self._state)

            session_id = str(uuid.uuid4())
            self._state = ConversationState(
                state=DialogueState.IDLE,
                user_id=user_id,
                session_id=session_id,
                start_time=time.time(),
            )
            self._total_sessions += 1
            # Immediately transition to GREETING
            self._apply_transition("session_start")
            logger.info(
                "Session started: session_id=%s user_id=%s", session_id, user_id
            )
            return self._state

    def end_session(self) -> Optional[ConversationState]:
        """Gracefully end the current session and return the final state."""
        with self._lock:
            if self._state is None:
                logger.warning("end_session called with no active session.")
                return None
            self._apply_transition("goodbye")
            self._apply_transition("session_end")
            self._total_turns += self._state.turn_count
            final = self._state
            self._history.append(final)
            self._state = None
            logger.info(
                "Session ended: session_id=%s turns=%d",
                final.session_id,
                final.turn_count,
            )
            return final

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def transition(self, event: str) -> DialogueState:
        """Apply *event* to the current state machine and return the new state.

        Raises ``ValueError`` if the event is not valid from the current state.
        """
        with self._lock:
            if self._state is None:
                raise RuntimeError("No active session. Call begin_session() first.")
            return self._apply_transition(event)

    def _apply_transition(self, event: str) -> DialogueState:
        """Internal (lock must already be held) transition helper."""
        current = self._state.state
        allowed = self._transitions.get(current, {})
        if event not in allowed:
            logger.error(
                "Invalid event '%s' from state %s. Allowed: %s",
                event, current.value, list(allowed.keys()),
            )
            raise ValueError(
                f"Event '{event}' is not valid from state '{current.value}'. "
                f"Allowed events: {list(allowed.keys())}"
            )
        next_state = allowed[event]
        logger.debug(
            "Transition: %s -[%s]-> %s (session=%s)",
            current.value, event, next_state.value, self._state.session_id,
        )
        self._state.state = next_state
        return next_state

    # ------------------------------------------------------------------
    # Accessors / mutators
    # ------------------------------------------------------------------

    def get_current_state(self) -> Optional[ConversationState]:
        """Return a *copy* of the current conversation state, or None."""
        with self._lock:
            if self._state is None:
                return None
            # Return a shallow copy so callers cannot mutate internals
            import copy
            return copy.copy(self._state)

    def increment_turn(self) -> int:
        """Increment turn counter and return the new count."""
        with self._lock:
            if self._state is None:
                raise RuntimeError("No active session.")
            self._state.turn_count += 1
            logger.debug("Turn incremented to %d", self._state.turn_count)
            return self._state.turn_count

    def set_last_intent(self, intent_name: str) -> None:
        """Record the most recently recognised intent."""
        with self._lock:
            if self._state is None:
                raise RuntimeError("No active session.")
            self._state.last_intent = intent_name
            logger.debug("Last intent set to '%s'", intent_name)

    def update_context(self, key: str, value) -> None:
        """Store an arbitrary key/value pair in the session context."""
        with self._lock:
            if self._state is None:
                raise RuntimeError("No active session.")
            self._state.context[key] = value

    def set_error(self, message: str) -> None:
        """Record an error message and transition to ERROR state."""
        with self._lock:
            if self._state is None:
                raise RuntimeError("No active session.")
            self._state.error_msg = message
            try:
                self._apply_transition("error")
            except ValueError:
                pass  # Already in ERROR or incompatible state
            logger.error("Session error recorded: %s", message)

    # ------------------------------------------------------------------
    # Summaries and stats
    # ------------------------------------------------------------------

    def get_conversation_summary(self) -> Dict:
        """Return a human-readable summary of the current session."""
        with self._lock:
            if self._state is None:
                return {"status": "no_active_session"}
            return {
                "session_id": self._state.session_id,
                "user_id": self._state.user_id,
                "current_state": self._state.state.value,
                "turn_count": self._state.turn_count,
                "last_intent": self._state.last_intent,
                "age_seconds": round(self._state.age_seconds(), 2),
                "context_keys": list(self._state.context.keys()),
                "error_msg": self._state.error_msg,
            }

    def get_stats(self) -> Dict:
        """Return cumulative statistics across all sessions."""
        with self._lock:
            archived_turns = sum(s.turn_count for s in self._history)
            current_turns = self._state.turn_count if self._state else 0
            return {
                "total_sessions": self._total_sessions,
                "total_turns": self._total_turns + current_turns,
                "archived_session_count": len(self._history),
                "has_active_session": self._state is not None,
                "current_session_id": (
                    self._state.session_id if self._state else None
                ),
                "archived_turns": archived_turns,
            }

    def get_history(self) -> List[Dict]:
        """Return serialised snapshots of all archived sessions."""
        with self._lock:
            return [s.to_dict() for s in self._history]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_manager_instance: Optional[ConversationStateManager] = None
_manager_lock: threading.Lock = threading.Lock()


def get_conversation_state_manager() -> ConversationStateManager:
    """Return the process-wide singleton ``ConversationStateManager``."""
    global _manager_instance
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = ConversationStateManager()
    return _manager_instance


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    mgr = get_conversation_state_manager()

    print("=== Conversation State Manager Demo ===\n")

    # Start a session
    state = mgr.begin_session(user_id="demo_user_001")
    print(f"Session started. State: {state.state.value}")
    print(f"Session ID: {state.session_id}\n")

    # Simulate a few turns
    mgr.transition("intent_received")
    mgr.set_last_intent("set_timer")
    mgr.increment_turn()
    print(f"After intent_received: {mgr.get_current_state().state.value}")

    mgr.transition("needs_clarification")
    mgr.increment_turn()
    print(f"After needs_clarification: {mgr.get_current_state().state.value}")

    mgr.transition("clarification_received")
    mgr.increment_turn()
    print(f"After clarification_received: {mgr.get_current_state().state.value}")

    mgr.transition("execute_task")
    mgr.increment_turn()
    print(f"After execute_task: {mgr.get_current_state().state.value}")

    mgr.transition("task_complete")
    mgr.increment_turn()
    print(f"After task_complete: {mgr.get_current_state().state.value}")

    print(f"\nSummary: {mgr.get_conversation_summary()}")

    final = mgr.end_session()
    print(f"\nSession ended. Final state: {final.state.value}")
    print(f"Stats: {mgr.get_stats()}")

    # Test invalid transition
    try:
        mgr2 = ConversationStateManager()
        mgr2.begin_session("user2")
        mgr2.transition("task_complete")  # invalid from GREETING
    except ValueError as exc:
        print(f"\nExpected ValueError caught: {exc}")

    print("\nDemo complete.")
