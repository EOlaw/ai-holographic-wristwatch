"""Turn Taking Management — AI Holographic Wristwatch

Coordinates conversational turn-taking between the user and the AI system,
detecting end-of-utterance events, managing overlap/interruptions, and
providing metrics on speaking durations and silence gaps.
"""
from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

try:
    from src.core.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TurnState(Enum):
    """The current state of the conversational floor."""
    USER_SPEAKING = "user_speaking"
    AI_SPEAKING = "ai_speaking"
    SILENCE = "silence"
    TRANSITION = "transition"
    OVERLAP = "overlap"


# ---------------------------------------------------------------------------
# Internal metrics tracking
# ---------------------------------------------------------------------------

@dataclass
class _TurnMetrics:
    """Running metrics for a single turn."""
    state: TurnState
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    peak_energy_db: float = -60.0

    def duration(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return max(0.0, end - self.start_time)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class TurnTakingManager:
    """Thread-safe manager for conversational turn-taking.

    Uses acoustic energy and duration heuristics to detect end-of-utterance
    events and decide whether the AI should interrupt the user.
    """

    # Thresholds
    _SILENCE_THRESHOLD_SECS: float = 0.8   # silence gap before floor releases
    _OVERLAP_ENERGY_DB: float = -30.0       # energy above which user is speaking
    _EOT_ENERGY_FLOOR_DB: float = -45.0     # energy below which we count silence
    _MIN_UTTERANCE_SECS: float = 0.25       # minimum valid utterance length
    _INTERRUPT_BASE_URGENCY: int = 7        # minimum urgency to interrupt user

    def __init__(self) -> None:
        self._state: TurnState = TurnState.SILENCE
        self._lock: threading.RLock = threading.RLock()
        self._silence_threshold: float = self._SILENCE_THRESHOLD_SECS
        self._turn_start_time: float = time.time()
        self._current_metrics: Optional[_TurnMetrics] = None
        self._silence_start: Optional[float] = None

        # Cumulative stats
        self._user_turn_count: int = 0
        self._ai_turn_count: int = 0
        self._total_user_speaking_secs: float = 0.0
        self._total_ai_speaking_secs: float = 0.0
        self._total_silence_secs: float = 0.0
        self._overlap_count: int = 0

        logger.debug("TurnTakingManager initialised.")

    # ------------------------------------------------------------------
    # User turn signals
    # ------------------------------------------------------------------

    def signal_user_turn_start(self) -> None:
        """Notify the manager that the user has begun speaking."""
        with self._lock:
            prev_state = self._state
            if prev_state == TurnState.AI_SPEAKING:
                self._state = TurnState.OVERLAP
                self._overlap_count += 1
                logger.info("Turn overlap detected — user interrupted AI.")
            else:
                self._state = TurnState.USER_SPEAKING
            self._current_metrics = _TurnMetrics(state=self._state)
            self._silence_start = None
            self._user_turn_count += 1
            logger.debug(
                "User turn start. Previous state: %s -> New state: %s",
                prev_state.value, self._state.value,
            )

    def signal_user_turn_end(self) -> float:
        """Notify the manager that the user has stopped speaking.

        Returns the duration of the user turn in seconds.
        """
        with self._lock:
            if self._state not in (TurnState.USER_SPEAKING, TurnState.OVERLAP):
                logger.warning(
                    "signal_user_turn_end called from unexpected state: %s",
                    self._state.value,
                )
            duration = 0.0
            if self._current_metrics is not None:
                self._current_metrics.end_time = time.time()
                duration = self._current_metrics.duration()
                self._total_user_speaking_secs += duration

            self._state = TurnState.TRANSITION
            self._silence_start = time.time()
            logger.debug(
                "User turn ended. Duration=%.2fs. State -> TRANSITION.", duration
            )
            return duration

    # ------------------------------------------------------------------
    # AI turn signals
    # ------------------------------------------------------------------

    def signal_ai_turn_start(self) -> None:
        """Notify the manager that the AI has begun speaking."""
        with self._lock:
            self._state = TurnState.AI_SPEAKING
            self._current_metrics = _TurnMetrics(state=TurnState.AI_SPEAKING)
            self._silence_start = None
            self._ai_turn_count += 1
            logger.debug("AI turn started.")

    def signal_ai_turn_end(self) -> float:
        """Notify the manager that the AI has finished speaking.

        Returns the duration of the AI turn in seconds.
        """
        with self._lock:
            if self._state != TurnState.AI_SPEAKING:
                logger.warning(
                    "signal_ai_turn_end called from unexpected state: %s",
                    self._state.value,
                )
            duration = 0.0
            if self._current_metrics is not None:
                self._current_metrics.end_time = time.time()
                duration = self._current_metrics.duration()
                self._total_ai_speaking_secs += duration

            self._state = TurnState.SILENCE
            self._silence_start = time.time()
            logger.debug(
                "AI turn ended. Duration=%.2fs. State -> SILENCE.", duration
            )
            return duration

    # ------------------------------------------------------------------
    # End-of-utterance detection
    # ------------------------------------------------------------------

    def detect_end_of_utterance(
        self, energy_db: float, duration_secs: float
    ) -> bool:
        """Determine whether the user has finished their utterance.

        Uses a simple energy-threshold + silence-duration heuristic.

        Args:
            energy_db:     Current audio frame energy in dBFS (negative float).
            duration_secs: How long this low-energy segment has persisted.

        Returns:
            True if the end-of-utterance criterion is met, False otherwise.
        """
        with self._lock:
            if self._state not in (TurnState.USER_SPEAKING, TurnState.OVERLAP):
                return False

            # Update peak energy
            if self._current_metrics is not None:
                if energy_db > self._current_metrics.peak_energy_db:
                    self._current_metrics.peak_energy_db = energy_db

            # EOT heuristic: energy below floor AND silence duration exceeded
            below_floor = energy_db <= self._EOT_ENERGY_FLOOR_DB
            long_enough = duration_secs >= self._silence_threshold

            if below_floor and long_enough:
                # Confirm utterance was not too short
                utterance_so_far = (
                    self._current_metrics.duration()
                    if self._current_metrics else 0.0
                )
                if utterance_so_far >= self._MIN_UTTERANCE_SECS:
                    logger.debug(
                        "EOT detected: energy=%.1fdB silence=%.2fs utterance=%.2fs",
                        energy_db, duration_secs, utterance_so_far,
                    )
                    return True

            return False

    # ------------------------------------------------------------------
    # Interruption decision
    # ------------------------------------------------------------------

    def should_interrupt(self, urgency_level: int) -> bool:
        """Decide whether the AI should interrupt the user.

        Args:
            urgency_level: Integer 0–10 indicating how urgent the AI's message
                           is (0 = not urgent, 10 = emergency).

        Returns:
            True if the AI should interrupt the user's current turn.
        """
        with self._lock:
            clamped = max(0, min(10, urgency_level))
            if self._state == TurnState.SILENCE:
                # Floor is free — no need to interrupt
                return False
            if self._state == TurnState.USER_SPEAKING:
                # Only interrupt if urgency is high enough
                decision = clamped >= self._INTERRUPT_BASE_URGENCY
                if decision:
                    logger.info(
                        "AI interrupt decision: YES (urgency=%d)", clamped
                    )
                else:
                    logger.debug(
                        "AI interrupt decision: NO (urgency=%d < threshold=%d)",
                        clamped, self._INTERRUPT_BASE_URGENCY,
                    )
                return decision
            if self._state == TurnState.AI_SPEAKING:
                # Already speaking — nothing to interrupt
                return False
            # TRANSITION / OVERLAP — allow if urgency is moderate+
            return clamped >= 5

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_current_state(self) -> TurnState:
        """Return the current turn state."""
        with self._lock:
            return self._state

    def get_silence_duration(self) -> float:
        """Return how long (seconds) the current silence has lasted, or 0."""
        with self._lock:
            if self._silence_start is None:
                return 0.0
            if self._state in (TurnState.USER_SPEAKING, TurnState.AI_SPEAKING):
                return 0.0
            return time.time() - self._silence_start

    def set_silence_threshold(self, seconds: float) -> None:
        """Override the default silence threshold for EOT detection."""
        with self._lock:
            self._silence_threshold = max(0.1, seconds)
            logger.info("Silence threshold updated to %.2fs", self._silence_threshold)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_turn_stats(self) -> Dict:
        """Return cumulative turn-taking statistics."""
        with self._lock:
            total = (
                self._total_user_speaking_secs
                + self._total_ai_speaking_secs
                + self._total_silence_secs
            )
            return {
                "current_state": self._state.value,
                "user_turn_count": self._user_turn_count,
                "ai_turn_count": self._ai_turn_count,
                "overlap_count": self._overlap_count,
                "total_user_speaking_secs": round(self._total_user_speaking_secs, 3),
                "total_ai_speaking_secs": round(self._total_ai_speaking_secs, 3),
                "total_silence_secs": round(self._total_silence_secs, 3),
                "floor_utilisation_pct": round(
                    100.0 * (self._total_user_speaking_secs + self._total_ai_speaking_secs) / total
                    if total > 0 else 0.0,
                    1,
                ),
                "silence_threshold_secs": self._silence_threshold,
                "current_silence_secs": round(self.get_silence_duration(), 3),
            }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[TurnTakingManager] = None
_instance_lock: threading.Lock = threading.Lock()


def get_turn_taking_manager() -> TurnTakingManager:
    """Return the process-wide singleton ``TurnTakingManager``."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = TurnTakingManager()
    return _instance


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    mgr = get_turn_taking_manager()
    print("=== Turn Taking Manager Demo ===\n")

    print(f"Initial state: {mgr.get_current_state().value}")

    # Simulate user speaking
    mgr.signal_user_turn_start()
    print(f"After user starts: {mgr.get_current_state().value}")

    # Simulate silence detection
    eot = mgr.detect_end_of_utterance(energy_db=-50.0, duration_secs=0.9)
    print(f"EOT detected (should be True): {eot}")

    duration = mgr.signal_user_turn_end()
    print(f"User turn ended. State: {mgr.get_current_state().value}, duration: {duration:.3f}s")

    # AI speaks
    mgr.signal_ai_turn_start()
    print(f"AI speaking: {mgr.get_current_state().value}")

    # Try to interrupt with various urgency levels
    print(f"Should interrupt urgency=5: {mgr.should_interrupt(5)}")
    print(f"Should interrupt urgency=9: {mgr.should_interrupt(9)}")

    mgr.signal_ai_turn_end()
    print(f"AI done. State: {mgr.get_current_state().value}")

    print(f"\nStats: {mgr.get_turn_stats()}")
    print("\nDemo complete.")
