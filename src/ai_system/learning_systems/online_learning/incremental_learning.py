"""Incremental Learning module — AI Holographic Wristwatch.

Implements an online learning loop that tracks per-example outcomes,
maintains a sliding accuracy window, and detects concept drift.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LearningResult:
    """Summary of a learning update batch."""

    examples_seen: int
    recent_accuracy: float
    drift_detected: bool
    model_updated: bool
    timestamp: float

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LearningResult(seen={self.examples_seen}, "
            f"accuracy={self.recent_accuracy:.3f}, "
            f"drift={self.drift_detected}, updated={self.model_updated})"
        )


class IncrementalLearner:
    """Online learner with concept-drift detection.

    The "model" here is a lightweight running accuracy estimate built from
    per-example (input, label, correct?) tuples stored in a sliding window.
    The system detects drift by comparing the accuracy of the most-recent
    half of the window against the older half.  When drift exceeds
    ``_drift_threshold``, the window is reset and the model re-calibrates.

    In a real deployment this class would delegate heavy inference to a
    proper ML back-end; the structure shown here is compatible with that.
    """

    def __init__(self) -> None:
        self._examples_seen: int = 0
        self._correct: int = 0
        # Each element: (is_correct: bool, weight: float)
        self._window: Deque[Tuple[bool, float]] = deque(maxlen=100)
        self._lock: threading.Lock = threading.Lock()
        self._drift_threshold: float = 0.2
        self._last_update: float = 0.0
        logger.debug("IncrementalLearner initialised")

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn_from_example(
        self,
        input_data: Dict,
        label: str,
        weight: float = 1.0,
    ) -> None:
        """Incorporate a single labelled example into the window.

        Parameters
        ----------
        input_data:
            Feature dict for the example.
        label:
            Ground-truth label string.
        weight:
            Importance multiplier for this example (default 1.0).
        """
        # Simulate a prediction: in real usage call the model.
        # Here we estimate "correctness" from a content heuristic so the
        # accuracy window is non-trivial.
        predicted = self._mock_predict(input_data)
        is_correct = predicted == label
        with self._lock:
            self._examples_seen += 1
            if is_correct:
                self._correct += 1
            self._window.append((is_correct, weight))

    def update_model(self, batch: List[Dict]) -> LearningResult:
        """Process a batch of labelled examples and return a learning result.

        Parameters
        ----------
        batch:
            List of dicts, each with ``input_data``, ``label``, and
            optional ``weight`` keys.

        Returns
        -------
        LearningResult
            Summary of the update.
        """
        for item in batch:
            self.learn_from_example(
                input_data=item.get("input_data", {}),
                label=item.get("label", ""),
                weight=item.get("weight", 1.0),
            )

        drift = self.detect_concept_drift()
        if drift:
            logger.warning(
                "Concept drift detected — resetting accuracy window "
                "(threshold=%.2f)",
                self._drift_threshold,
            )
            self.reset_window()

        accuracy = self.get_accuracy_estimate()
        with self._lock:
            self._last_update = time.time()

        result = LearningResult(
            examples_seen=self._examples_seen,
            recent_accuracy=accuracy,
            drift_detected=drift,
            model_updated=len(batch) > 0,
            timestamp=time.time(),
        )
        logger.debug("update_model: %s", result)
        return result

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_concept_drift(self) -> bool:
        """Compare recent vs. older accuracy inside the window.

        Returns
        -------
        bool
            True if the absolute accuracy gap exceeds ``_drift_threshold``.
        """
        with self._lock:
            window = list(self._window)

        if len(window) < 20:
            return False

        mid = len(window) // 2
        old_half = window[:mid]
        new_half = window[mid:]

        def weighted_acc(items: List[Tuple[bool, float]]) -> float:
            total_w = sum(w for _, w in items)
            if total_w == 0:
                return 0.0
            correct_w = sum(w for correct, w in items if correct)
            return correct_w / total_w

        old_acc = weighted_acc(old_half)
        new_acc = weighted_acc(new_half)
        drift = abs(new_acc - old_acc) > self._drift_threshold
        logger.debug(
            "drift check: old=%.3f new=%.3f diff=%.3f drift=%s",
            old_acc, new_acc, abs(new_acc - old_acc), drift,
        )
        return drift

    def get_accuracy_estimate(self) -> float:
        """Weighted accuracy over the current window."""
        with self._lock:
            window = list(self._window)
        if not window:
            return 0.0
        total_w = sum(w for _, w in window)
        if total_w == 0:
            return 0.0
        correct_w = sum(w for correct, w in window if correct)
        return correct_w / total_w

    def reset_window(self) -> None:
        """Clear the sliding accuracy window (e.g. after drift detection)."""
        with self._lock:
            self._window.clear()
        logger.info("IncrementalLearner window reset")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_model_stats(self) -> Dict:
        """Return a summary dict suitable for logging / monitoring."""
        with self._lock:
            window_size = len(self._window)
        return {
            "examples_seen": self._examples_seen,
            "overall_correct": self._correct,
            "overall_accuracy": (
                self._correct / self._examples_seen
                if self._examples_seen else 0.0
            ),
            "window_accuracy": self.get_accuracy_estimate(),
            "window_size": window_size,
            "drift_threshold": self._drift_threshold,
            "last_update": self._last_update,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_predict(input_data: Dict) -> str:
        """Trivial heuristic predictor used in the absence of a real model.

        Returns the string value of the first key whose value is a string,
        or "unknown" otherwise.
        """
        for v in input_data.values():
            if isinstance(v, str):
                return v
        return "unknown"


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_learner_instance: Optional[IncrementalLearner] = None
_learner_lock: threading.Lock = threading.Lock()


def get_incremental_learner() -> IncrementalLearner:
    """Return the process-wide IncrementalLearner singleton."""
    global _learner_instance
    if _learner_instance is None:
        with _learner_lock:
            if _learner_instance is None:
                _learner_instance = IncrementalLearner()
    return _learner_instance
