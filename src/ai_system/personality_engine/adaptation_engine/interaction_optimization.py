"""Interaction Optimization module — AI Holographic Wristwatch.

Analyses historical interactions, identifies recurring patterns and applies
data-driven optimisations to improve future response quality and user
satisfaction.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InteractionPattern:
    pattern_type: str
    frequency: int
    avg_satisfaction: float
    context: Dict
    last_seen: float


@dataclass
class OptimizationResult:
    original_response: str
reoptimized_response: str
    changes_made: List[str]
    expected_improvement: float


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class InteractionOptimizer:
    """Learn from past interactions and optimise response generation."""

    # Minimum satisfaction score (0-1) to consider a pattern positive
    _SATISFACTION_THRESHOLD = 0.65

    # Heuristic max response length in words for conciseness optimisation
    _VERBOSE_WORD_THRESHOLD = 120

    def __init__(self) -> None:
        self._patterns: Dict[str, InteractionPattern] = {}
        self._feedback_history: Deque[Tuple[str, float, Dict]] = deque(maxlen=200)
        self._lock = threading.Lock()
        self._optimize_count: int = 0
        logger.debug("InteractionOptimizer initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_interaction(
        self,
        response: str,
        user_feedback: float,
        context: Dict,
    ) -> None:
        """Record a completed interaction and its feedback score.

        Args:
            response: The response that was delivered to the user.
            user_feedback: Satisfaction score in [0.0, 1.0].
            context: Metadata dict (e.g. intent, topic, time_of_day).
        """
        user_feedback = max(0.0, min(1.0, user_feedback))
        with self._lock:
            self._feedback_history.append((response, user_feedback, context))
            pattern_type = context.get("intent", "general")
            self._update_pattern(pattern_type, user_feedback, context)
        logger.debug(
            "Recorded interaction: intent=%s feedback=%.2f",
            context.get("intent", "?"), user_feedback,
        )

    def optimize_response(self, response: str, context: Dict) -> OptimizationResult:
        """Apply learned optimisations to a candidate response.

        Optimisations performed:
        - Conciseness: trim overly verbose responses.
        - Clarity: remove filler phrases.
        - Positive framing: replace weak negatives with constructive alternatives.
        - Context-specific improvements based on pattern history.

        Args:
            response: Draft response text.
            context: Metadata dict describing the interaction context.

        Returns:
            OptimizationResult with the improved response and change log.
        """
        with self._lock:
            self._optimize_count += 1
            pattern_type = context.get("intent", "general")
            pattern = self._patterns.get(pattern_type)

        optimized = response
        changes: List[str] = []

        # --- Conciseness ---
        words = optimized.split()
        if len(words) > self._VERBOSE_WORD_THRESHOLD:
            # Trim redundant filler sentences
            filler_sentences = [
                r"\bAs an AI(?:[^.]*)\.\s*",
                r"\bIt is worth noting that\b",
                r"\bI would like to point out that\b",
                r"\bPlease note that\b",
                r"\bOf course,?\s*",
                r"\bCertainly,?\s*",
                r"\bAbsolutely,?\s*",
            ]
            for pattern_re in filler_sentences:
                before = optimized
                optimized = re.sub(pattern_re, "", optimized, flags=re.IGNORECASE)
                if optimized != before:
                    changes.append(f"Removed filler: '{pattern_re[:40]}'")

        # --- Clarity improvements ---
        clarity_replacements = [
            (r"\bin order to\b", "to"),
            (r"\bdue to the fact that\b", "because"),
            (r"\bat this point in time\b", "now"),
            (r"\bfor the purpose of\b", "to"),
            (r"\bin the event that\b", "if"),
            (r"\bwith regard to\b", "regarding"),
        ]
        for bad, good in clarity_replacements:
            before = optimized
            optimized = re.sub(bad, good, optimized, flags=re.IGNORECASE)
            if optimized != before:
                changes.append(f"Clarity: '{bad}' → '{good}'")

        # --- Positive framing ---
        negative_replacements = [
            ("I can't help with that.", "Let me suggest an alternative approach."),
            ("I don't know.", "I don't have that information right now, but I can find out."),
            ("That's wrong.", "There may be a more accurate way to look at that."),
            ("You failed to", "You might want to"),
        ]
        for neg, pos in negative_replacements:
            if neg.lower() in optimized.lower():
                optimized = re.sub(re.escape(neg), pos, optimized, flags=re.IGNORECASE)
                changes.append(f"Positive framing: '{neg[:30]}' → '{pos[:30]}'")

        # --- Pattern-driven adjustment ---
        if pattern and pattern.avg_satisfaction < self._SATISFACTION_THRESHOLD:
            # Low satisfaction for this pattern type: add clarifying offer
            if not optimized.rstrip().endswith("?"):
                optimized = optimized.rstrip() + " Would you like me to elaborate further?"
                changes.append("Added clarification offer (low historical satisfaction)")

        expected_improvement = min(0.30, len(changes) * 0.06)
        logger.debug(
            "Response optimised: %d change(s), expected improvement=%.2f",
            len(changes), expected_improvement,
        )
        return OptimizationResult(
            original_response=response,
            optimized_response=optimized,
            changes_made=changes,
            expected_improvement=expected_improvement,
        )

    def identify_interaction_patterns(self) -> List[InteractionPattern]:
        """Return all known interaction patterns sorted by frequency (descending).

        Returns:
            List of InteractionPattern objects.
        """
        with self._lock:
            patterns = sorted(
                self._patterns.values(),
                key=lambda p: p.frequency,
                reverse=True,
            )
        return list(patterns)

    def apply_optimization(self, interaction_type: str, context: Dict) -> Dict:
        """Return parameter adjustment hints for a specific interaction type.

        Used by upstream generators to tune response parameters before
        generation (e.g. temperature, length, formality).

        Args:
            interaction_type: Category of interaction (e.g. "question", "command").
            context: Additional context for fine-tuning.

        Returns:
            Dict of parameter adjustments.
        """
        with self._lock:
            pattern = self._patterns.get(interaction_type)

        adjustments: Dict = {
            "length_factor": 1.0,
            "formality_delta": 0.0,
            "add_examples": False,
            "add_clarification_offer": False,
        }

        if pattern is None:
            return adjustments

        # If pattern satisfaction is high: replicate the style
        if pattern.avg_satisfaction >= 0.80:
            # Mirror context parameters that led to success
            saved_ctx = pattern.context
            if saved_ctx.get("was_concise"):
                adjustments["length_factor"] = 0.8
            if saved_ctx.get("used_examples"):
                adjustments["add_examples"] = True

        # Low satisfaction: prompt more explanation
        elif pattern.avg_satisfaction < self._SATISFACTION_THRESHOLD:
            adjustments["add_examples"] = True
            adjustments["add_clarification_offer"] = True
            adjustments["length_factor"] = 1.15

        # Formality hint from context
        if context.get("formal_setting"):
            adjustments["formality_delta"] = +0.2
        elif context.get("casual_setting"):
            adjustments["formality_delta"] = -0.2

        return adjustments

    def get_optimization_summary(self) -> Dict:
        """Return a high-level summary of optimisation effectiveness.

        Returns:
            Dict with aggregate stats.
        """
        with self._lock:
            if not self._feedback_history:
                return {"message": "No interactions recorded yet."}

            scores = [fb for (_, fb, _) in self._feedback_history]
            avg_satisfaction = sum(scores) / len(scores)
            above_threshold = sum(1 for s in scores if s >= self._SATISFACTION_THRESHOLD)
            positive_patterns = [
                p for p in self._patterns.values()
                if p.avg_satisfaction >= self._SATISFACTION_THRESHOLD
            ]
            negative_patterns = [
                p for p in self._patterns.values()
                if p.avg_satisfaction < self._SATISFACTION_THRESHOLD
            ]
            return {
                "total_interactions": len(self._feedback_history),
                "avg_satisfaction": round(avg_satisfaction, 3),
                "above_threshold_pct": round(100 * above_threshold / len(scores), 1),
                "positive_patterns": len(positive_patterns),
                "negative_patterns": len(negative_patterns),
                "optimize_calls": self._optimize_count,
            }

    def get_stats(self) -> Dict:
        """Return runtime statistics.

        Returns:
            Dict with stats fields.
        """
        summary = self.get_optimization_summary()
        summary["known_patterns"] = len(self._patterns)
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_pattern(
        self,
        pattern_type: str,
        satisfaction: float,
        context: Dict,
    ) -> None:
        """Update or create an InteractionPattern entry (must hold _lock)."""
        existing = self._patterns.get(pattern_type)
        if existing is None:
            self._patterns[pattern_type] = InteractionPattern(
                pattern_type=pattern_type,
                frequency=1,
                avg_satisfaction=satisfaction,
                context=dict(context),
                last_seen=time.time(),
            )
        else:
            # Exponential moving average
            alpha = 0.25
            new_avg = (1 - alpha) * existing.avg_satisfaction + alpha * satisfaction
            existing.avg_satisfaction = new_avg
            existing.frequency += 1
            existing.last_seen = time.time()
            # Merge context keys (keep most recent values)
            existing.context.update(context)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_interaction_optimizer_instance: Optional[InteractionOptimizer] = None
_interaction_optimizer_lock = threading.Lock()


def get_interaction_optimizer() -> InteractionOptimizer:
    """Return the process-wide singleton InteractionOptimizer.

    Returns:
        Singleton InteractionOptimizer instance.
    """
    global _interaction_optimizer_instance
    if _interaction_optimizer_instance is None:
        with _interaction_optimizer_lock:
            if _interaction_optimizer_instance is None:
                _interaction_optimizer_instance = InteractionOptimizer()
    return _interaction_optimizer_instance
