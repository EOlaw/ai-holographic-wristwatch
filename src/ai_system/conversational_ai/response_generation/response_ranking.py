"""Response ranking module for the AI Holographic Wristwatch.

Given multiple candidate response strings, ranks them by relevance,
naturalness, and safety, and returns an ordered list of :class:`RankedResponse`
objects. Provides a thread-safe singleton via :func:`get_response_ranker`.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RankedResponse:
    """A single candidate response together with its scoring breakdown."""
    text: str
    score: float
    relevance: float
    naturalness: float
    safety: float
    original_intent: str
    rank: int = 0

    def __repr__(self) -> str:
        return (
            f"RankedResponse(rank={self.rank}, score={self.score:.3f}, "
            f"text={self.text[:55]!r})"
        )


# ---------------------------------------------------------------------------
# Scoring constants & patterns
# ---------------------------------------------------------------------------

# Keywords that improve relevance if they appear in the response for a given
# intent group.
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "health": ["heart", "bpm", "steps", "sleep", "calories", "oxygen", "stress",
               "rate", "blood", "fitness", "health", "activity"],
    "time": ["time", "hour", "minute", "second", "clock", "alarm", "timer",
             "schedule", "today", "date"],
    "weather": ["weather", "rain", "sunny", "cloudy", "temperature", "forecast",
                "wind", "humidity", "storm", "celsius", "fahrenheit"],
    "navigation": ["navigate", "route", "direction", "turn", "arrive", "destination",
                   "eta", "miles", "kilometres", "map"],
    "messaging": ["message", "send", "contact", "reply", "call", "notify",
                  "notification", "text", "email"],
    "general": ["help", "sure", "ok", "yes", "no", "sorry", "please", "thank",
                "great", "good", "interesting"],
}

# Regex patterns that, if found, lower the safety score.
_UNSAFE_PATTERNS: List[Tuple[str, float]] = [
    (r"\b(kill|murder|harm|hurt|attack|threaten)\b", 0.0),
    (r"\b(bomb|weapon|explosive|gun|knife)\b", 0.05),
    (r"\b(hate|racist|sexist|discriminat)\b", 0.1),
    (r"\b(die|dead|suicide|overdose)\b", 0.3),
    (r"\b(drug|narcotic|illegal)\b", 0.4),
    (r"(https?://|www\.)\S+", 0.85),    # URLs — mild concern on wristwatch
]

# Naturalness heuristics.
_CONTRACTION_PATTERN = re.compile(
    r"\b(don't|can't|won't|isn't|aren't|it's|I'm|I've|you're|we're|they're|"
    r"that's|there's|here's|what's|how's)\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?]$")

# Ideal response length range in characters (for wristwatch short replies).
_IDEAL_MIN_CHARS = 20
_IDEAL_MAX_CHARS = 180


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ResponseRanker:
    """Ranks a list of candidate response strings for a given intent/context.

    Thread-safe; a shared singleton is available via :func:`get_response_ranker`.
    """

    # Default composite weights: relevance, naturalness, safety.
    DEFAULT_WEIGHTS: Tuple[float, float, float] = (0.40, 0.30, 0.30)

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._rank_count: int = 0
        self._start_time: float = time.monotonic()
        logger.debug("ResponseRanker initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank_candidates(
        self,
        candidates: List[str],
        context: Dict,
        intent: str,
        weights: Optional[Tuple[float, float, float]] = None,
    ) -> List[RankedResponse]:
        """Score and rank all *candidates* for the given *intent* and *context*.

        Args:
            candidates: Raw response strings to evaluate.
            context: Current dialogue/session context dictionary.
            intent: The detected intent name (used for relevance scoring).
            weights: Optional (relevance_w, naturalness_w, safety_w) tuple.
                Must sum to 1.0. Defaults to :attr:`DEFAULT_WEIGHTS`.

        Returns:
            A list of :class:`RankedResponse` objects sorted best-first.
        """
        if not candidates:
            logger.warning("rank_candidates called with empty candidate list.")
            return []

        if weights is None:
            weights = self.DEFAULT_WEIGHTS

        with self._lock:
            self._rank_count += 1

        ranked: List[RankedResponse] = []
        for text in candidates:
            rel = self.score_relevance(text, context, intent)
            nat = self.score_naturalness(text)
            saf = self.score_safety(text)
            composite = self._compute_composite(rel, nat, saf, weights)
            ranked.append(
                RankedResponse(
                    text=text,
                    score=composite,
                    relevance=rel,
                    naturalness=nat,
                    safety=saf,
                    original_intent=intent,
                )
            )

        ranked.sort(key=lambda r: r.score, reverse=True)
        for idx, resp in enumerate(ranked, start=1):
            resp.rank = idx

        logger.debug(
            "Ranked %d candidates for intent=%r; top score=%.3f",
            len(ranked),
            intent,
            ranked[0].score if ranked else 0.0,
        )
        return ranked

    def score_relevance(self, response: str, context: Dict, intent: str) -> float:
        """Score how well *response* addresses the given *intent* and *context*.

        Scoring factors:
        - Keyword overlap between response tokens and intent-specific keywords.
        - Length appropriateness (too short or too long reduces score).
        - Presence of context-specific entities (e.g. contact names, locations).

        Returns a float in [0.0, 1.0].
        """
        score = 0.5  # neutral baseline

        lower = response.lower()
        tokens = set(re.findall(r"\b\w+\b", lower))

        # --- Keyword overlap ---
        # Determine which keyword group(s) apply to this intent.
        intent_lower = intent.lower()
        keyword_score = 0.0
        matched_groups = 0
        for group, keywords in _INTENT_KEYWORDS.items():
            if group in intent_lower or any(kw in intent_lower for kw in keywords[:3]):
                overlap = tokens.intersection(set(keywords))
                if keywords:
                    keyword_score += len(overlap) / len(keywords)
                matched_groups += 1
        if matched_groups:
            keyword_score /= matched_groups
            score = 0.3 + keyword_score * 0.5   # map [0,1] → [0.3, 0.8]

        # --- Length appropriateness ---
        length = len(response)
        if _IDEAL_MIN_CHARS <= length <= _IDEAL_MAX_CHARS:
            length_bonus = 0.15
        elif length < _IDEAL_MIN_CHARS:
            length_bonus = (length / _IDEAL_MIN_CHARS) * 0.05
        else:  # too long
            overshoot = (length - _IDEAL_MAX_CHARS) / _IDEAL_MAX_CHARS
            length_bonus = max(-0.15, -overshoot * 0.15)
        score += length_bonus

        # --- Context entity matching ---
        context_values: List[str] = []
        for v in context.values():
            if isinstance(v, str):
                context_values.append(v.lower())
            elif isinstance(v, (int, float)):
                context_values.append(str(v))

        entity_hits = sum(1 for cv in context_values if cv in lower)
        if context_values:
            score += min(0.1, entity_hits / len(context_values) * 0.1)

        return max(0.0, min(1.0, score))

    def score_naturalness(self, response: str) -> float:
        """Evaluate how natural / conversational the *response* reads.

        Heuristics:
        - Proper sentence casing and end punctuation.
        - Presence of contractions (signals natural speech).
        - Reasonable sentence count and average sentence length.
        - Absence of repeated words (suggests template fill errors).

        Returns a float in [0.0, 1.0].
        """
        score = 0.5

        if not response.strip():
            return 0.0

        # Capitalised start.
        if response[0].isupper():
            score += 0.05

        # Ends with punctuation.
        if _SENTENCE_END.search(response.rstrip()):
            score += 0.05

        # Contractions present (natural conversational register).
        contractions_found = len(_CONTRACTION_PATTERN.findall(response))
        score += min(0.1, contractions_found * 0.05)

        # Reasonable length.
        char_count = len(response)
        if _IDEAL_MIN_CHARS <= char_count <= _IDEAL_MAX_CHARS:
            score += 0.15
        elif char_count < 10:
            score -= 0.15
        elif char_count > 300:
            score -= 0.10

        # Sentence count.
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]
        if 1 <= len(sentences) <= 3:
            score += 0.10

        # Average words per sentence.
        words = response.split()
        if sentences:
            avg_wps = len(words) / len(sentences)
            if 5 <= avg_wps <= 20:
                score += 0.05

        # No repeated consecutive words (template artefact).
        word_list = [w.lower() for w in words]
        repeats = sum(
            1 for i in range(1, len(word_list)) if word_list[i] == word_list[i - 1]
        )
        if repeats:
            score -= repeats * 0.05

        # Unfilled placeholders (e.g. "{duration}") are very bad.
        unfilled = len(re.findall(r"\{[^}]+\}", response))
        score -= unfilled * 0.20

        return max(0.0, min(1.0, score))

    def score_safety(self, response: str) -> float:
        """Check for potentially harmful or inappropriate content.

        Returns a float in [0.0, 1.0] where 1.0 is fully safe.
        """
        score = 1.0
        lower = response.lower()
        for pattern, penalty_level in _UNSAFE_PATTERNS:
            if re.search(pattern, lower, re.IGNORECASE):
                # The penalty_level is the minimum score allowed after this hit.
                score = min(score, penalty_level)
        return max(0.0, min(1.0, score))

    def _compute_composite(
        self,
        relevance: float,
        naturalness: float,
        safety: float,
        weights: Tuple[float, float, float],
    ) -> float:
        """Compute the weighted composite score.

        Safety acts as a hard cap: if safety < 0.5 the composite is clamped
        to at most 0.2 regardless of other scores.
        """
        w_rel, w_nat, w_saf = weights
        composite = w_rel * relevance + w_nat * naturalness + w_saf * safety

        if safety < 0.5:
            composite = min(composite, 0.2)

        return round(max(0.0, min(1.0, composite)), 4)

    def get_top_response(
        self,
        candidates: List[str],
        context: Dict,
        intent: str,
        weights: Optional[Tuple[float, float, float]] = None,
    ) -> Optional[RankedResponse]:
        """Return the single best-scoring candidate, or ``None`` if the list is empty."""
        ranked = self.rank_candidates(candidates, context, intent, weights)
        return ranked[0] if ranked else None

    def get_stats(self) -> Dict:
        """Return runtime statistics."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            return {
                "rank_count": self._rank_count,
                "uptime_seconds": round(elapsed, 2),
                "default_weights": self.DEFAULT_WEIGHTS,
                "unsafe_pattern_count": len(_UNSAFE_PATTERNS),
            }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_ranker_instance: Optional[ResponseRanker] = None
_ranker_lock = threading.Lock()


def get_response_ranker() -> ResponseRanker:
    """Return the module-level :class:`ResponseRanker` singleton.

    Thread-safe; the instance is created lazily on first call.
    """
    global _ranker_instance
    if _ranker_instance is None:
        with _ranker_lock:
            if _ranker_instance is None:
                _ranker_instance = ResponseRanker()
    return _ranker_instance


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ranker = get_response_ranker()

    candidates = [
        "Your heart rate is 78 bpm.",
        "78.",
        "I don't know.",
        "Heart rate: 78 beats per minute — within the healthy range for your activity level.",
        "kill me now",  # unsafe
    ]
    context = {"heart_rate": "78"}
    intent = "query_heart_rate"

    results = ranker.rank_candidates(candidates, context, intent)
    print("=== ResponseRanker Demo ===\n")
    for r in results:
        print(
            f"Rank {r.rank}: score={r.score:.3f}  rel={r.relevance:.2f}  "
            f"nat={r.naturalness:.2f}  saf={r.safety:.2f}\n"
            f"        {r.text!r}\n"
        )

    print("Stats:", ranker.get_stats())
