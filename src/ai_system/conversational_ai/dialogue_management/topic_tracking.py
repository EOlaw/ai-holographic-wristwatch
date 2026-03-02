"""Topic Tracking — AI Holographic Wristwatch

Tracks the conversational topic across dialogue turns, detects topic shifts,
maintains a topic history, and extracts keywords from utterances to build
lightweight topic profiles.
"""
from __future__ import annotations

import re
import threading
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque, Set

try:
    from src.core.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common stop words (minimal English set — no NLTK dependency)
# ---------------------------------------------------------------------------

_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "it", "they", "them", "their", "this",
    "that", "these", "those", "what", "which", "who", "how", "when", "where",
    "why", "please", "ok", "okay", "yeah", "yes", "no", "not", "just",
    "also", "so", "then", "now", "up", "out", "about", "from",
}

# Domain keyword hints used for coarse domain assignment
_DOMAIN_HINTS: Dict[str, List[str]] = {
    "health": [
        "heart", "rate", "pulse", "blood", "pressure", "steps", "calories",
        "sleep", "oxygen", "bpm", "workout", "exercise", "health", "fitness",
        "stress", "breath", "temperature", "medical", "doctor", "symptom",
    ],
    "navigation": [
        "directions", "navigate", "map", "location", "route", "turn", "left",
        "right", "north", "south", "east", "west", "address", "destination",
        "distance", "miles", "kilometers", "gps", "traffic",
    ],
    "communication": [
        "call", "message", "text", "email", "contact", "phone", "send",
        "reply", "forward", "inbox", "notification", "alert", "reminder",
        "voicemail", "chat",
    ],
    "media": [
        "play", "pause", "stop", "music", "song", "album", "artist",
        "playlist", "podcast", "video", "volume", "skip", "next", "previous",
        "track", "shuffle", "repeat",
    ],
    "timer": [
        "timer", "alarm", "stopwatch", "countdown", "minutes", "seconds",
        "hours", "schedule", "remind", "set", "cancel", "snooze",
    ],
    "hologram": [
        "hologram", "display", "projection", "show", "hide", "brightness",
        "colour", "color", "image", "3d", "render", "visualize", "visualise",
    ],
    "general": [],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Topic:
    """Represents a conversational topic identified during a session."""
    name: str
    domain: str
    start_time: float = field(default_factory=time.time)
    turn_count: int = 0
    keywords: List[str] = field(default_factory=list)

    def age_seconds(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "domain": self.domain,
            "start_time": self.start_time,
            "age_seconds": round(self.age_seconds(), 2),
            "turn_count": self.turn_count,
            "keywords": list(self.keywords),
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class TopicTracker:
    """Tracks and transitions between conversational topics.

    Uses keyword overlap to measure topic similarity and detect shifts.
    """

    _SHIFT_THRESHOLD: float = 0.20   # similarity below this -> topic shift
    _MAX_KEYWORDS: int = 12           # keywords retained per topic

    def __init__(self) -> None:
        self._current_topic: Optional[Topic] = None
        self._history: Deque[Topic] = deque(maxlen=20)
        self._lock: threading.RLock = threading.RLock()
        self._topic_shift_count: int = 0
        logger.debug("TopicTracker initialised.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_topic(
        self,
        utterance: str,
        intent_name: str = "",
        intent_domain: str = "",
    ) -> Topic:
        """Update the current topic based on a new utterance.

        If a significant topic shift is detected the old topic is archived
        and a new one is created.

        Args:
            utterance:     Raw user utterance text.
            intent_name:   Name of the recognised intent (optional).
            intent_domain: Domain of the recognised intent (optional).

        Returns:
            The (potentially new) current ``Topic``.
        """
        with self._lock:
            keywords = self._extract_keywords(utterance)
            domain = intent_domain or self._infer_domain(keywords)
            topic_name = intent_name or self._infer_topic_name(keywords, domain)

            if self._current_topic is None:
                # First utterance — create initial topic
                self._current_topic = Topic(
                    name=topic_name,
                    domain=domain,
                    keywords=keywords[: self._MAX_KEYWORDS],
                )
                logger.info(
                    "Initial topic set: '%s' (domain=%s)", topic_name, domain
                )
            else:
                similarity = self._calculate_topic_similarity(
                    self._current_topic, keywords
                )
                if similarity < self._SHIFT_THRESHOLD:
                    # Topic shift
                    logger.info(
                        "Topic shift detected (similarity=%.2f): '%s' -> '%s'",
                        similarity, self._current_topic.name, topic_name,
                    )
                    self._history.append(self._current_topic)
                    self._topic_shift_count += 1
                    self._current_topic = Topic(
                        name=topic_name,
                        domain=domain,
                        keywords=keywords[: self._MAX_KEYWORDS],
                    )
                else:
                    # Same topic — update keywords (union, capped)
                    existing = set(self._current_topic.keywords)
                    for kw in keywords:
                        if kw not in existing and len(self._current_topic.keywords) < self._MAX_KEYWORDS:
                            self._current_topic.keywords.append(kw)
                            existing.add(kw)
                    # Update domain if the intent gave us a better one
                    if intent_domain and intent_domain != "general":
                        self._current_topic.domain = intent_domain
                    logger.debug(
                        "Topic continued: '%s' (similarity=%.2f)",
                        self._current_topic.name, similarity,
                    )

            self._current_topic.turn_count += 1
            return self._current_topic

    def get_current_topic(self) -> Optional[Topic]:
        """Return the current topic, or None if no utterances have been seen."""
        with self._lock:
            return self._current_topic

    def detect_topic_shift(self, utterance: str) -> float:
        """Compute the probability (0–1) that *utterance* represents a topic shift.

        Returns:
            0.0 if no current topic exists (no shift possible).
            A value close to 1.0 for very low similarity (likely shift).
            A value close to 0.0 for high similarity (probably same topic).
        """
        with self._lock:
            if self._current_topic is None:
                return 0.0
            keywords = self._extract_keywords(utterance)
            if not keywords:
                return 0.0
            similarity = self._calculate_topic_similarity(self._current_topic, keywords)
            # Invert similarity to get shift probability
            shift_prob = max(0.0, min(1.0, 1.0 - similarity))
            logger.debug(
                "Topic shift probability for utterance: %.3f (similarity=%.3f)",
                shift_prob, similarity,
            )
            return shift_prob

    def get_topic_history(self) -> List[Topic]:
        """Return a list of archived (past) topics, oldest first."""
        with self._lock:
            return list(self._history)

    def get_stats(self) -> Dict:
        """Return summary statistics."""
        with self._lock:
            return {
                "current_topic": self._current_topic.to_dict() if self._current_topic else None,
                "topic_shift_count": self._topic_shift_count,
                "history_length": len(self._history),
                "history_topics": [t.name for t in self._history],
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from *text*, filtering stop words."""
        # Lowercase, strip punctuation, split
        clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        tokens = clean.split()
        keywords: List[str] = []
        seen: Set[str] = set()
        for tok in tokens:
            if len(tok) >= 3 and tok not in _STOP_WORDS and tok not in seen:
                keywords.append(tok)
                seen.add(tok)
        return keywords

    def _infer_domain(self, keywords: List[str]) -> str:
        """Map keywords to the best-matching domain."""
        scores: Dict[str, int] = {d: 0 for d in _DOMAIN_HINTS}
        kw_set = set(keywords)
        for domain, hints in _DOMAIN_HINTS.items():
            for hint in hints:
                if hint in kw_set:
                    scores[domain] += 1
        best_domain = max(scores, key=lambda d: scores[d])
        return best_domain if scores[best_domain] > 0 else "general"

    def _infer_topic_name(self, keywords: List[str], domain: str) -> str:
        """Build a short topic name from top keywords."""
        if keywords:
            return "_".join(keywords[:3])
        return domain if domain != "general" else "unknown"

    def _calculate_topic_similarity(
        self, topic: Topic, new_keywords: List[str]
    ) -> float:
        """Compute Jaccard similarity between a topic's keywords and new ones."""
        if not topic.keywords and not new_keywords:
            return 1.0
        if not topic.keywords or not new_keywords:
            return 0.0
        set_a = set(topic.keywords)
        set_b = set(new_keywords)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[TopicTracker] = None
_instance_lock: threading.Lock = threading.Lock()


def get_topic_tracker() -> TopicTracker:
    """Return the process-wide singleton ``TopicTracker``."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = TopicTracker()
    return _instance


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    tracker = get_topic_tracker()
    print("=== Topic Tracker Demo ===\n")

    utterances = [
        ("How many steps have I taken today?", "query_steps", "health"),
        ("What is my heart rate right now?", "query_heart_rate", "health"),
        ("Play some jazz music please", "play_music", "media"),
        ("Skip to the next track", "media_skip", "media"),
        ("Navigate me to the nearest coffee shop", "navigate_to", "navigation"),
    ]

    for utt, intent, domain in utterances:
        shift_prob = tracker.detect_topic_shift(utt)
        topic = tracker.update_topic(utt, intent, domain)
        print(f"Utterance : {utt!r}")
        print(f"  Shift prob: {shift_prob:.2f}  Topic: {topic.name}  Domain: {topic.domain}")
        print(f"  Keywords : {topic.keywords[:6]}\n")

    print(f"Topic history: {[t.name for t in tracker.get_topic_history()]}")
    print(f"Stats: {tracker.get_stats()}")
    print("\nDemo complete.")
