"""Knowledge Expansion module — AI Holographic Wristwatch.

Extracts and organises new knowledge items from live interactions, tracks
knowledge gaps against a reference topic set, and supports domain-scoped
queries.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeDomain(Enum):
    """High-level domain categories for knowledge items."""

    HEALTH = "health"
    TECHNOLOGY = "technology"
    PERSONAL = "personal"
    COMMUNICATION = "communication"
    NAVIGATION = "navigation"
    GENERAL = "general"


@dataclass
class KnowledgeItem:
    """A discrete piece of knowledge held by the system."""

    id: str
    domain: KnowledgeDomain
    topic: str
    content: str
    confidence: float  # 0.0 – 1.0
    learned_at: float
    usage_count: int = 0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"KnowledgeItem(domain={self.domain.value}, topic={self.topic!r}, "
            f"conf={self.confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# Domain keyword heuristics for automatic classification
# ---------------------------------------------------------------------------
_DOMAIN_KEYWORDS: Dict[KnowledgeDomain, List[str]] = {
    KnowledgeDomain.HEALTH: [
        "health", "medical", "symptom", "exercise", "diet", "heart", "sleep",
        "blood", "stress", "fitness", "calories", "medication", "doctor",
    ],
    KnowledgeDomain.TECHNOLOGY: [
        "device", "software", "hardware", "sensor", "holographic", "display",
        "battery", "wifi", "bluetooth", "app", "update", "firmware",
    ],
    KnowledgeDomain.PERSONAL: [
        "user", "preference", "name", "birthday", "family", "friend",
        "hobby", "goal", "reminder", "calendar", "schedule",
    ],
    KnowledgeDomain.COMMUNICATION: [
        "message", "call", "email", "chat", "notification", "language",
        "speech", "tone", "conversation",
    ],
    KnowledgeDomain.NAVIGATION: [
        "location", "map", "route", "direction", "gps", "address",
        "distance", "travel", "destination", "navigate",
    ],
}


def _infer_domain(text: str) -> KnowledgeDomain:
    """Return the best-matching KnowledgeDomain for *text*."""
    text_lower = text.lower()
    best_domain = KnowledgeDomain.GENERAL
    best_score = 0
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


class KnowledgeExpander:
    """Builds and maintains the AI's growing knowledge base.

    Knowledge items are stored in ``_items`` (keyed by UUID) and can be
    searched by keyword or filtered by domain.
    """

    def __init__(self) -> None:
        self._items: Dict[str, KnowledgeItem] = {}
        self._lock: threading.Lock = threading.Lock()
        self._expand_count: int = 0
        logger.debug("KnowledgeExpander initialised")

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_knowledge_item(
        self,
        domain: KnowledgeDomain,
        topic: str,
        content: str,
        confidence: float = 0.7,
    ) -> KnowledgeItem:
        """Manually add a knowledge item.

        Parameters
        ----------
        domain:
            Broad domain classification.
        topic:
            Short label for the subject (e.g. "step_counting").
        content:
            Detailed knowledge string.
        confidence:
            Certainty of this knowledge in [0, 1].

        Returns
        -------
        KnowledgeItem
            The newly created item.
        """
        confidence = max(0.0, min(1.0, confidence))
        item = KnowledgeItem(
            id=str(uuid.uuid4()),
            domain=domain,
            topic=topic,
            content=content,
            confidence=confidence,
            learned_at=time.time(),
        )
        with self._lock:
            self._items[item.id] = item
            self._expand_count += 1
        logger.debug("Knowledge added: %s", item)
        return item

    def expand_from_interaction(self, conversation_data: Dict) -> List[KnowledgeItem]:
        """Extract factual knowledge items from a conversation snapshot.

        Heuristics applied:
        * Sentences ending with a period that contain a verb are treated as
          candidate facts.
        * Domain is inferred from keyword density.
        * Confidence is set to 0.6 (hedged — from conversation, not verified).

        Parameters
        ----------
        conversation_data:
            Dict with at least ``text`` (str) and optional ``user_id`` (str).

        Returns
        -------
        List[KnowledgeItem]
            Newly created items extracted from the conversation.
        """
        text: str = conversation_data.get("text", "")
        if not text:
            return []

        # Split into sentence candidates
        sentences = re.split(r"[.!?]", text)
        new_items: List[KnowledgeItem] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            # Crude fact filter: must contain a verb-like word
            if not re.search(r"\b(is|are|was|were|has|have|can|will|does|do)\b", sentence, re.I):
                continue
            domain = _infer_domain(sentence)
            # Derive a short topic label from the first 3 non-stop words
            words = [w for w in sentence.split() if len(w) > 3]
            topic = "_".join(w.lower() for w in words[:3]) or "general_fact"

            item = self.add_knowledge_item(
                domain=domain,
                topic=topic,
                content=sentence,
                confidence=0.6,
            )
            new_items.append(item)

        logger.info(
            "expand_from_interaction: extracted %d items from %d chars",
            len(new_items), len(text),
        )
        return new_items

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_knowledge_gaps(self, reference_topics: List[str]) -> List[str]:
        """Return topics from *reference_topics* not covered in the knowledge base.

        A topic is considered covered if any stored item's topic or content
        contains the reference string (case-insensitive substring match).

        Parameters
        ----------
        reference_topics:
            List of topic strings expected to be present.

        Returns
        -------
        List[str]
            Topics with no matching knowledge item.
        """
        with self._lock:
            items = list(self._items.values())

        def is_covered(ref: str) -> bool:
            ref_lower = ref.lower()
            return any(
                ref_lower in item.topic.lower() or ref_lower in item.content.lower()
                for item in items
            )

        gaps = [ref for ref in reference_topics if not is_covered(ref)]
        logger.debug("Knowledge gaps: %d/%d topics missing", len(gaps), len(reference_topics))
        return gaps

    def search(self, query: str) -> List[KnowledgeItem]:
        """Keyword search over topic and content fields.

        Returns items sorted by relevance (keyword count) then confidence.
        """
        keywords = query.lower().split()
        with self._lock:
            items = list(self._items.values())

        scored: List[tuple[int, float, KnowledgeItem]] = []
        for item in items:
            haystack = (item.topic + " " + item.content).lower()
            hits = sum(1 for kw in keywords if kw in haystack)
            if hits > 0:
                item.usage_count += 1
                scored.append((hits, item.confidence, item))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item for _, _, item in scored]

    def get_by_domain(self, domain: KnowledgeDomain) -> List[KnowledgeItem]:
        """Return all items in *domain*, sorted by confidence descending."""
        with self._lock:
            items = [it for it in self._items.values() if it.domain == domain]
        items.sort(key=lambda it: it.confidence, reverse=True)
        return items

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return a summary dict suitable for logging / monitoring."""
        with self._lock:
            items = list(self._items.values())
        domain_counts: Dict[str, int] = {d.value: 0 for d in KnowledgeDomain}
        for item in items:
            domain_counts[item.domain.value] += 1
        confidences = [it.confidence for it in items]
        return {
            "total_items": len(items),
            "total_expanded": self._expand_count,
            "domain_distribution": domain_counts,
            "avg_confidence": (
                sum(confidences) / len(confidences)
            ) if confidences else 0.0,
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_expander_instance: Optional[KnowledgeExpander] = None
_expander_lock: threading.Lock = threading.Lock()


def get_knowledge_expander() -> KnowledgeExpander:
    """Return the process-wide KnowledgeExpander singleton."""
    global _expander_instance
    if _expander_instance is None:
        with _expander_lock:
            if _expander_instance is None:
                _expander_instance = KnowledgeExpander()
    return _expander_instance
