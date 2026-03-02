"""Semantic Memory module — AI Holographic Wristwatch.

Stores subject–predicate–object triples (facts) with confidence weights.
Supports contradiction detection and a simple knowledge-graph export.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Fact:
    """A single semantic fact stored as an SPO triple."""

    id: str
    subject: str
    predicate: str
    object: str
    confidence: float  # 0.0 – 1.0
    source: str
    timestamp: float
    contradiction_ids: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Fact({self.subject!r} --{self.predicate!r}--> {self.object!r} "
            f"conf={self.confidence:.2f})"
        )


class SemanticMemory:
    """Associative fact store organised as a subject-indexed SPO graph.

    Two indexes are maintained for efficient lookup:
    * ``_facts``          – {fact_id: Fact}
    * ``_subject_index``  – {subject: [fact_id, …]}
    """

    def __init__(self) -> None:
        self._facts: Dict[str, Fact] = {}
        self._subject_index: Dict[str, List[str]] = {}
        self._lock: threading.Lock = threading.Lock()
        logger.debug("SemanticMemory initialised")

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def store_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 0.8,
        source: str = "inference",
    ) -> Fact:
        """Persist a new SPO triple.

        Automatically detects contradictions with existing facts that share
        the same subject+predicate but a different object.

        Returns
        -------
        Fact
            The newly created Fact, with ``contradiction_ids`` populated if
            contradictions were found.
        """
        confidence = max(0.0, min(1.0, confidence))
        fact = Fact(
            id=str(uuid.uuid4()),
            subject=subject.lower().strip(),
            predicate=predicate.lower().strip(),
            object=object.lower().strip(),
            confidence=confidence,
            source=source,
            timestamp=time.time(),
        )
        # Detect contradictions before inserting
        contradictions = self.find_contradictions(fact)
        fact.contradiction_ids = [c.id for c in contradictions]
        if contradictions:
            logger.warning(
                "Semantic fact %r contradicts %d existing facts",
                str(fact), len(contradictions),
            )

        with self._lock:
            self._facts[fact.id] = fact
            self._subject_index.setdefault(fact.subject, []).append(fact.id)

        logger.debug("Semantic stored: %s", fact)
        return fact

    def update_belief(self, fact_id: str, new_confidence: float) -> bool:
        """Update the confidence of an existing fact.

        Returns
        -------
        bool
            True if the fact was found and updated.
        """
        new_confidence = max(0.0, min(1.0, new_confidence))
        with self._lock:
            fact = self._facts.get(fact_id)
            if fact is None:
                return False
            fact.confidence = new_confidence
        logger.debug("Semantic belief updated %s → %.3f", fact_id[:8], new_confidence)
        return True

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query_facts(
        self, subject: str, predicate: Optional[str] = None
    ) -> List[Fact]:
        """Return facts about *subject*, optionally filtered by *predicate*.

        Results are sorted by confidence descending.
        """
        subject_key = subject.lower().strip()
        with self._lock:
            fact_ids = list(self._subject_index.get(subject_key, []))
            facts = [self._facts[fid] for fid in fact_ids if fid in self._facts]

        if predicate is not None:
            pred_key = predicate.lower().strip()
            facts = [f for f in facts if f.predicate == pred_key]

        facts.sort(key=lambda f: f.confidence, reverse=True)
        return facts

    def query_by_object(self, object_: str) -> List[Fact]:
        """Return all facts whose object matches *object_*.

        Performs a linear scan; for large stores consider an object index.
        """
        obj_key = object_.lower().strip()
        with self._lock:
            matching = [f for f in self._facts.values() if f.object == obj_key]
        matching.sort(key=lambda f: f.confidence, reverse=True)
        return matching

    def find_contradictions(self, fact: Fact) -> List[Fact]:
        """Return facts that share the same subject+predicate but differ in object.

        Parameters
        ----------
        fact:
            The candidate fact to check.  It does not need to be stored yet.

        Returns
        -------
        List[Fact]
            Conflicting facts sorted by confidence descending.
        """
        existing = self.query_facts(fact.subject, fact.predicate)
        contradictions = [
            f for f in existing
            if f.object != fact.object and f.id != fact.id
        ]
        return contradictions

    def get_knowledge_graph(self) -> Dict[str, List[Dict]]:
        """Export the fact store as a subject-keyed adjacency dict.

        Returns
        -------
        Dict[str, List[Dict]]
            ``{subject: [{predicate, object, confidence, fact_id}, …]}``
        """
        graph: Dict[str, List[Dict]] = {}
        with self._lock:
            facts = list(self._facts.values())
        for f in facts:
            graph.setdefault(f.subject, []).append(
                {
                    "predicate": f.predicate,
                    "object": f.object,
                    "confidence": f.confidence,
                    "fact_id": f.id,
                }
            )
        return graph

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return a summary dict suitable for logging / monitoring."""
        with self._lock:
            facts = list(self._facts.values())
            n_subjects = len(self._subject_index)
        confidences = [f.confidence for f in facts]
        n_contradictions = sum(1 for f in facts if f.contradiction_ids)
        sources: Dict[str, int] = {}
        for f in facts:
            sources[f.source] = sources.get(f.source, 0) + 1
        return {
            "total_facts": len(facts),
            "unique_subjects": n_subjects,
            "avg_confidence": (
                sum(confidences) / len(confidences)
            ) if confidences else 0.0,
            "facts_with_contradictions": n_contradictions,
            "sources": sources,
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_semantic_instance: Optional[SemanticMemory] = None
_semantic_lock: threading.Lock = threading.Lock()


def get_semantic_memory() -> SemanticMemory:
    """Return the process-wide SemanticMemory singleton."""
    global _semantic_instance
    if _semantic_instance is None:
        with _semantic_lock:
            if _semantic_instance is None:
                _semantic_instance = SemanticMemory()
    return _semantic_instance
