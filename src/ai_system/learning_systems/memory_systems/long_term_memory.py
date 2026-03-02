"""Long-Term Memory module — AI Holographic Wristwatch.

Implements a persistent associative store with Ebbinghaus forgetting-curve
decay, spaced-repetition strengthening, and keyword-based retrieval.
"""
from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Memory:
    """A single long-term memory record."""

    id: str
    content: str
    category: str
    strength: float  # 0.0 – 1.0
    created_at: float
    last_accessed: float
    access_count: int = 0
    source: str = "experience"
    tags: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Memory(id={self.id[:8]}, category={self.category!r}, "
            f"strength={self.strength:.3f}, content={self.content[:40]!r})"
        )


class LongTermMemory:
    """Persistent associative memory with Ebbinghaus forgetting dynamics.

    Memories are stored in a dict keyed by UUID.  Two index structures
    accelerate retrieval:

    * keyword → set[memory_id]   (built lazily on each store/retrieve)
    * Direct dict lookup by ID   (O(1))
    """

    CONSOLIDATION_THRESHOLD: float = 0.3  # min strength to retain
    FORGETTING_RATE: float = 0.01  # strength lost per day (linear approximation)

    def __init__(self) -> None:
        self._memories: Dict[str, Memory] = {}
        self._lock: threading.Lock = threading.Lock()
        self._store_count: int = 0
        logger.debug("LongTermMemory initialised")

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        category: str,
        initial_strength: float = 0.5,
        source: str = "experience",
        tags: Optional[List[str]] = None,
    ) -> Memory:
        """Persist a new memory.

        Parameters
        ----------
        content:
            Human-readable description of what is remembered.
        category:
            Broad domain label (e.g. "health", "preference", "fact").
        initial_strength:
            Starting retention strength in [0, 1].
        source:
            How the memory was formed (e.g. "experience", "inference").
        tags:
            Optional keyword labels for retrieval.

        Returns
        -------
        Memory
            The newly created Memory object.
        """
        initial_strength = max(0.0, min(1.0, initial_strength))
        now = time.time()
        mem = Memory(
            id=str(uuid.uuid4()),
            content=content,
            category=category,
            strength=initial_strength,
            created_at=now,
            last_accessed=now,
            access_count=0,
            source=source,
            tags=tags or [],
        )
        with self._lock:
            self._memories[mem.id] = mem
            self._store_count += 1
        logger.debug(
            "LTM stored memory #%d id=%s cat=%s strength=%.2f",
            self._store_count, mem.id[:8], category, initial_strength,
        )
        return mem

    def strengthen(self, memory_id: str, boost: float = 0.1) -> bool:
        """Spaced-repetition boost — increase strength by *boost*, capped at 1.

        Also increments the access counter and updates last_accessed.

        Returns
        -------
        bool
            True if the memory was found and updated.
        """
        with self._lock:
            mem = self._memories.get(memory_id)
            if mem is None:
                return False
            mem.strength = min(1.0, mem.strength + boost)
            mem.access_count += 1
            mem.last_accessed = time.time()
        logger.debug("LTM strengthened %s → %.3f", memory_id[:8], mem.strength)
        return True

    def weaken(self, memory_id: str, decay: float = 0.05) -> bool:
        """Reduce a memory's strength by *decay*.

        If strength falls below CONSOLIDATION_THRESHOLD the memory is kept
        but flagged for eventual pruning.

        Returns
        -------
        bool
            True if the memory was found.
        """
        with self._lock:
            mem = self._memories.get(memory_id)
            if mem is None:
                return False
            mem.strength = max(0.0, mem.strength - decay)
        logger.debug("LTM weakened %s → %.3f", memory_id[:8], mem.strength)
        return True

    def forget(self, memory_id: str) -> bool:
        """Permanently remove a memory.

        Returns
        -------
        bool
            True if the memory existed and was deleted.
        """
        with self._lock:
            existed = memory_id in self._memories
            if existed:
                del self._memories[memory_id]
        if existed:
            logger.debug("LTM forgot memory %s", memory_id[:8])
        return existed

    def apply_forgetting_curve(self, days_elapsed: float = 1.0) -> int:
        """Apply Ebbinghaus-inspired linear decay to all memories.

        Memories below CONSOLIDATION_THRESHOLD are removed.

        Parameters
        ----------
        days_elapsed:
            How many days have passed since the last forgetting-curve pass.

        Returns
        -------
        int
            Number of memories whose strength was weakened (not removed).
        """
        decay = self.FORGETTING_RATE * days_elapsed
        weakened = 0
        to_remove: List[str] = []

        with self._lock:
            for mem_id, mem in self._memories.items():
                # Boost based on access frequency (spaced-repetition effect)
                frequency_bonus = min(0.5, mem.access_count * 0.02)
                effective_decay = max(0.0, decay - frequency_bonus)
                mem.strength = max(0.0, mem.strength - effective_decay)
                weakened += 1
                if mem.strength < self.CONSOLIDATION_THRESHOLD:
                    to_remove.append(mem_id)
            for mem_id in to_remove:
                del self._memories[mem_id]

        logger.info(
            "LTM forgetting curve: %d weakened, %d pruned (days=%.2f)",
            weakened, len(to_remove), days_elapsed,
        )
        return weakened

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def retrieve(self, query: str, n_results: int = 5) -> List[Memory]:
        """Keyword + strength retrieval.

        Scores each memory as:
            score = keyword_overlap_ratio * 0.6 + strength * 0.4

        Parameters
        ----------
        query:
            Space-separated keywords.
        n_results:
            Maximum number of results to return.

        Returns
        -------
        List[Memory]
            Top matches sorted by score descending.
        """
        keywords = set(query.lower().split())
        results: List[tuple[float, Memory]] = []

        with self._lock:
            mems = list(self._memories.values())

        for mem in mems:
            haystack_words = set(
                (mem.content + " " + " ".join(mem.tags) + " " + mem.category)
                .lower()
                .split()
            )
            if not keywords:
                overlap = 0.0
            else:
                overlap = len(keywords & haystack_words) / len(keywords)
            score = overlap * 0.6 + mem.strength * 0.4
            if score > 0:
                results.append((score, mem))

        results.sort(key=lambda x: x[0], reverse=True)
        top = [mem for _, mem in results[:n_results]]

        # Update access metadata for retrieved memories
        now = time.time()
        with self._lock:
            for mem in top:
                if mem.id in self._memories:
                    self._memories[mem.id].access_count += 1
                    self._memories[mem.id].last_accessed = now

        return top

    def get_by_category(self, category: str) -> List[Memory]:
        """Return all memories in *category*, sorted by strength descending."""
        with self._lock:
            mems = [m for m in self._memories.values() if m.category == category]
        mems.sort(key=lambda m: m.strength, reverse=True)
        return mems

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return a summary dict suitable for logging / monitoring."""
        with self._lock:
            mems = list(self._memories.values())
        strengths = [m.strength for m in mems]
        categories: Dict[str, int] = {}
        for m in mems:
            categories[m.category] = categories.get(m.category, 0) + 1
        return {
            "total_memories": len(mems),
            "total_stored": self._store_count,
            "avg_strength": (sum(strengths) / len(strengths)) if strengths else 0.0,
            "max_strength": max(strengths, default=0.0),
            "min_strength": min(strengths, default=0.0),
            "categories": categories,
            "consolidation_threshold": self.CONSOLIDATION_THRESHOLD,
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_ltm_instance: Optional[LongTermMemory] = None
_ltm_lock: threading.Lock = threading.Lock()


def get_long_term_memory() -> LongTermMemory:
    """Return the process-wide LongTermMemory singleton."""
    global _ltm_instance
    if _ltm_instance is None:
        with _ltm_lock:
            if _ltm_instance is None:
                _ltm_instance = LongTermMemory()
    return _ltm_instance
