"""Short-Term Memory module — AI Holographic Wristwatch.

Implements a capacity-limited working memory using Miller's Law (7 ± 2 items)
with exponential decay of memory strength over time.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryItem:
    """A single item held in short-term memory."""

    content: Any
    importance: float  # 0.0 – 1.0
    timestamp: float = field(default_factory=time.time)
    decay_rate: float = 0.1  # strength half-life control
    tags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def age_seconds(self) -> float:
        """Seconds elapsed since the item was stored."""
        return time.time() - self.timestamp

    @property
    def current_strength(self) -> float:
        """Exponential decay: importance * exp(-decay_rate * age_hours).

        Returns a value in [0, 1].  Items lose roughly half their strength
        every (ln2 / decay_rate) hours.
        """
        age_hours = self.age_seconds / 3600.0
        return self.importance * math.exp(-self.decay_rate * age_hours)

    @property
    def is_alive(self) -> bool:
        """True while the item retains meaningful strength (> 0.05)."""
        return self.current_strength > 0.05

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MemoryItem(strength={self.current_strength:.3f}, "
            f"tags={self.tags}, content={str(self.content)[:40]!r})"
        )


class ShortTermMemory:
    """Capacity-limited working memory inspired by Miller's Law.

    Stores up to *CAPACITY* logically accessible items (7 by default).
    Internally a deque of size 50 is maintained so that recently displaced
    items can still be retrieved for a short window while their decay runs.
    """

    CAPACITY: int = 7  # Miller's Law

    def __init__(self) -> None:
        self._items: Deque[MemoryItem] = deque(maxlen=50)
        self._lock: threading.Lock = threading.Lock()
        self._add_count: int = 0
        logger.debug("ShortTermMemory initialised (capacity=%d)", self.CAPACITY)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add(
        self,
        content: Any,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> MemoryItem:
        """Add a new item to short-term memory.

        Parameters
        ----------
        content:
            Arbitrary payload (string, dict, …) to remember.
        importance:
            Initial salience in [0, 1].  Higher values decay more slowly.
        tags:
            Optional keyword labels for later search.

        Returns
        -------
        MemoryItem
            The newly created item.
        """
        importance = max(0.0, min(1.0, importance))
        item = MemoryItem(
            content=content,
            importance=importance,
            tags=tags or [],
        )
        with self._lock:
            self._items.append(item)
            self._add_count += 1
        logger.debug(
            "STM add #%d: importance=%.2f tags=%s",
            self._add_count, importance, tags,
        )
        return item

    def clear(self) -> int:
        """Remove all items.  Returns the count cleared."""
        with self._lock:
            n = len(self._items)
            self._items.clear()
        logger.info("STM cleared %d items", n)
        return n

    def prune_decayed(self) -> int:
        """Remove items whose strength has fallen below the alive threshold.

        Returns
        -------
        int
            Number of items removed.
        """
        with self._lock:
            before = len(self._items)
            alive = deque(
                (item for item in self._items if item.is_alive),
                maxlen=50,
            )
            removed = before - len(alive)
            self._items = alive
        if removed:
            logger.debug("STM pruned %d decayed items", removed)
        return removed

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_recent(self, n: int = 7) -> List[MemoryItem]:
        """Return the *n* strongest items, sorted by current strength descending."""
        with self._lock:
            items = list(self._items)
        items.sort(key=lambda x: x.current_strength, reverse=True)
        return items[:n]

    def get_context_window(self) -> List[MemoryItem]:
        """Return alive items sorted by most-recently-stored first."""
        with self._lock:
            items = list(self._items)
        alive = [item for item in items if item.is_alive]
        # Most recently stored items have the highest index in the deque.
        alive.sort(key=lambda x: x.timestamp, reverse=True)
        return alive[: self.CAPACITY]

    def search(self, query: str) -> List[MemoryItem]:
        """Keyword search over stringified content and tags.

        Parameters
        ----------
        query:
            Space-separated keywords.  All keywords must appear somewhere
            in the item's content string or tag list.

        Returns
        -------
        List[MemoryItem]
            Matching alive items sorted by strength descending.
        """
        keywords = query.lower().split()
        with self._lock:
            items = list(self._items)
        results: List[MemoryItem] = []
        for item in items:
            if not item.is_alive:
                continue
            haystack = str(item.content).lower() + " " + " ".join(item.tags).lower()
            if all(kw in haystack for kw in keywords):
                results.append(item)
        results.sort(key=lambda x: x.current_strength, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return a summary dict suitable for logging / monitoring."""
        with self._lock:
            items = list(self._items)
        alive = [it for it in items if it.is_alive]
        strengths = [it.current_strength for it in alive]
        return {
            "total_items": len(items),
            "alive_items": len(alive),
            "capacity": self.CAPACITY,
            "total_added": self._add_count,
            "avg_strength": (sum(strengths) / len(strengths)) if strengths else 0.0,
            "max_strength": max(strengths, default=0.0),
            "min_strength": min(strengths, default=0.0),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_stm_instance: Optional[ShortTermMemory] = None
_stm_lock: threading.Lock = threading.Lock()


def get_short_term_memory() -> ShortTermMemory:
    """Return the process-wide ShortTermMemory singleton."""
    global _stm_instance
    if _stm_instance is None:
        with _stm_lock:
            if _stm_instance is None:
                _stm_instance = ShortTermMemory()
    return _stm_instance
