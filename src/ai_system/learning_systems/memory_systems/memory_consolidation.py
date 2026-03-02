"""Memory Consolidation module — AI Holographic Wristwatch.

Orchestrates the nightly (or on-demand) transfer of important short-term and
episodic memories into long-term storage, with deduplication, merging, and
pruning of weak traces.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConsolidationReport:
    """Summary produced after a consolidation cycle."""

    episodes_processed: int
    memories_created: int
    memories_merged: int
    memories_pruned: int
    duration_secs: float
    timestamp: float

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ConsolidationReport(episodes={self.episodes_processed}, "
            f"created={self.memories_created}, merged={self.memories_merged}, "
            f"pruned={self.memories_pruned}, dur={self.duration_secs:.2f}s)"
        )


class MemoryConsolidator:
    """Coordinates cross-system memory consolidation.

    Workflow
    --------
    1. Pull alive items from STM and recent episodes from EpisodicMemory.
    2. Deduplicate / merge similar memories.
    3. Store important ones into LongTermMemory.
    4. Prune weak LTM memories below threshold.
    5. Return a ConsolidationReport.
    """

    SIMILARITY_THRESHOLD: float = 0.7

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._consolidation_count: int = 0
        self._last_run: float = 0.0
        logger.debug("MemoryConsolidator initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consolidate_session(self, session_memories: List[Dict]) -> List[Dict]:
        """Deduplicate and strengthen memories from a single session.

        Parameters
        ----------
        session_memories:
            List of memory dicts, each with at least ``content`` and
            optionally ``strength`` / ``importance`` keys.

        Returns
        -------
        List[Dict]
            Deduplicated and merged list, with ``strength`` values updated.
        """
        if not session_memories:
            return []

        consolidated: List[Dict] = []
        for incoming in session_memories:
            matched = False
            for existing in consolidated:
                sim = self._calculate_similarity(incoming, existing)
                if sim >= self.SIMILARITY_THRESHOLD:
                    merged = self.merge_similar_memories(existing, incoming)
                    # Replace in-place
                    idx = consolidated.index(existing)
                    consolidated[idx] = merged
                    matched = True
                    break
            if not matched:
                consolidated.append(dict(incoming))

        logger.debug(
            "consolidate_session: %d → %d memories",
            len(session_memories), len(consolidated),
        )
        return consolidated

    def prune_weak_memories(
        self, memories: List[Dict], threshold: float = 0.1
    ) -> int:
        """Remove in-place any memories whose strength is below *threshold*.

        Parameters
        ----------
        memories:
            Mutable list of memory dicts.  Modified in-place.
        threshold:
            Minimum acceptable strength.

        Returns
        -------
        int
            Number of memories removed.
        """
        before = len(memories)
        to_keep = [
            m for m in memories
            if m.get("strength", m.get("importance", 0.0)) >= threshold
        ]
        removed = before - len(to_keep)
        memories[:] = to_keep
        if removed:
            logger.debug("prune_weak_memories removed %d items (threshold=%.2f)", removed, threshold)
        return removed

    def merge_similar_memories(self, memory_a: Dict, memory_b: Dict) -> Dict:
        """Combine two similar memories into one.

        Strategy:
        * Concatenate content strings (deduplicated words).
        * Take the maximum of their strength / importance values.
        * Union the tag sets.

        Parameters
        ----------
        memory_a, memory_b:
            Memory dicts with at least ``content`` key.

        Returns
        -------
        Dict
            A new merged memory dict.
        """
        content_a = str(memory_a.get("content", ""))
        content_b = str(memory_b.get("content", ""))
        # Produce a merged content string with unique words preserved in order
        words_a = content_a.split()
        words_b = content_b.split()
        seen: set = set()
        merged_words: List[str] = []
        for w in words_a + words_b:
            if w.lower() not in seen:
                merged_words.append(w)
                seen.add(w.lower())
        merged_content = " ".join(merged_words)

        strength_a = memory_a.get("strength", memory_a.get("importance", 0.0))
        strength_b = memory_b.get("strength", memory_b.get("importance", 0.0))

        tags_a = set(memory_a.get("tags", []))
        tags_b = set(memory_b.get("tags", []))

        merged: Dict[str, Any] = {
            "content": merged_content,
            "strength": max(strength_a, strength_b),
            "tags": list(tags_a | tags_b),
            "category": memory_a.get("category", memory_b.get("category", "general")),
            "source": "consolidation_merge",
        }
        return merged

    def run_consolidation_cycle(
        self,
        ltm: Any,   # LongTermMemory
        stm: Any,   # ShortTermMemory
        episodic: Any,  # EpisodicMemory
    ) -> ConsolidationReport:
        """Execute a full consolidation pass across all memory systems.

        Steps
        -----
        1. Collect STM alive items (context window).
        2. Collect recent episodic episodes (last 20).
        3. Convert to memory dicts and deduplicate.
        4. Store important memories (strength >= 0.3) into LTM.
        5. Apply Ebbinghaus forgetting curve to LTM.
        6. Prune STM decayed items.

        Parameters
        ----------
        ltm:
            LongTermMemory instance.
        stm:
            ShortTermMemory instance.
        episodic:
            EpisodicMemory instance.

        Returns
        -------
        ConsolidationReport
        """
        t_start = time.time()
        memories_created = 0
        memories_merged = 0
        memories_pruned = 0

        # --- 1. Gather STM items ---
        stm_items = stm.get_context_window()
        stm_dicts: List[Dict] = []
        for item in stm_items:
            stm_dicts.append(
                {
                    "content": str(item.content),
                    "strength": item.current_strength,
                    "tags": list(item.tags),
                    "category": "stm_item",
                    "source": "short_term_memory",
                }
            )

        # --- 2. Gather episodic episodes ---
        recent_episodes = episodic.replay_recent_episodes(n=20)
        episode_dicts: List[Dict] = []
        for ep in recent_episodes:
            summary = episodic.consolidate_to_long_term(ep)
            episode_dicts.append(summary)

        episodes_processed = len(recent_episodes)

        # --- 3. Deduplicate ---
        all_dicts = stm_dicts + episode_dicts
        consolidated = self.consolidate_session(all_dicts)
        memories_merged = len(all_dicts) - len(consolidated)

        # --- 4. Store important ones into LTM ---
        for mem_dict in consolidated:
            strength = mem_dict.get("strength", mem_dict.get("initial_strength", 0.0))
            if strength >= 0.3:
                ltm.store(
                    content=mem_dict.get("content", ""),
                    category=mem_dict.get("category", "general"),
                    initial_strength=strength,
                    source=mem_dict.get("source", "consolidation"),
                    tags=mem_dict.get("tags", []),
                )
                memories_created += 1

        # --- 5. Forgetting curve (1 day = default) ---
        ltm.apply_forgetting_curve(days_elapsed=1.0)

        # --- 6. Prune STM ---
        memories_pruned = stm.prune_decayed()

        duration = time.time() - t_start

        with self._lock:
            self._consolidation_count += 1
            self._last_run = time.time()

        report = ConsolidationReport(
            episodes_processed=episodes_processed,
            memories_created=memories_created,
            memories_merged=memories_merged,
            memories_pruned=memories_pruned,
            duration_secs=duration,
            timestamp=time.time(),
        )
        logger.info("Consolidation cycle complete: %s", report)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_similarity(self, m1: Dict, m2: Dict) -> float:
        """Jaccard-like similarity on lowercased content words.

        Parameters
        ----------
        m1, m2:
            Memory dicts with ``content`` key.

        Returns
        -------
        float
            Jaccard similarity in [0, 1].
        """
        words1 = set(str(m1.get("content", "")).lower().split())
        words2 = set(str(m2.get("content", "")).lower().split())
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return a summary dict suitable for logging / monitoring."""
        return {
            "consolidation_cycles": self._consolidation_count,
            "last_run": self._last_run,
            "similarity_threshold": self.SIMILARITY_THRESHOLD,
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_consolidator_instance: Optional[MemoryConsolidator] = None
_consolidator_lock: threading.Lock = threading.Lock()


def get_memory_consolidator() -> MemoryConsolidator:
    """Return the process-wide MemoryConsolidator singleton."""
    global _consolidator_instance
    if _consolidator_instance is None:
        with _consolidator_lock:
            if _consolidator_instance is None:
                _consolidator_instance = MemoryConsolidator()
    return _consolidator_instance
