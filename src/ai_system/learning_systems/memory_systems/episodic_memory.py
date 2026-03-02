"""Episodic Memory module — AI Holographic Wristwatch.

Stores autobiographical episodes (events experienced by the AI/user) with
emotional valence and contextual metadata.  Supports cue-based recall and
automatic consolidation summaries for long-term memory.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionalValence(Enum):
    """Broad emotional colouring of an episode."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


@dataclass
class Episode:
    """A single autobiographical episode."""

    id: str
    event_type: str
    content: str
    context: Dict
    emotion: EmotionalValence
    timestamp: float
    importance: float  # 0.0 – 1.0
    replay_count: int = 0
    tags: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Episode(id={self.id[:8]}, type={self.event_type!r}, "
            f"emotion={self.emotion.value}, importance={self.importance:.2f})"
        )


class EpisodicMemory:
    """Store and recall autobiographical episodes.

    Episodes are kept in insertion order.  Recall is scored by a combination
    of keyword match, tag overlap, event-type match, importance, and recency.
    """

    def __init__(self) -> None:
        self._episodes: List[Episode] = []
        self._lock: threading.Lock = threading.Lock()
        self._encode_count: int = 0
        logger.debug("EpisodicMemory initialised")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_episode(
        self,
        event_type: str,
        content: str,
        context: Optional[Dict] = None,
        emotion: EmotionalValence = EmotionalValence.NEUTRAL,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> Episode:
        """Store a new episode.

        Parameters
        ----------
        event_type:
            High-level category (e.g. "conversation", "health_alert",
            "navigation_complete").
        content:
            Human-readable narrative of what happened.
        context:
            Arbitrary key/value metadata (location, device state, etc.).
        emotion:
            Predominant emotional valence.
        importance:
            Salience weight in [0, 1].
        tags:
            Optional keyword labels.

        Returns
        -------
        Episode
            The encoded episode.
        """
        importance = max(0.0, min(1.0, importance))
        episode = Episode(
            id=str(uuid.uuid4()),
            event_type=event_type,
            content=content,
            context=context or {},
            emotion=emotion,
            timestamp=time.time(),
            importance=importance,
            tags=tags or [],
        )
        with self._lock:
            self._episodes.append(episode)
            self._encode_count += 1
        logger.debug(
            "Episodic encode #%d: type=%s emotion=%s importance=%.2f",
            self._encode_count, event_type, emotion.value, importance,
        )
        return episode

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall_episode(self, cue: str, n_results: int = 5) -> List[Episode]:
        """Retrieve episodes matching a textual cue.

        Scoring formula:
            score = content_kw_match * 0.4
                  + tag_match * 0.2
                  + type_match * 0.2
                  + importance * 0.2

        Parameters
        ----------
        cue:
            Free-text retrieval cue.
        n_results:
            Maximum episodes to return.

        Returns
        -------
        List[Episode]
            Top matching episodes sorted by score descending.
        """
        with self._lock:
            episodes = list(self._episodes)

        scored: List[tuple[float, Episode]] = []
        for ep in episodes:
            score = self._score_episode(ep, cue)
            if score > 0:
                scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [ep for _, ep in scored[:n_results]]

        # Mark recalled episodes
        recalled_ids = {ep.id for ep in top}
        with self._lock:
            for ep in self._episodes:
                if ep.id in recalled_ids:
                    ep.replay_count += 1

        return top

    def replay_recent_episodes(self, n: int = 5) -> List[Episode]:
        """Return the *n* most recently encoded episodes."""
        with self._lock:
            recent = list(self._episodes[-n:])
        recent.sort(key=lambda ep: ep.timestamp, reverse=True)
        return recent

    def get_episodes_by_type(self, event_type: str) -> List[Episode]:
        """Return all episodes with *event_type*, newest first."""
        with self._lock:
            matching = [ep for ep in self._episodes if ep.event_type == event_type]
        matching.sort(key=lambda ep: ep.timestamp, reverse=True)
        return matching

    def get_emotional_episodes(self, valence: EmotionalValence) -> List[Episode]:
        """Return all episodes with the given emotional valence, newest first."""
        with self._lock:
            matching = [ep for ep in self._episodes if ep.emotion == valence]
        matching.sort(key=lambda ep: ep.timestamp, reverse=True)
        return matching

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate_to_long_term(self, episode: Episode) -> Dict:
        """Create a summary dict suitable for LongTermMemory.store().

        Parameters
        ----------
        episode:
            The episode to summarise.

        Returns
        -------
        Dict
            Keys: content, category, initial_strength, source, tags.
        """
        # Emotional events are remembered more strongly
        emotion_boost = {
            EmotionalValence.POSITIVE: 0.15,
            EmotionalValence.NEGATIVE: 0.20,
            EmotionalValence.MIXED: 0.10,
            EmotionalValence.NEUTRAL: 0.0,
        }
        strength = min(
            1.0,
            episode.importance + emotion_boost.get(episode.emotion, 0.0),
        )
        summary = (
            f"[{episode.event_type}] {episode.content} "
            f"(emotion={episode.emotion.value}, replays={episode.replay_count})"
        )
        return {
            "content": summary,
            "category": episode.event_type,
            "initial_strength": strength,
            "source": "episodic_consolidation",
            "tags": episode.tags + [episode.emotion.value],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_episode(self, episode: Episode, cue: str) -> float:
        """Compute a relevance score for *episode* given *cue*."""
        if not cue:
            return episode.importance

        keywords = set(cue.lower().split())

        # Content keyword match
        content_words = set(episode.content.lower().split())
        content_score = (
            len(keywords & content_words) / len(keywords) if keywords else 0.0
        )

        # Tag match
        tag_words = set(" ".join(episode.tags).lower().split())
        tag_score = (
            len(keywords & tag_words) / len(keywords) if keywords else 0.0
        )

        # Event-type match
        type_score = 1.0 if any(kw in episode.event_type.lower() for kw in keywords) else 0.0

        score = (
            content_score * 0.4
            + tag_score * 0.2
            + type_score * 0.2
            + episode.importance * 0.2
        )
        return score

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return a summary dict suitable for logging / monitoring."""
        with self._lock:
            episodes = list(self._episodes)
        emotion_counts: Dict[str, int] = {v.value: 0 for v in EmotionalValence}
        type_counts: Dict[str, int] = {}
        for ep in episodes:
            emotion_counts[ep.emotion.value] += 1
            type_counts[ep.event_type] = type_counts.get(ep.event_type, 0) + 1
        return {
            "total_episodes": len(episodes),
            "total_encoded": self._encode_count,
            "emotion_distribution": emotion_counts,
            "event_types": type_counts,
            "avg_importance": (
                sum(ep.importance for ep in episodes) / len(episodes)
            ) if episodes else 0.0,
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_episodic_instance: Optional[EpisodicMemory] = None
_episodic_lock: threading.Lock = threading.Lock()


def get_episodic_memory() -> EpisodicMemory:
    """Return the process-wide EpisodicMemory singleton."""
    global _episodic_instance
    if _episodic_instance is None:
        with _episodic_lock:
            if _episodic_instance is None:
                _episodic_instance = EpisodicMemory()
    return _episodic_instance
