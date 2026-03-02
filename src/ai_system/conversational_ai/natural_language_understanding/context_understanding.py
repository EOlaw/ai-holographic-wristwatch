"""Context Understanding — AI Holographic Wristwatch

Maintains a rolling window of dialogue context, resolves pronominal
co-references ("it", "that", "them" etc.) to previously mentioned entities,
and detects topic shifts — enabling the NLU pipeline to interpret each new
utterance in light of everything said so far.
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
# Pronoun / anaphora sets
# ---------------------------------------------------------------------------

PRONOUN_MAP: Set[str] = {
    "it", "that", "this", "them", "they", "he", "she", "him", "her",
}

# Pronouns that refer to a person (singular / plural)
_PERSON_PRONOUNS: Set[str] = {"he", "she", "him", "her", "they", "them"}
# Pronouns that refer to a thing / concept
_THING_PRONOUNS: Set[str] = {"it", "that", "this"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContextState:
    """Snapshot of dialogue context at a given turn."""
    topic: str = "unknown"
    entities: Dict[str, str] = field(default_factory=dict)
    last_intent: str = ""
    turn_number: int = 0
    timestamp: float = field(default_factory=time.time)
    # Maps pronoun -> resolved referent (e.g. "it" -> "Bohemian Rhapsody")
    references: Dict[str, str] = field(default_factory=dict)

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def to_dict(self) -> Dict:
        return {
            "topic": self.topic,
            "entities": dict(self.entities),
            "last_intent": self.last_intent,
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "age_seconds": round(self.age_seconds(), 2),
            "references": dict(self.references),
        }


# ---------------------------------------------------------------------------
# Domain keyword hints (same lightweight approach as topic_tracking)
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "health":       ["heart", "rate", "steps", "calories", "sleep", "pulse", "bpm",
                     "workout", "exercise", "blood", "pressure", "oxygen"],
    "navigation":   ["directions", "navigate", "map", "route", "location", "address",
                     "turn", "left", "right", "miles", "kilometers"],
    "media":        ["play", "pause", "stop", "music", "song", "album", "artist",
                     "playlist", "podcast", "video", "track"],
    "communication":["call", "message", "text", "email", "contact", "phone", "send"],
    "timer":        ["timer", "alarm", "countdown", "minutes", "seconds", "schedule"],
    "hologram":     ["hologram", "display", "projection", "show", "brightness", "render"],
    "weather":      ["weather", "temperature", "forecast", "rain", "sunny", "cloudy"],
    "general":      [],
}

# Shift detection: if Jaccard similarity between current and new keywords < threshold
_SHIFT_JACCARD_THRESHOLD: float = 0.15


# ---------------------------------------------------------------------------
# Context Understanding
# ---------------------------------------------------------------------------

class ContextUnderstanding:
    """Maintains rolling dialogue context and resolves co-references.

    Thread-safe via ``threading.RLock``.
    """

    def __init__(self) -> None:
        self._history: Deque[ContextState] = deque(maxlen=20)
        self._lock: threading.RLock = threading.RLock()
        # Maps pronoun -> most recent referent
        self._coreference_map: Dict[str, str] = {}
        # Maps entity_type -> entity_value (e.g. "person" -> "Sarah")
        self._entity_map: Dict[str, str] = {}
        self._turn_counter: int = 0
        self._current_state: Optional[ContextState] = None
        logger.debug("ContextUnderstanding initialised.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_context(
        self,
        utterance: str,
        intent: str = "",
        entities: Optional[Dict] = None,
    ) -> ContextState:
        """Record a new utterance and return the updated ``ContextState``.

        Args:
            utterance: The raw (or already coreference-resolved) utterance.
            intent:    The NLU-recognised intent (optional).
            entities:  Extracted entity dict (optional, e.g. {"person": "Sarah"}).

        Returns:
            The newly created ``ContextState`` snapshot.
        """
        with self._lock:
            if entities is None:
                entities = {}

            self._turn_counter += 1
            topic = self._infer_topic(utterance, intent)

            # Update entity map and coreference map from newly extracted entities
            self._update_coreference_map(entities)

            # Resolve coreferences in the utterance to build reference snapshot
            resolved_refs = self._build_reference_snapshot(utterance)

            state = ContextState(
                topic=topic,
                entities=dict(entities),
                last_intent=intent,
                turn_number=self._turn_counter,
                references=resolved_refs,
            )

            if self._current_state is not None:
                self._history.append(self._current_state)

            self._current_state = state
            logger.debug(
                "Context updated: turn=%d intent=%s topic=%s entities=%s",
                self._turn_counter, intent, topic, entities,
            )
            return state

    def resolve_coreference(self, utterance: str) -> str:
        """Replace pronouns in *utterance* with their resolved referents.

        Pronouns in ``PRONOUN_MAP`` are substituted if a referent is known.
        Unresolved pronouns are left in place.

        Args:
            utterance: Raw user utterance.

        Returns:
            The utterance with pronouns replaced where possible.
        """
        with self._lock:
            if not self._coreference_map:
                return utterance

            tokens = utterance.split()
            resolved_tokens: List[str] = []

            for token in tokens:
                clean_token = re.sub(r"[^\w']", "", token).lower()
                if clean_token in PRONOUN_MAP and clean_token in self._coreference_map:
                    referent = self._coreference_map[clean_token]
                    # Preserve trailing punctuation from original token
                    suffix = re.sub(r"[\w']", "", token)
                    resolved_tokens.append(referent + suffix)
                    logger.debug(
                        "Coreference resolved: '%s' -> '%s'", clean_token, referent
                    )
                else:
                    resolved_tokens.append(token)

            return " ".join(resolved_tokens)

    def get_current_topic(self) -> str:
        """Return the topic name from the most recent context state."""
        with self._lock:
            if self._current_state is None:
                return "unknown"
            return self._current_state.topic

    def get_context_window(self, n: int = 5) -> List[ContextState]:
        """Return the *n* most recent context states (most recent last).

        Args:
            n: Number of states to return. Clamped to history length.

        Returns:
            List of up to *n* ``ContextState`` objects, oldest first.
        """
        with self._lock:
            history_list = list(self._history)
            window = history_list[-(n - 1):] if len(history_list) >= n else history_list
            if self._current_state is not None:
                window = window + [self._current_state]
            return window[-n:]

    def detect_topic_shift(self) -> bool:
        """Return True if the most recent turn represents a topic shift.

        Compares the topic/keywords of the current state against the
        immediately preceding state using a Jaccard similarity heuristic.
        """
        with self._lock:
            if self._current_state is None or len(self._history) == 0:
                return False
            prev = self._history[-1]
            current = self._current_state

            # Compare topics directly
            if prev.topic == current.topic:
                return False

            # Compute keyword overlap via entity keys + topic tokens
            prev_kw = set(prev.topic.split("_")) | set(prev.entities.keys())
            curr_kw = set(current.topic.split("_")) | set(current.entities.keys())

            if not prev_kw and not curr_kw:
                return False

            intersection = len(prev_kw & curr_kw)
            union = len(prev_kw | curr_kw)
            similarity = intersection / union if union > 0 else 0.0

            is_shift = similarity < _SHIFT_JACCARD_THRESHOLD
            if is_shift:
                logger.info(
                    "Topic shift detected: '%s' -> '%s' (similarity=%.2f)",
                    prev.topic, current.topic, similarity,
                )
            return is_shift

    def get_entity(self, entity_type: str) -> Optional[str]:
        """Retrieve the most recently mentioned entity of the given type."""
        with self._lock:
            return self._entity_map.get(entity_type)

    def get_coreference_map(self) -> Dict[str, str]:
        """Return a copy of the current coreference map."""
        with self._lock:
            return dict(self._coreference_map)

    def get_stats(self) -> Dict:
        """Return summary statistics about the context window."""
        with self._lock:
            return {
                "turn_count": self._turn_counter,
                "history_depth": len(self._history),
                "current_topic": self._current_state.topic if self._current_state else None,
                "entity_count": len(self._entity_map),
                "coreference_map": dict(self._coreference_map),
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_coreference_map(self, entities: Dict) -> None:
        """Update the pronoun -> referent map from *entities*.

        Convention for entity dict keys:
          - "person" -> maps to "he"/"she"/"him"/"her"/"they"/"them"
          - any other key -> maps to "it"/"that"/"this"
        """
        for entity_type, entity_value in entities.items():
            if not entity_value:
                continue
            # Store in entity map
            self._entity_map[entity_type] = entity_value
            # Map person pronouns
            if entity_type in ("person", "contact", "user", "name"):
                for pronoun in _PERSON_PRONOUNS:
                    self._coreference_map[pronoun] = entity_value
            # Map thing pronouns
            else:
                for pronoun in _THING_PRONOUNS:
                    self._coreference_map[pronoun] = entity_value

    def _build_reference_snapshot(self, utterance: str) -> Dict[str, str]:
        """Return a snapshot of pronoun resolutions applicable to *utterance*."""
        snapshot: Dict[str, str] = {}
        tokens = {re.sub(r"[^\w']", "", t).lower() for t in utterance.split()}
        for pronoun in PRONOUN_MAP:
            if pronoun in tokens and pronoun in self._coreference_map:
                snapshot[pronoun] = self._coreference_map[pronoun]
        return snapshot

    def _infer_topic(self, utterance: str, intent: str) -> str:
        """Infer a topic label from the intent name or utterance keywords."""
        if intent:
            # Use the intent domain part if available (e.g. "check_heart_rate" -> "health")
            for domain, kws in _DOMAIN_KEYWORDS.items():
                if any(kw in intent.lower() for kw in kws):
                    return domain
            # Fall back to first word of intent
            topic_from_intent = intent.split("_")[0] if "_" in intent else intent
            if topic_from_intent:
                return topic_from_intent

        # Keyword scan of utterance
        lower = utterance.lower()
        best_domain = "general"
        best_score = 0
        for domain, kws in _DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in kws if kw in lower)
            if score > best_score:
                best_score = score
                best_domain = domain
        return best_domain


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[ContextUnderstanding] = None
_instance_lock: threading.Lock = threading.Lock()


def get_context_understanding() -> ContextUnderstanding:
    """Return the process-wide singleton ``ContextUnderstanding``."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ContextUnderstanding()
    return _instance


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    ctx = get_context_understanding()
    print("=== Context Understanding Demo ===\n")

    # Turn 1: introduce a contact
    state1 = ctx.update_context(
        "Call Sarah and tell her I'm on my way",
        intent="call_contact",
        entities={"person": "Sarah"},
    )
    print(f"Turn 1 - Topic: {state1.topic}, Entities: {state1.entities}")
    resolved = ctx.resolve_coreference("Can you call her again?")
    print(f"  Resolved: 'Can you call her again?' -> '{resolved}'")

    # Turn 2: media, new entity
    state2 = ctx.update_context(
        "Play Bohemian Rhapsody",
        intent="play_music",
        entities={"song": "Bohemian Rhapsody"},
    )
    print(f"\nTurn 2 - Topic: {state2.topic}, Entities: {state2.entities}")
    resolved2 = ctx.resolve_coreference("I love it, play it again")
    print(f"  Resolved: 'I love it, play it again' -> '{resolved2}'")

    # Turn 3: health topic shift
    state3 = ctx.update_context(
        "What is my heart rate right now?",
        intent="check_heart_rate",
        entities={},
    )
    print(f"\nTurn 3 - Topic: {state3.topic}")
    print(f"  Topic shift detected: {ctx.detect_topic_shift()}")

    print(f"\nContext window (last 3): {[s.topic for s in ctx.get_context_window(3)]}")
    print(f"\nStats: {ctx.get_stats()}")
    print("\nDemo complete.")
