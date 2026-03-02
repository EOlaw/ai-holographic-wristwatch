"""
Entity Extraction — AI Holographic Wristwatch

Extracts named entities and slot values from user utterances:
- Named Entity Recognition (NER): persons, places, organizations, dates, times
- Slot filling for dialogue parameters (contact names, durations, locations)
- Number and quantity parsing (including health values, times, distances)
- Wristwatch-specific entities (health metrics, hologram commands)
- Normalization of extracted values to canonical forms
"""

from __future__ import annotations

import re
import threading
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class EntityType(Enum):
    PERSON       = "person"
    LOCATION     = "location"
    ORGANIZATION = "organization"
    DATE         = "date"
    TIME         = "time"
    DURATION     = "duration"
    NUMBER       = "number"
    PERCENTAGE   = "percentage"
    HEALTH_METRIC = "health_metric"
    HOLOGRAM_CMD  = "hologram_cmd"
    CONTACT_NAME  = "contact_name"
    REMINDER_TEXT = "reminder_text"


@dataclass
class Entity:
    """A single extracted entity."""
    text: str                          # original text span
    entity_type: EntityType
    value: Optional[str] = None        # normalized value
    start: int = 0
    end: int = 0
    confidence: float = 0.85


@dataclass
class ExtractionResult:
    """All entities extracted from an utterance."""
    utterance: str
    entities: List[Entity] = field(default_factory=list)
    slots: Dict[str, str] = field(default_factory=dict)    # slot_name → value
    timestamp: float = field(default_factory=time.time)

    def get(self, entity_type: EntityType) -> List[Entity]:
        return [e for e in self.entities if e.entity_type == entity_type]

    def get_slot(self, slot_name: str, default: str = "") -> str:
        return self.slots.get(slot_name, default)


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

class TimeParser:
    """Parses time expressions like '5 minutes', '3pm', 'in an hour'."""

    DURATION_PATTERN = re.compile(
        r"(\d+)\s*(second|sec|minute|min|hour|hr|day|week|month)s?",
        re.IGNORECASE
    )
    CLOCK_PATTERN = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)
    RELATIVE_PATTERN = re.compile(
        r"\b(in\s+an?\s+hour|in\s+an?\s+minute|tomorrow|tonight|this\s+(morning|afternoon|evening))\b",
        re.IGNORECASE
    )

    UNIT_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60,
                    "hour": 3600, "hr": 3600, "day": 86400, "week": 604800}

    def parse_duration_seconds(self, text: str) -> Optional[int]:
        m = self.DURATION_PATTERN.search(text)
        if not m:
            return None
        value = int(m.group(1))
        unit  = m.group(2).lower().rstrip("s")
        return value * self.UNIT_SECONDS.get(unit, 60)

    def extract_entities(self, text: str) -> List[Entity]:
        entities = []
        for m in self.DURATION_PATTERN.finditer(text):
            entities.append(Entity(
                text=m.group(0),
                entity_type=EntityType.DURATION,
                value=str(self.parse_duration_seconds(m.group(0))),
                start=m.start(), end=m.end(),
            ))
        for m in self.CLOCK_PATTERN.finditer(text):
            entities.append(Entity(
                text=m.group(0),
                entity_type=EntityType.TIME,
                value=m.group(0),
                start=m.start(), end=m.end(),
            ))
        return entities


class NumberParser:
    """Extracts numeric values and percentages."""

    NUMBER_PATTERN     = re.compile(r"\b\d+(?:\.\d+)?\b")
    PERCENTAGE_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*%")

    def extract_entities(self, text: str) -> List[Entity]:
        entities = []
        for m in self.PERCENTAGE_PATTERN.finditer(text):
            entities.append(Entity(text=m.group(0), entity_type=EntityType.PERCENTAGE,
                                   value=m.group(1), start=m.start(), end=m.end()))
        for m in self.NUMBER_PATTERN.finditer(text):
            # Skip if already captured as percentage
            if not any(e.start <= m.start() < e.end for e in entities):
                entities.append(Entity(text=m.group(0), entity_type=EntityType.NUMBER,
                                       value=m.group(0), start=m.start(), end=m.end()))
        return entities


class ContactExtractor:
    """
    Extracts contact names from communication intents.
    In production, matches against user's contact list via fuzzy matching.
    """

    CALL_PATTERN    = re.compile(r"\b(?:call|phone|ring|facetime)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.IGNORECASE)
    MESSAGE_PATTERN = re.compile(r"\b(?:message|text|send(?:\s+a\s+message)?\s+to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.IGNORECASE)

    def extract_entities(self, text: str) -> List[Entity]:
        entities = []
        for pattern in (self.CALL_PATTERN, self.MESSAGE_PATTERN):
            for m in pattern.finditer(text):
                entities.append(Entity(
                    text=m.group(1),
                    entity_type=EntityType.CONTACT_NAME,
                    value=m.group(1).strip(),
                    start=m.start(1), end=m.end(1),
                ))
        return entities


class HealthMetricExtractor:
    """Extracts health metric targets mentioned by user."""

    METRIC_PATTERNS: List[Tuple[str, str]] = [
        (r"\b(\d+)\s*bpm\b",         "heart_rate"),
        (r"\b(\d+(?:\.\d+)?)\s*%\s+(?:oxygen|spo2|saturation)\b", "spo2"),
        (r"\b(\d+)\s+steps?\b",      "steps"),
        (r"\b(\d+(?:\.\d+)?)\s*(?:degrees?|°)\b", "temperature"),
        (r"\b(\d+(?:\.\d+)?)\s*(?:kg|lb|pounds?|kilograms?)\b", "weight"),
    ]

    def extract_entities(self, text: str) -> List[Entity]:
        entities = []
        for pattern, metric_name in self.METRIC_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=m.group(0),
                    entity_type=EntityType.HEALTH_METRIC,
                    value=f"{metric_name}:{m.group(1)}",
                    start=m.start(), end=m.end(),
                ))
        return entities


# ---------------------------------------------------------------------------
# Entity Extractor
# ---------------------------------------------------------------------------

_GLOBAL_EE: Optional["EntityExtractor"] = None
_GLOBAL_EE_LOCK = threading.Lock()


class EntityExtractor:
    """Orchestrates all sub-extractors and deduplicates overlapping spans."""

    def __init__(self) -> None:
        self._time_parser   = TimeParser()
        self._number_parser = NumberParser()
        self._contact_ext   = ContactExtractor()
        self._health_ext    = HealthMetricExtractor()
        self._lock          = threading.RLock()
        self._extract_count = 0

    def extract(self, utterance: str) -> ExtractionResult:
        with self._lock:
            all_entities: List[Entity] = []
            all_entities += self._time_parser.extract_entities(utterance)
            all_entities += self._number_parser.extract_entities(utterance)
            all_entities += self._contact_ext.extract_entities(utterance)
            all_entities += self._health_ext.extract_entities(utterance)

            # Deduplicate overlapping spans (prefer longer spans)
            all_entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
            deduped: List[Entity] = []
            covered = set()
            for entity in all_entities:
                span = range(entity.start, entity.end)
                if not any(i in covered for i in span):
                    deduped.append(entity)
                    covered.update(span)

            # Build slots dict
            slots: Dict[str, str] = {}
            for e in deduped:
                key = e.entity_type.value
                if e.value:
                    slots[key] = e.value
                    if e.entity_type == EntityType.CONTACT_NAME:
                        slots["contact"] = e.value
                    elif e.entity_type == EntityType.DURATION:
                        slots["duration_seconds"] = e.value

            self._extract_count += 1
            return ExtractionResult(utterance=utterance, entities=deduped, slots=slots)

    def get_stats(self) -> Dict:
        return {"extract_count": self._extract_count}


def get_entity_extractor() -> EntityExtractor:
    global _GLOBAL_EE
    with _GLOBAL_EE_LOCK:
        if _GLOBAL_EE is None:
            _GLOBAL_EE = EntityExtractor()
        return _GLOBAL_EE


def run_entity_extraction_tests() -> bool:
    ee = EntityExtractor()

    r1 = ee.extract("Set a timer for 5 minutes")
    assert any(e.entity_type == EntityType.DURATION for e in r1.entities), \
        f"Expected DURATION entity, got: {[e.entity_type for e in r1.entities]}"

    r2 = ee.extract("Call Sarah at 3pm")
    contacts = [e for e in r2.entities if e.entity_type == EntityType.CONTACT_NAME]
    assert contacts, f"Expected CONTACT entity"
    assert contacts[0].value == "Sarah"

    r3 = ee.extract("My heart rate is 85 bpm, is that normal?")
    metrics = [e for e in r3.entities if e.entity_type == EntityType.HEALTH_METRIC]
    assert metrics

    logger.info("EntityExtractor tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_entity_extraction_tests()
    ee = get_entity_extractor()
    samples = [
        "Remind me to take my medication in 2 hours",
        "Message John that I'll be 15 minutes late",
        "Set an alarm for 7:30am tomorrow",
    ]
    for s in samples:
        r = ee.extract(s)
        print(f"  '{s}'")
        for e in r.entities:
            print(f"    {e.entity_type.value}: '{e.text}' → {e.value}")
