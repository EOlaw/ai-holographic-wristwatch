"""Structured fact storage with insertion, querying, and expiration support."""
from __future__ import annotations
import threading, time, logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from src.core.utils.logging_utils import get_logger
logger = get_logger(__name__)


class FactStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    RETRACTED = "retracted"
    UNVERIFIED = "unverified"


@dataclass
class Fact:
    """A single piece of structured knowledge."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[float] = None          # seconds; None = permanent
    status: FactStatus = FactStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl

    def is_active(self) -> bool:
        return self.status == FactStatus.ACTIVE and not self.is_expired()


class FactDatabase:
    """Thread-safe storage for structured (subject, predicate, object) facts."""

    def __init__(self, expiry_check_interval: float = 60.0):
        self._lock = threading.RLock()
        self._facts: Dict[Tuple[str, str, str], Fact] = {}
        self._subject_index: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        self._predicate_index: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        self._running = False
        self._expiry_interval = expiry_check_interval
        self._expiry_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        self._expiry_thread = threading.Thread(
            target=self._expiry_loop, daemon=True, name="fact-db-expiry"
        )
        self._expiry_thread.start()
        logger.info("FactDatabase started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._expiry_thread:
            self._expiry_thread.join(timeout=5.0)
        logger.info("FactDatabase stopped")

    # ------------------------------------------------------------------
    def insert(self, fact: Fact) -> bool:
        """Insert or replace a fact. Returns True if new, False if updated."""
        with self._lock:
            is_new = fact.key not in self._facts
            self._facts[fact.key] = fact
            if is_new:
                self._subject_index[fact.subject].append(fact.key)
                self._predicate_index[fact.predicate].append(fact.key)
            logger.debug("Fact inserted", extra={"key": str(fact.key)})
            return is_new

    def retract(self, subject: str, predicate: str, obj: str) -> bool:
        """Mark a fact as retracted. Returns True if found."""
        key = (subject, predicate, obj)
        with self._lock:
            if key in self._facts:
                self._facts[key].status = FactStatus.RETRACTED
                return True
        return False

    def query_by_subject(self, subject: str) -> List[Fact]:
        """Return all active facts with the given subject."""
        with self._lock:
            keys = self._subject_index.get(subject, [])
            return [self._facts[k] for k in keys if k in self._facts and self._facts[k].is_active()]

    def query_by_predicate(self, predicate: str) -> List[Fact]:
        """Return all active facts with the given predicate."""
        with self._lock:
            keys = self._predicate_index.get(predicate, [])
            return [self._facts[k] for k in keys if k in self._facts and self._facts[k].is_active()]

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None,
              obj: Optional[str] = None, min_confidence: float = 0.0) -> List[Fact]:
        """Flexible SPO query with optional wildcards."""
        with self._lock:
            results = []
            for fact in self._facts.values():
                if not fact.is_active():
                    continue
                if subject and fact.subject != subject:
                    continue
                if predicate and fact.predicate != predicate:
                    continue
                if obj and fact.object != obj:
                    continue
                if fact.confidence < min_confidence:
                    continue
                results.append(fact)
            return results

    def count(self) -> Dict[str, int]:
        with self._lock:
            total = len(self._facts)
            active = sum(1 for f in self._facts.values() if f.is_active())
            return {"total": total, "active": active, "inactive": total - active}

    # ------------------------------------------------------------------
    def _expiry_loop(self) -> None:
        while self._running:
            time.sleep(self._expiry_interval)
            self._expire_facts()

    def _expire_facts(self) -> None:
        with self._lock:
            expired = 0
            for fact in self._facts.values():
                if fact.status == FactStatus.ACTIVE and fact.is_expired():
                    fact.status = FactStatus.EXPIRED
                    expired += 1
            if expired:
                logger.debug(f"Expired {expired} facts")


# Singleton
_FACT_DB: Optional[FactDatabase] = None
_FACT_DB_LOCK = threading.Lock()


def get_fact_database() -> FactDatabase:
    global _FACT_DB
    with _FACT_DB_LOCK:
        if _FACT_DB is None:
            _FACT_DB = FactDatabase()
            _FACT_DB.start()
        return _FACT_DB


def run_tests() -> bool:
    db = FactDatabase()
    db.start()
    f1 = Fact("Paris", "is_capital_of", "France", confidence=1.0, source="geo-kb")
    f2 = Fact("Paris", "has_population", "2_million", confidence=0.9, source="census")
    f3 = Fact("Eiffel_Tower", "located_in", "Paris", confidence=1.0, ttl=2.0)
    db.insert(f1); db.insert(f2); db.insert(f3)
    assert len(db.query_by_subject("Paris")) == 2
    assert len(db.query(predicate="located_in")) == 1
    time.sleep(3.0)  # let f3 expire
    assert len(db.query_by_subject("Eiffel_Tower")) == 0
    counts = db.count()
    assert counts["active"] == 2
    db.stop()
    logger.info("FactDatabase tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_tests()
