"""Semantic search engine using TF-IDF-style similarity over knowledge entries."""
from __future__ import annotations
import threading, time, logging, math, re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from src.core.utils.logging_utils import get_logger
logger = get_logger(__name__)


@dataclass
class SearchDocument:
    doc_id: str
    content: str
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class SearchResult:
    doc_id: str
    score: float
    content: str
    category: str
    snippet: str = ""

    def __lt__(self, other: "SearchResult") -> bool:
        return self.score > other.score  # reverse: higher score is better


class SemanticSearchEngine:
    """TF-IDF based semantic search over indexed documents."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._documents: Dict[str, SearchDocument] = {}
        self._term_freq: Dict[str, Dict[str, float]] = {}   # doc_id -> {term: tf}
        self._doc_freq: Dict[str, int] = defaultdict(int)   # term -> num docs
        self._idf_cache: Dict[str, float] = {}
        self._dirty = False

    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if len(t) > 1]

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        freq: Dict[str, float] = defaultdict(float)
        for t in tokens:
            freq[t] += 1.0
        total = len(tokens) or 1
        return {t: v / total for t, v in freq.items()}

    def _invalidate_idf_cache(self) -> None:
        self._idf_cache.clear()
        self._dirty = True

    def _get_idf(self, term: str) -> float:
        if term not in self._idf_cache:
            n_docs = len(self._documents) or 1
            df = self._doc_freq.get(term, 0)
            self._idf_cache[term] = math.log((n_docs + 1) / (df + 1)) + 1.0
        return self._idf_cache[term]

    # ------------------------------------------------------------------
    def index(self, doc: SearchDocument) -> None:
        with self._lock:
            tokens = self._tokenize(doc.content)
            tf = self._compute_tf(tokens)
            # Update doc freq for new terms
            if doc.doc_id not in self._term_freq:
                for term in tf:
                    self._doc_freq[term] += 1
            else:
                old_tf = self._term_freq[doc.doc_id]
                for term in old_tf:
                    if term not in tf:
                        self._doc_freq[term] = max(0, self._doc_freq[term] - 1)
                for term in tf:
                    if term not in old_tf:
                        self._doc_freq[term] += 1
            self._documents[doc.doc_id] = doc
            self._term_freq[doc.doc_id] = tf
            self._invalidate_idf_cache()

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._documents:
                return False
            for term in self._term_freq.get(doc_id, {}):
                self._doc_freq[term] = max(0, self._doc_freq[term] - 1)
            del self._documents[doc_id]
            del self._term_freq[doc_id]
            self._invalidate_idf_cache()
            return True

    def search(self, query: str, top_k: int = 10,
               category_filter: Optional[str] = None) -> List[SearchResult]:
        """Return top-k documents by TF-IDF cosine similarity."""
        with self._lock:
            q_tokens = self._tokenize(query)
            if not q_tokens:
                return []
            q_tf = self._compute_tf(q_tokens)
            scores: List[Tuple[float, str]] = []
            for doc_id, doc in self._documents.items():
                if category_filter and doc.category != category_filter:
                    continue
                tf = self._term_freq.get(doc_id, {})
                score = 0.0
                for term, qtf in q_tf.items():
                    dtf = tf.get(term, 0.0)
                    if dtf > 0:
                        idf = self._get_idf(term)
                        score += qtf * dtf * idf * idf
                if score > 0:
                    scores.append((score, doc_id))
            scores.sort(reverse=True)
            results = []
            for score, doc_id in scores[:top_k]:
                doc = self._documents[doc_id]
                snippet = doc.content[:120] + "..." if len(doc.content) > 120 else doc.content
                results.append(SearchResult(doc_id=doc_id, score=score,
                                            content=doc.content, category=doc.category,
                                            snippet=snippet))
            return results

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"indexed": len(self._documents), "unique_terms": len(self._doc_freq)}


_ENGINE: Optional[SemanticSearchEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_search_engine() -> SemanticSearchEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = SemanticSearchEngine()
        return _ENGINE


def run_tests() -> bool:
    engine = SemanticSearchEngine()
    docs = [
        SearchDocument("d1", "The human heart pumps blood through the circulatory system.", "health"),
        SearchDocument("d2", "Python programming language is widely used in data science.", "tech"),
        SearchDocument("d3", "Machine learning models learn patterns from data.", "tech"),
        SearchDocument("d4", "Blood pressure is a key health indicator for cardiovascular risk.", "health"),
    ]
    for d in docs:
        engine.index(d)
    results = engine.search("blood health heart", top_k=3)
    assert len(results) > 0
    assert results[0].category == "health"
    tech = engine.search("machine learning data", category_filter="tech")
    assert all(r.category == "tech" for r in tech)
    engine.remove("d1")
    assert engine.stats()["indexed"] == 3
    logger.info("SemanticSearchEngine tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_tests()
