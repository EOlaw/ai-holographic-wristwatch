"""
Knowledge System Interface Contracts for AI Holographic Wristwatch System

Defines abstract contracts for the AI knowledge management subsystem:
knowledge storage, querying, updating, and real-time information integration.
All concrete knowledge backends (local graph DB, vector store, cloud knowledge
service) implement these interfaces, enabling transparent backend swapping.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ..exceptions import KnowledgeError


# ============================================================================
# Enumerations
# ============================================================================

class KnowledgeType(Enum):
    """Category of stored knowledge."""
    FACTUAL = "factual"             # Objective world knowledge
    PROCEDURAL = "procedural"       # How to do things
    PERSONAL = "personal"           # User-specific facts and preferences
    CONTEXTUAL = "contextual"       # Short-lived situational knowledge
    DOMAIN = "domain"               # Specialist domain knowledge
    REAL_TIME = "real_time"         # Live external data (weather, news)


class KnowledgeConfidence(Enum):
    """Reliability tier for a knowledge entry."""
    VERIFIED = "verified"           # Confirmed from authoritative source
    HIGH = "high"                   # Strong evidence
    MEDIUM = "medium"               # Moderate evidence
    LOW = "low"                     # Weak or inferred
    UNCERTAIN = "uncertain"         # Should be treated as speculation


class RetrievalStrategy(Enum):
    """Method used to retrieve knowledge."""
    EXACT_MATCH = "exact_match"
    SEMANTIC_SEARCH = "semantic_search"
    GRAPH_TRAVERSAL = "graph_traversal"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


# ============================================================================
# Data Containers
# ============================================================================

@dataclass
class Fact:
    """A single unit of knowledge stored in the knowledge base."""
    fact_id: str
    subject: str
    predicate: str
    object_value: Any
    knowledge_type: KnowledgeType
    confidence: KnowledgeConfidence = KnowledgeConfidence.HIGH
    source: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Return True if this fact has a TTL and it has elapsed."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


@dataclass
class QueryContext:
    """Context hints that improve knowledge retrieval relevance."""
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    location: Optional[str] = None
    current_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_intent: Optional[str] = None
    domain_filter: Optional[str] = None
    language: str = "en"
    max_results: int = 10
    min_confidence: KnowledgeConfidence = KnowledgeConfidence.LOW
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID


@dataclass
class KnowledgeResult:
    """Response from a knowledge base query."""
    query: str
    facts: List[Fact]
    retrieval_time_ms: float
    strategy_used: RetrievalStrategy
    total_matches: int
    confidence_weighted_score: float    # 0.0–1.0 aggregate quality
    related_queries: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_best_fact(self) -> Optional[Fact]:
        """Return the fact with the highest confidence, or None."""
        if not self.facts:
            return None
        confidence_order = [
            KnowledgeConfidence.VERIFIED, KnowledgeConfidence.HIGH,
            KnowledgeConfidence.MEDIUM, KnowledgeConfidence.LOW,
            KnowledgeConfidence.UNCERTAIN
        ]
        return min(self.facts, key=lambda f: confidence_order.index(f.confidence))


@dataclass
class KnowledgeStats:
    """Operational statistics for the knowledge base."""
    total_facts: int
    facts_by_type: Dict[str, int]
    facts_by_confidence: Dict[str, int]
    storage_size_mb: float
    index_size_mb: float
    average_query_time_ms: float
    cache_hit_rate: float
    last_updated: datetime


@dataclass
class KnowledgeUpdateResult:
    """Result of a knowledge base write operation."""
    fact_id: str
    operation: str          # "create", "update", "delete"
    success: bool
    previous_value: Optional[Any] = None
    conflict_detected: bool = False
    conflict_resolution: Optional[str] = None


@dataclass
class KnowledgeGraphNode:
    """A node in the knowledge graph (entity)."""
    node_id: str
    entity_type: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None   # vector representation


@dataclass
class KnowledgeGraphEdge:
    """A directed edge in the knowledge graph (relation)."""
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Knowledge Base Interface
# ============================================================================

class KnowledgeBaseInterface(ABC):
    """
    Abstract contract for the knowledge storage and retrieval system.

    Implementations may use graph databases (Neo4j, ArangoDB), vector stores
    (Chroma, Pinecone, Weaviate), relational databases, or hybrid approaches.
    The interface is backend-agnostic.
    """

    @abstractmethod
    async def query(self, query: str,
                    context: Optional[QueryContext] = None) -> KnowledgeResult:
        """
        Retrieve relevant facts for a natural language query.

        Args:
            query:   Natural language question or search phrase.
            context: Optional query context for relevance filtering.
        Returns:
            KnowledgeResult with matching facts and metadata.
        Raises:
            KnowledgeError: If retrieval fails.
        """
        ...

    @abstractmethod
    async def store(self, fact: Fact) -> str:
        """
        Store a new fact in the knowledge base.

        Args:
            fact: The Fact to store.
        Returns:
            The assigned fact_id (may differ from input if auto-generated).
        Raises:
            KnowledgeError: If storage fails.
        """
        ...

    @abstractmethod
    async def update(self, fact_id: str,
                     updates: Dict[str, Any]) -> KnowledgeUpdateResult:
        """
        Update an existing fact by ID.

        Args:
            fact_id: ID of the fact to update.
            updates: Dict of fields to change.
        Returns:
            KnowledgeUpdateResult with outcome.
        """
        ...

    @abstractmethod
    async def delete(self, fact_id: str) -> bool:
        """
        Delete a fact by ID.

        Args:
            fact_id: Unique identifier of the fact.
        Returns:
            True if found and deleted; False if not found.
        """
        ...

    @abstractmethod
    async def get_fact(self, fact_id: str) -> Optional[Fact]:
        """
        Retrieve a specific fact by ID.

        Args:
            fact_id: Unique identifier.
        Returns:
            Fact if found, None otherwise.
        """
        ...

    @abstractmethod
    async def batch_store(self, facts: List[Fact]) -> List[str]:
        """
        Store multiple facts efficiently in a single operation.

        Args:
            facts: List of Facts to persist.
        Returns:
            List of assigned fact_ids in the same order.
        """
        ...

    @abstractmethod
    async def search_by_entity(self, entity: str,
                               knowledge_type: Optional[KnowledgeType] = None
                               ) -> List[Fact]:
        """
        Find all facts about a specific entity.

        Args:
            entity:         Subject or object entity name.
            knowledge_type: Optional filter by knowledge category.
        Returns:
            List of matching Facts.
        """
        ...

    @abstractmethod
    async def get_related_entities(self, entity: str,
                                   max_depth: int = 2) -> List[str]:
        """
        Traverse the knowledge graph to find entities related to the given one.

        Args:
            entity:    Starting entity name.
            max_depth: Maximum graph traversal depth.
        Returns:
            List of related entity names.
        """
        ...

    @abstractmethod
    def get_knowledge_stats(self) -> KnowledgeStats:
        """Return operational statistics for the knowledge base."""
        ...

    @abstractmethod
    async def clear_expired(self) -> int:
        """
        Remove all expired facts from the knowledge base.

        Returns:
            Number of facts deleted.
        """
        ...


# ============================================================================
# Personal Knowledge Interface
# ============================================================================

class PersonalKnowledgeInterface(ABC):
    """
    Contract for user-specific knowledge (preferences, habits, history).

    Personal knowledge is always scoped to a user_id and subject to stricter
    privacy controls than general knowledge.
    """

    @abstractmethod
    async def learn_preference(self, user_id: str, category: str,
                               preference: Any,
                               confidence: float = 0.8) -> str:
        """
        Record a user preference discovered through interaction.

        Args:
            user_id:    User identifier.
            category:   Preference category (e.g., "music_genre", "wake_time").
            preference: The preference value.
            confidence: How confident we are in this preference (0.0–1.0).
        Returns:
            fact_id of the stored preference.
        """
        ...

    @abstractmethod
    async def get_preference(self, user_id: str,
                             category: str) -> Optional[Any]:
        """
        Retrieve a user's preference for a category.

        Args:
            user_id:  User identifier.
            category: Preference category to look up.
        Returns:
            Preference value, or None if not known.
        """
        ...

    @abstractmethod
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Return a comprehensive user profile from personal knowledge.

        Args:
            user_id: User identifier.
        Returns:
            Dict with preferences, habits, health baselines, and history.
        """
        ...

    @abstractmethod
    async def forget_user_data(self, user_id: str,
                               categories: Optional[List[str]] = None) -> int:
        """
        Delete personal knowledge for a user (right-to-be-forgotten).

        Args:
            user_id:    User whose data to delete.
            categories: Optional list of categories to delete; deletes all if None.
        Returns:
            Number of facts deleted.
        """
        ...


# ============================================================================
# Real-Time Information Interface
# ============================================================================

class RealTimeInformationInterface(ABC):
    """Contract for integrating live external data into the knowledge system."""

    @abstractmethod
    async def get_current_weather(self, location: str) -> Dict[str, Any]:
        """Return current weather conditions for a location."""
        ...

    @abstractmethod
    async def get_news_headlines(self, category: Optional[str] = None,
                                  max_results: int = 5) -> List[Dict[str, Any]]:
        """Return current news headlines, optionally filtered by category."""
        ...

    @abstractmethod
    async def get_calendar_events(self, user_id: str,
                                   hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Return upcoming calendar events for the user."""
        ...

    @abstractmethod
    async def get_live_data(self, data_type: str,
                             parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generic live data fetch for arbitrary real-time sources.

        Args:
            data_type:  Type identifier (e.g., "stock_price", "traffic").
            parameters: Query parameters for the data source.
        Returns:
            Dict with fetched data and timestamp.
        """
        ...

    @abstractmethod
    async def subscribe_to_updates(self, data_type: str,
                                    callback: Any,
                                    interval_seconds: float = 60.0) -> str:
        """
        Subscribe to periodic updates for a data type.

        Args:
            data_type:         Data type to subscribe to.
            callback:          Async callable receiving update dicts.
            interval_seconds:  Update interval.
        Returns:
            Subscription ID for later cancellation.
        """
        ...

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Cancel a real-time data subscription."""
        ...


# ============================================================================
# Knowledge Graph Interface
# ============================================================================

class KnowledgeGraphInterface(ABC):
    """Contract for graph-based knowledge representation and traversal."""

    @abstractmethod
    async def add_node(self, node: KnowledgeGraphNode) -> str:
        """Add an entity node to the graph. Returns assigned node_id."""
        ...

    @abstractmethod
    async def add_edge(self, edge: KnowledgeGraphEdge) -> str:
        """Add a directed relation edge. Returns assigned edge_id."""
        ...

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[KnowledgeGraphNode]:
        """Retrieve a node by ID."""
        ...

    @abstractmethod
    async def traverse(self, start_node_id: str,
                       relation_filter: Optional[List[str]] = None,
                       max_depth: int = 3) -> List[KnowledgeGraphNode]:
        """
        Breadth-first graph traversal from a start node.

        Args:
            start_node_id:   Starting node identifier.
            relation_filter: Only follow edges with these relation labels.
            max_depth:       Maximum traversal depth.
        Returns:
            List of discovered nodes within max_depth.
        """
        ...

    @abstractmethod
    async def find_path(self, from_node_id: str,
                        to_node_id: str) -> Optional[List[KnowledgeGraphNode]]:
        """
        Find the shortest path between two nodes.

        Args:
            from_node_id: Source node ID.
            to_node_id:   Target node ID.
        Returns:
            Ordered list of nodes on the path, or None if unreachable.
        """
        ...

    @abstractmethod
    async def semantic_search(self, query_embedding: List[float],
                              top_k: int = 10) -> List[Tuple[KnowledgeGraphNode, float]]:
        """
        Find nodes most similar to a query embedding vector.

        Args:
            query_embedding: Dense vector representation of the query.
            top_k:           Number of nearest neighbors to return.
        Returns:
            List of (node, similarity_score) tuples.
        """
        ...


# ============================================================================
# Module Metadata
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    "KnowledgeType", "KnowledgeConfidence", "RetrievalStrategy",
    "Fact", "QueryContext", "KnowledgeResult", "KnowledgeStats",
    "KnowledgeUpdateResult", "KnowledgeGraphNode", "KnowledgeGraphEdge",
    "KnowledgeBaseInterface", "PersonalKnowledgeInterface",
    "RealTimeInformationInterface", "KnowledgeGraphInterface",
]
