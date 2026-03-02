"""Knowledge Base sub-package: structured facts, graphs, and semantic search."""
from __future__ import annotations

from .fact_database import Fact, FactDatabase, get_fact_database
from .knowledge_graphs import KnowledgeGraph, GraphNode, GraphEdge, get_knowledge_graph
from .semantic_search import SemanticSearchEngine, SearchResult, get_search_engine
from .concept_mapping import ConceptMap, ConceptNode, get_concept_map
from .knowledge_verification import KnowledgeVerifier, VerificationResult, get_verifier

__all__ = [
    "Fact", "FactDatabase", "get_fact_database",
    "KnowledgeGraph", "GraphNode", "GraphEdge", "get_knowledge_graph",
    "SemanticSearchEngine", "SearchResult", "get_search_engine",
    "ConceptMap", "ConceptNode", "get_concept_map",
    "KnowledgeVerifier", "VerificationResult", "get_verifier",
]
