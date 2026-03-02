"""Graph-based knowledge representation with BFS/DFS traversal."""
from __future__ import annotations
import threading, time, logging, math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Deque, Tuple, Iterator
from collections import deque, defaultdict
from src.core.utils.logging_utils import get_logger
logger = get_logger(__name__)


class EdgeType(Enum):
    IS_A = "is_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    PRECEDES = "precedes"
    SYNONYMOUS = "synonymous"
    OPPOSITE_OF = "opposite_of"
    INSTANCE_OF = "instance_of"


@dataclass
class GraphNode:
    node_id: str
    label: str
    node_type: str = "concept"
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphNode):
            return False
        return self.node_id == other.node_id


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.source_id, self.target_id, self.edge_type.value)


class KnowledgeGraph:
    """Directed weighted knowledge graph with BFS, DFS, and path-finding."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[Tuple[str, str, str], GraphEdge] = {}
        self._adj: Dict[str, List[GraphEdge]] = defaultdict(list)   # outgoing
        self._radj: Dict[str, List[GraphEdge]] = defaultdict(list)  # incoming

    # ------------------------------------------------------------------
    def add_node(self, node: GraphNode) -> None:
        with self._lock:
            self._nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        with self._lock:
            if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
                raise ValueError(f"Both nodes must exist before adding edge {edge.key}")
            self._edges[edge.key] = edge
            self._adj[edge.source_id].append(edge)
            self._radj[edge.target_id].append(edge)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        with self._lock:
            return self._nodes.get(node_id)

    def neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[GraphNode]:
        with self._lock:
            edges = self._adj.get(node_id, [])
            if edge_type:
                edges = [e for e in edges if e.edge_type == edge_type]
            return [self._nodes[e.target_id] for e in edges if e.target_id in self._nodes]

    # ------------------------------------------------------------------
    def bfs(self, start_id: str, max_depth: int = 5) -> List[str]:
        """Breadth-first traversal; returns node IDs in BFS order."""
        with self._lock:
            if start_id not in self._nodes:
                return []
            visited: Set[str] = set()
            queue: Deque[Tuple[str, int]] = deque([(start_id, 0)])
            result: List[str] = []
            while queue:
                nid, depth = queue.popleft()
                if nid in visited or depth > max_depth:
                    continue
                visited.add(nid)
                result.append(nid)
                for edge in self._adj.get(nid, []):
                    if edge.target_id not in visited:
                        queue.append((edge.target_id, depth + 1))
            return result

    def dfs(self, start_id: str, max_depth: int = 5) -> List[str]:
        """Depth-first traversal; returns node IDs in DFS order."""
        visited: Set[str] = set()
        result: List[str] = []

        def _dfs(nid: str, depth: int) -> None:
            if nid in visited or depth > max_depth:
                return
            visited.add(nid)
            result.append(nid)
            with self._lock:
                edges = list(self._adj.get(nid, []))
            for edge in edges:
                _dfs(edge.target_id, depth + 1)

        _dfs(start_id, 0)
        return result

    def shortest_path(self, start_id: str, end_id: str) -> Optional[List[str]]:
        """Dijkstra-based shortest path by inverse edge weight."""
        import heapq
        with self._lock:
            if start_id not in self._nodes or end_id not in self._nodes:
                return None
            dist: Dict[str, float] = {start_id: 0.0}
            prev: Dict[str, Optional[str]] = {start_id: None}
            heap: List[Tuple[float, str]] = [(0.0, start_id)]
            visited: Set[str] = set()
            while heap:
                d, nid = heapq.heappop(heap)
                if nid in visited:
                    continue
                visited.add(nid)
                if nid == end_id:
                    break
                for edge in self._adj.get(nid, []):
                    cost = d + (1.0 / max(edge.weight, 1e-9))
                    if cost < dist.get(edge.target_id, math.inf):
                        dist[edge.target_id] = cost
                        prev[edge.target_id] = nid
                        heapq.heappush(heap, (cost, edge.target_id))
            if end_id not in prev:
                return None
            path: List[str] = []
            cur: Optional[str] = end_id
            while cur is not None:
                path.append(cur)
                cur = prev.get(cur)
            path.reverse()
            return path

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"nodes": len(self._nodes), "edges": len(self._edges)}


_KG: Optional[KnowledgeGraph] = None
_KG_LOCK = threading.Lock()


def get_knowledge_graph() -> KnowledgeGraph:
    global _KG
    with _KG_LOCK:
        if _KG is None:
            _KG = KnowledgeGraph()
        return _KG


def run_tests() -> bool:
    kg = KnowledgeGraph()
    nodes = [
        GraphNode("A", "Animal"), GraphNode("B", "Bird"), GraphNode("C", "Canary"),
        GraphNode("D", "Dog"),
    ]
    for n in nodes:
        kg.add_node(n)
    kg.add_edge(GraphEdge("B", "A", EdgeType.IS_A))
    kg.add_edge(GraphEdge("C", "B", EdgeType.IS_A))
    kg.add_edge(GraphEdge("D", "A", EdgeType.IS_A))
    bfs = kg.bfs("A")
    assert "A" in bfs
    path = kg.shortest_path("C", "A")
    assert path is not None and path[0] == "C" and path[-1] == "A"
    stats = kg.stats()
    assert stats["nodes"] == 4 and stats["edges"] == 3
    logger.info("KnowledgeGraph tests PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_tests()
