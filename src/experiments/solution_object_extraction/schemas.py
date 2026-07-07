"""Typed records for canonical arithmetic graphs and extracted object vectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One role-bearing value in a canonical solution graph."""

    id: str
    kind: str
    role: str
    value: float


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """One directed arithmetic relation in a canonical solution graph."""

    id: str
    kind: str
    op: str
    src: tuple[str, ...]
    dst: str


@dataclass(frozen=True, slots=True)
class CanonicalGraph:
    """A vocabulary-independent arithmetic state."""

    canonical_graph_id: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    target: str
    graph_hash: str = ""

    def to_record(self) -> dict[str, Any]:
        """Return the stable JSON representation, including a content hash."""
        record = {
            "canonical_graph_id": self.canonical_graph_id,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [
                {**asdict(edge), "src": list(edge.src)} for edge in self.edges
            ],
            "target": self.target,
        }
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        record["graph_hash"] = "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
        return record

    @property
    def operation(self) -> str:
        """Return the graph's final operation or ``NONE`` for a bound-only state."""
        return self.edges[-1].op if self.edges else "NONE"

    @property
    def values(self) -> dict[str, float]:
        """Return role-to-value bindings."""
        return {node.role: node.value for node in self.nodes}


@dataclass(slots=True)
class ObjectRecord:
    """Metadata aligned to one pooled hidden-state vector."""

    record_id: str
    model: str
    layer: int
    trace_id: str
    token_start: int
    token_end: int
    vector_index: int
    projection_id: str | None
    canonical_graph_id: str
    gold_graph_id: str
    edit_id: str
    edit_type: str
    split: str
    question_id: str
    surface: dict[str, Any]
    expected: dict[str, Any]
    observed: dict[str, Any]
    is_correct: bool
    hard_negative_type: str | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible record."""
        return asdict(self)


def make_graph(
    *,
    graph_id: str,
    operation: str,
    operand_a: float,
    operand_b: float,
    result: float | None,
) -> CanonicalGraph:
    """Build a bound-only or operated canonical graph."""
    nodes = [
        GraphNode("q1", "quantity", "operand_a", float(operand_a)),
        GraphNode("q2", "quantity", "operand_b", float(operand_b)),
    ]
    edges: list[GraphEdge] = []
    target = "q3"
    if result is not None:
        nodes.append(GraphNode("q3", "quantity", "result", float(result)))
        edges.append(
            GraphEdge("e1", "relation", operation, ("q1", "q2"), "q3")
        )
    else:
        target = "q2"
    return CanonicalGraph(graph_id, tuple(nodes), tuple(edges), target)
