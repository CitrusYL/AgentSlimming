from collections import Counter, defaultdict, deque
from typing import Optional

from pydantic import BaseModel, ConfigDict, PrivateAttr

from src.core.edge import Edge
from src.core.nodes.base import Node


class GraphFlow(BaseModel):
    """Workflow DAG definition and graph query helpers."""
    model_config = ConfigDict(extra="forbid")

    nodes: list[Node]
    edges: list[Edge]
    entry_node_ids: list[str]
    final_node_id: str
    description: Optional[str] = None
    _node_map: dict[str, Node] = PrivateAttr(default_factory=dict)
    _incoming_edges: dict[str, list[Edge]] = PrivateAttr(default_factory=dict)
    _outgoing_edges: dict[str, list[Edge]] = PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        self._rebuild_indexes()
        self._validate()

    def _rebuild_indexes(self) -> None:
        self._node_map = {node.node_id: node for node in self.nodes}
        self._incoming_edges = {node.node_id: [] for node in self.nodes}
        self._outgoing_edges = {node.node_id: [] for node in self.nodes}
        for edge in self.edges:
            self._outgoing_edges.setdefault(edge.source, []).append(edge)
            self._incoming_edges.setdefault(edge.target, []).append(edge)

    def _validate(self) -> None:
        """Validate graph shape before execution."""
        node_ids = [node.node_id for node in self.nodes]
        if not node_ids:
            raise ValueError("GraphFlow must contain at least one node")

        duplicate_ids = sorted(
            node_id for node_id, count in Counter(node_ids).items() if count > 1
        )
        if duplicate_ids:
            raise ValueError(f"Duplicate node ids: {duplicate_ids}")

        node_id_set = set(node_ids)
        duplicate_entries = sorted(
            node_id for node_id, count in Counter(self.entry_node_ids).items() if count > 1
        )
        if duplicate_entries:
            raise ValueError(f"Duplicate entry node ids: {duplicate_entries}")

        missing_entries = [node_id for node_id in self.entry_node_ids if node_id not in node_id_set]
        if missing_entries:
            raise ValueError(f"Entry nodes not found: {missing_entries}")

        if self.final_node_id not in node_id_set:
            raise ValueError(f"Final node not found: {self.final_node_id}")

        missing_edges = [
            (edge.source, edge.target)
            for edge in self.edges
            if edge.source not in node_id_set or edge.target not in node_id_set
        ]
        if missing_edges:
            raise ValueError(f"Edges reference missing nodes: {missing_edges}")

        duplicate_edges = [
            edge_key
            for edge_key, count in Counter(
                (edge.source, edge.target, edge.key) for edge in self.edges
            ).items()
            if count > 1
        ]
        if duplicate_edges:
            raise ValueError(f"Duplicate edges: {duplicate_edges}")

        duplicate_input_names = self._duplicate_input_names()
        if duplicate_input_names:
            raise ValueError(f"Duplicate input names for target nodes: {duplicate_input_names}")

        order = self.get_topological_order()
        if len(order) != len(node_id_set):
            unresolved = sorted(node_id_set - set(order))
            raise ValueError(f"Graph contains a cycle or unreachable dependency: {unresolved}")

        if self.final_node_id not in order:
            raise ValueError(f"Final node is not executable: {self.final_node_id}")

        reachable = set()
        queue = deque(self.entry_node_ids)
        while queue:
            node_id = queue.popleft()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            queue.extend(self.successors(node_id))

        unreachable = sorted(node_id_set - reachable)
        if unreachable:
            raise ValueError(f"Nodes are not reachable from entry nodes: {unreachable}")

    def _duplicate_input_names(self) -> dict[str, list[str]]:
        names_by_target: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            names_by_target[edge.target].append(edge.input_name())

        duplicates = {}
        for target, names in names_by_target.items():
            duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
            if duplicate_names:
                duplicates[target] = duplicate_names
        return duplicates

    def get_topological_order(self) -> list[str]:
        """Return nodes in dependency order."""
        in_degree = defaultdict(int)
        for node_id in self._node_map:
            in_degree[node_id] = 0

        for edge in self.edges:
            in_degree[edge.target] += 1

        queue = deque()
        for entry_node_id in self.entry_node_ids:
            if in_degree[entry_node_id] == 0:
                queue.append(entry_node_id)

        for node_id in self._node_map:
            if in_degree[node_id] == 0 and node_id not in self.entry_node_ids:
                queue.append(node_id)

        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in self.successors(u):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        return order

    def get_node(self, node_id: str) -> Node:
        """Return node by id."""
        if node_id not in self._node_map:
            raise KeyError(f"Node {node_id} not found")
        return self._node_map[node_id]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        self.get_node(node_id)
        return list(self._incoming_edges.get(node_id, []))

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        self.get_node(node_id)
        return list(self._outgoing_edges.get(node_id, []))

    def predecessors(self, node_id: str) -> list[str]:
        return [edge.source for edge in self.incoming_edges(node_id)]

    def successors(self, node_id: str) -> list[str]:
        return [edge.target for edge in self.outgoing_edges(node_id)]
