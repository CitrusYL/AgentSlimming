from collections import defaultdict, deque

from src.core.graphflow import GraphFlow


def mutable_node_ids(graph: GraphFlow) -> set[str]:
    """Return nodes that may be pruned or quantized."""
    return {
        node.node_id
        for node in graph.nodes
        if node.node_id not in graph.entry_node_ids
        and node.node_id != graph.final_node_id
    }


def node_degrees(graph: GraphFlow) -> dict[str, tuple[int, int]]:
    return {
        node.node_id: (
            len(graph.incoming_edges(node.node_id)),
            len(graph.outgoing_edges(node.node_id)),
        )
        for node in graph.nodes
    }


def is_low_degree_node(in_degree: int, out_degree: int) -> bool:
    return in_degree + out_degree <= 2


def betweenness_centrality(graph: GraphFlow) -> dict[str, float]:
    """Compute directed betweenness centrality for non-entry, non-final nodes."""
    mutable_nodes = mutable_node_ids(graph)
    all_nodes = [node.node_id for node in graph.nodes]
    betweenness = defaultdict(float)
    for node_id in all_nodes:
        if node_id in mutable_nodes:
            betweenness[node_id] = 0.0

    for source in all_nodes:
        stack: list[str] = []
        paths = defaultdict(list)
        sigma = defaultdict(int)
        dist = defaultdict(lambda: -1)
        delta = defaultdict(float)

        sigma[source] = 1
        dist[source] = 0
        queue = deque([source])

        while queue:
            node_id = queue.popleft()
            stack.append(node_id)

            for child in graph.successors(node_id):
                if dist[child] < 0:
                    queue.append(child)
                    dist[child] = dist[node_id] + 1

                if dist[child] == dist[node_id] + 1:
                    sigma[child] += sigma[node_id]
                    paths[child].append(node_id)

        while stack:
            child = stack.pop()
            for parent in paths[child]:
                delta[parent] += (sigma[parent] / sigma[child]) * (1 + delta[child])

            if child != source and child in mutable_nodes:
                betweenness[child] += delta[child]

    node_count = len(all_nodes)
    if node_count > 2:
        normalization = (node_count - 1) * (node_count - 2)
        for node_id in betweenness:
            betweenness[node_id] /= normalization

    return dict(betweenness)
