from collections import deque
import re

from src.core.edge import Edge
from src.core.graphflow import GraphFlow


def build_pruned_graphflow(
    graphflow: GraphFlow,
    prune_candidates: list[str],
    original_code: str,
) -> tuple[GraphFlow, str]:
    pruned_graphflow = prune_graphflow(graphflow, prune_candidates)
    pruned_code = render_pruned_graph_code(original_code, prune_candidates, pruned_graphflow)
    return pruned_graphflow, pruned_code


def prune_graphflow(graphflow: GraphFlow, prune_candidates: list[str]) -> GraphFlow:
    prune_set = set(prune_candidates)
    remaining_nodes = [node for node in graphflow.nodes if node.node_id not in prune_set]
    remaining_node_ids = {node.node_id for node in remaining_nodes}

    if not remaining_nodes:
        raise ValueError("Pruning would remove every node")
    if graphflow.final_node_id in prune_set:
        raise ValueError(f"Cannot prune final node: {graphflow.final_node_id}")

    new_edges = [
        edge
        for edge in graphflow.edges
        if edge.source not in prune_set and edge.target not in prune_set
    ]
    edge_keys = {(edge.source, edge.target) for edge in new_edges}

    for node_id in prune_set:
        for incoming_edge in graphflow.incoming_edges(node_id):
            source = incoming_edge.source
            if source not in remaining_node_ids:
                continue

            for target in _bypass_targets(graphflow, node_id, prune_set, remaining_node_ids):
                if source == target or (source, target) in edge_keys:
                    continue
                new_edges.append(Edge(source=source, target=target))
                edge_keys.add((source, target))

    return GraphFlow(
        nodes=remaining_nodes,
        edges=new_edges,
        entry_node_ids=[node_id for node_id in graphflow.entry_node_ids if node_id not in prune_set],
        final_node_id=graphflow.final_node_id,
        description=f"Pruned graph with reconnection ({len(remaining_nodes)} nodes)",
    )


def apply_quantization_to_code(
    graph_code: str,
    quantized_nodes: list[str],
    low_model_name: str,
) -> str:
    for node_id in quantized_nodes:
        call_span = _find_node_assignment_call(graph_code, node_id)
        if call_span is None:
            continue

        start_pos, end_pos = call_span
        node_definition = graph_code[start_pos:end_pos]
        low_model_config = _low_model_config_code(low_model_name)

        if re.search(r"node_llm_config\s*=", node_definition):
            modified_definition = re.sub(
                r"(node_llm_config\s*=\s*)(?:\{[^}]*\}|[^\n,)]+)",
                lambda match: f"{match.group(1)}{low_model_config}",
                node_definition,
                count=1,
            )
        else:
            last_param_match = re.search(
                r"(.*)((?:node_description|description)\s*=\s*[\"'][^\"']*[\"'])\s*\)",
                node_definition,
                re.DOTALL,
            )
            if not last_param_match:
                continue
            prefix = last_param_match.group(1)
            last_param = last_param_match.group(2)
            modified_definition = (
                f"{prefix}{last_param},\n"
                f"            node_llm_config={low_model_config}\n"
                "        )"
            )

        graph_code = graph_code[:start_pos] + modified_definition + graph_code[end_pos:]
    return graph_code


def render_pruned_graph_code(
    original_code: str,
    pruned_nodes: list[str],
    pruned_graph: GraphFlow,
) -> str:
    lines = original_code.split("\n")
    new_lines: list[str] = []
    remaining_node_ids = [node.node_id for node in pruned_graph.nodes]
    new_edges_code = [_edge_to_code(edge) for edge in pruned_graph.edges]

    i = 0
    while i < len(lines):
        line = lines[i]
        if _starts_node_assignment(line, pruned_nodes):
            i = _skip_call_lines(lines, i)
            continue

        if not _starts_graphflow_definition(line):
            new_lines.append(line)
            i += 1
            continue

        new_lines.append(line)
        i += 1
        i = _replace_argument_block(lines, new_lines, i, "nodes=", _nodes_to_code(remaining_node_ids))
        i = _replace_argument_block(lines, new_lines, i, "edges=", ["edges=[", *new_edges_code, "],"])
        entry_ids = ", ".join(repr(node_id) for node_id in pruned_graph.entry_node_ids)
        i = _replace_argument_block(lines, new_lines, i, "entry_node_ids=", [f"entry_node_ids=[{entry_ids}],"])
        i = _replace_argument_block(lines, new_lines, i, "final_node_id=", [f"final_node_id={pruned_graph.final_node_id!r},"])
        new_lines.extend(lines[i:])
        break

    return "\n".join(new_lines)


def _bypass_targets(
    graphflow: GraphFlow,
    node_id: str,
    prune_set: set[str],
    remaining_node_ids: set[str],
) -> set[str]:
    targets: set[str] = set()
    visited: set[str] = set()
    queue = deque(edge.target for edge in graphflow.outgoing_edges(node_id))

    while queue:
        target = queue.popleft()
        if target in visited:
            continue
        visited.add(target)

        if target in remaining_node_ids:
            targets.add(target)
            continue

        if target in prune_set:
            queue.extend(edge.target for edge in graphflow.outgoing_edges(target))

    return targets


def _starts_graphflow_definition(line: str) -> bool:
    return (
        "self.workflow_graph = GraphFlow(" in line
        or re.search(r"\breturn\s+GraphFlow\(", line) is not None
    )


def _starts_node_assignment(line: str, node_ids: list[str]) -> bool:
    return any(
        re.match(rf"^\s*{re.escape(node_id)}\s*=\s*\w+Node\(", line)
        for node_id in node_ids
    )


def _skip_call_lines(lines: list[str], start_index: int) -> int:
    paren_count = lines[start_index].count("(") - lines[start_index].count(")")
    i = start_index + 1
    while i < len(lines) and paren_count != 0:
        paren_count += lines[i].count("(") - lines[i].count(")")
        i += 1
    return i


def _replace_argument_block(
    lines: list[str],
    new_lines: list[str],
    start_index: int,
    marker: str,
    replacement_lines: list[str],
) -> int:
    i = start_index
    while i < len(lines) and marker not in lines[i]:
        new_lines.append(lines[i])
        i += 1

    if i >= len(lines):
        return i

    indent = len(lines[i]) - len(lines[i].lstrip())
    new_lines.extend(" " * indent + line for line in replacement_lines)
    bracket_count = lines[i].count("[") - lines[i].count("]")
    paren_count = lines[i].count("(") - lines[i].count(")")
    i += 1
    while i < len(lines) and (bracket_count > 0 or paren_count > 0):
        bracket_count += lines[i].count("[") - lines[i].count("]")
        paren_count += lines[i].count("(") - lines[i].count(")")
        i += 1
    return i


def _nodes_to_code(node_ids: list[str]) -> list[str]:
    return [f"nodes=[{', '.join(node_ids)}],"]


def _edge_to_code(edge: Edge) -> str:
    args = [
        f"source={edge.source!r}",
        f"target={edge.target!r}",
    ]
    if edge.key is not None:
        args.append(f"key={edge.key!r}")
    if edge.description is not None:
        args.append(f"description={edge.description!r}")
    if edge.as_candidate:
        args.append("as_candidate=True")
    return f"    Edge({', '.join(args)}),"


def _find_node_assignment_call(graph_code: str, node_id: str) -> tuple[int, int] | None:
    match = re.search(rf"(?m)^\s*{re.escape(node_id)}\s*=\s*\w+\(", graph_code)
    if not match:
        return None

    open_paren_pos = graph_code.find("(", match.start())
    end_pos = _matching_paren_end(graph_code, open_paren_pos)
    if end_pos is None:
        return None
    return match.start(), end_pos


def _matching_paren_end(source: str, open_paren_pos: int) -> int | None:
    if open_paren_pos < 0:
        return None

    paren_count = 1
    i = open_paren_pos + 1
    while i < len(source) and paren_count > 0:
        if source[i] == "(":
            paren_count += 1
        elif source[i] == ")":
            paren_count -= 1
        i += 1

    return i if paren_count == 0 else None


def _low_model_config_code(low_model_name: str) -> str:
    return (
        "{\n"
        f'                "model": "{low_model_name}",\n'
        '                "temperature": 0.0\n'
        "            }"
    )
