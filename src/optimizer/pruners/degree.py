from src.optimizer.graph_metrics import is_low_degree_node, node_degrees
from src.optimizer.pruners.base import Pruner
from src.utils.logs import logger


class DegreePruner(Pruner):
    def identify_pruning_candidates(self) -> dict[str, float]:
        candidates: dict[str, float] = {}
        degrees = node_degrees(self.original_graph)

        for node in self.original_graph.nodes:
            node_id = node.node_id
            in_degree, out_degree = degrees[node_id]
            if node_id in self.original_graph.entry_node_ids or node_id == self.original_graph.final_node_id:
                continue
            if not is_low_degree_node(in_degree, out_degree):
                continue

            candidates[node_id] = 1.0
            logger.debug(
                f"[PRUNER] Candidate {node_id}: low degree "
                f"(in={in_degree}, out={out_degree})"
            )

        logger.debug(f"[PRUNER] Degree candidates: {candidates}")
        return candidates
