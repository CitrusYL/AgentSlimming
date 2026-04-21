from src.optimizer.graph_metrics import mutable_node_ids
from src.optimizer.pruners.base import Pruner


class TraversePruner(Pruner):
    include_all_pruned = False

    def identify_pruning_candidates(self) -> dict[str, float]:
        return {node_id: 1.0 for node_id in mutable_node_ids(self.original_graph)}
