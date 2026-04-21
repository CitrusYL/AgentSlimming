from src.optimizer.graph_metrics import betweenness_centrality, mutable_node_ids
from src.optimizer.pruners.base import Pruner
from src.utils.logs import logger


class BetweennessPruner(Pruner):
    def __init__(self, original_graph):
        super().__init__(original_graph)
        self.betweenness_scores: dict[str, float] = {}

    def compute_betweenness_centrality(self) -> dict[str, float]:
        return betweenness_centrality(self.original_graph)

    def identify_pruning_candidates(self) -> dict[str, float]:
        self.betweenness_scores = self.compute_betweenness_centrality()
        candidates = {
            node_id: self.betweenness_scores.get(node_id, 0.0)
            for node_id in mutable_node_ids(self.original_graph)
        }
        logger.debug(f"[PRUNER] Betweenness candidates: {candidates}")
        return candidates
