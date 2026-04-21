from src.optimizer.graph_metrics import betweenness_centrality, mutable_node_ids
from src.optimizer.quantizers.base import Quantizer
from src.utils.logs import logger


class BetweennessQuantizer(Quantizer):
    def __init__(self, original_graph):
        super().__init__(original_graph)
        self.betweenness_scores: dict[str, float] = {}

    def compute_betweenness_centrality(self) -> dict[str, float]:
        return betweenness_centrality(self.original_graph)

    def identify_quantization_candidates(self) -> dict[str, float]:
        self.betweenness_scores = self.compute_betweenness_centrality()
        candidates = {
            node_id: self.betweenness_scores.get(node_id, 0.0)
            for node_id in mutable_node_ids(self.original_graph)
        }
        logger.debug(f"[QUANTIZER] Betweenness candidates: {candidates}")
        return candidates
