from abc import ABC, abstractmethod

from src.core.graphflow import GraphFlow
from src.optimizer.graph_transform import prune_graphflow
from src.utils.logs import logger


class Pruner(ABC):
    include_all_pruned = True

    def __init__(self, original_graph: GraphFlow):
        self.original_graph = original_graph

    @abstractmethod
    def identify_pruning_candidates(self) -> dict[str, float]:
        """Return candidate node ids and their heuristic scores."""
        raise NotImplementedError

    def get_pruned(self) -> tuple[dict[str, GraphFlow], GraphFlow | None]:
        """Return one-candidate pruned graphs and an optional all-candidate graph."""
        candidate_ids = list(self.identify_pruning_candidates())

        pruned_graphs: dict[str, GraphFlow] = {}
        for node_id in candidate_ids:
            try:
                pruned_graphs[node_id] = prune_graphflow(self.original_graph, [node_id])
            except ValueError as exc:
                logger.warning(f"[PRUNER] Skip invalid pruned graph for {node_id}: {exc}")

        all_pruned_graph = None
        if self.include_all_pruned:
            try:
                all_pruned_graph = prune_graphflow(self.original_graph, candidate_ids)
            except ValueError as exc:
                logger.debug(f"[PRUNER] Skip invalid all-pruned graph: {exc}")

        return pruned_graphs, all_pruned_graph
