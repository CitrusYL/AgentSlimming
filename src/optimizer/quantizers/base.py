from abc import ABC, abstractmethod

from src.core.graphflow import GraphFlow


class Quantizer(ABC):
    include_all_quantized = True

    def __init__(self, original_graph: GraphFlow):
        self.original_graph = original_graph

    @abstractmethod
    def identify_quantization_candidates(self) -> dict[str, float]:
        """Return candidate node ids and their heuristic scores."""
        raise NotImplementedError

    def get_quantized(self) -> tuple[dict[str, GraphFlow], GraphFlow | None]:
        candidate_ids = list(self.identify_quantization_candidates())
        low_model_name = getattr(self, "low_model_name", None)
        if not low_model_name:
            raise ValueError("Quantizer.low_model_name must be set explicitly.")

        one_quantized_graphs: dict[str, GraphFlow] = {}
        for node_id in candidate_ids:
            new_nodes = self._quantized_nodes({node_id}, low_model_name)
            one_quantized_graphs[node_id] = GraphFlow(
                nodes=new_nodes,
                edges=self.original_graph.edges,
                entry_node_ids=self.original_graph.entry_node_ids,
                final_node_id=self.original_graph.final_node_id,
                description=f"Quantized: {node_id}",
            )

        all_quantized_graph = None
        if self.include_all_quantized:
            all_quantized_graph = GraphFlow(
                nodes=self._quantized_nodes(set(candidate_ids), low_model_name),
                edges=self.original_graph.edges,
                entry_node_ids=self.original_graph.entry_node_ids,
                final_node_id=self.original_graph.final_node_id,
                description="Quantized all candidates",
            )

        return one_quantized_graphs, all_quantized_graph

    def _quantized_nodes(self, node_ids: set[str], low_model_name: str) -> list:
        new_nodes = [node.model_copy(deep=True) for node in self.original_graph.nodes]
        for node in new_nodes:
            if node.node_id in node_ids:
                node.node_llm_config["model"] = low_model_name
            if hasattr(node, "node_llm_instance"):
                node.node_llm_instance = None
        return new_nodes
