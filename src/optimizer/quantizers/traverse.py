from src.optimizer.graph_metrics import mutable_node_ids
from src.optimizer.quantizers.base import Quantizer


class TraverseQuantizer(Quantizer):
    include_all_quantized = False

    def identify_quantization_candidates(self) -> dict[str, float]:
        return {node_id: 1.0 for node_id in mutable_node_ids(self.original_graph)}
