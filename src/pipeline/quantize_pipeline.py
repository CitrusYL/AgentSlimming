import asyncio

from src.evaluation.evaluator import Evaluator
from src.optimizer.data_utils import DataUtils
from src.optimizer.evaluation_utils import EvaluationUtils
from src.optimizer.graph_metrics import mutable_node_ids
from src.optimizer.manifest_utils import WorkflowRoundRef, write_round_manifest
from src.optimizer.graph_transform import apply_quantization_to_code
from src.optimizer.graph_utils import GraphUtils
from src.optimizer.workspace import WorkflowDirs
from src.utils.logs import logger

from src.pipeline.prune_quantize_utils import (
    build_quantize_ranking,
    evaluate_workflow_metrics,
    select_best_graph_round,
    set_workflow_graph,
)


class QuantizePipeline:
    max_candidates_per_iter = 2

    def __init__(
        self,
        workspace: str = "workspace",
        dataset: str = "MATH",
        exec_llm_config: dict | None = None,
        validation_rounds: int = 1,
        traverse_sample_size: int = 50,
        quantize_eval_samples: int = None,
        quantize_rate: float = 0.4,
        quantize_threshold: float = 0.95,
        quantize_low_model_name: str | None = None,
        w1: float = 1,
        w2: float = 1,
        w3: float = 2,
        w4: float = 2,
        initial_round: int = None,
    ):
        self.workspace = workspace.replace("\\", "/").rstrip("/")
        self.dataset = dataset
        if exec_llm_config is None:
            raise ValueError("exec_llm_config is required.")
        if not quantize_low_model_name:
            raise ValueError("quantize_low_model_name is required.")
        self.execute_llm_config = exec_llm_config
        self.validation_rounds = validation_rounds
        self.traverse_sample_size = traverse_sample_size
        self.quantize_eval_samples = quantize_eval_samples
        self.quantize_rate = quantize_rate
        self.quantize_threshold = quantize_threshold
        self.quantize_low_model_name = quantize_low_model_name
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.initial_round = initial_round
        self.root_path = f"{self.workspace}/{self.dataset}"
        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.graph = None
        self.round = 1

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._quantize())
        loop.close()

    async def _quantize(self) -> None:
        graph_root = str(self.graph_utils.store.workflow_path(WorkflowDirs.GRAPH))
        pruned_root = str(self.graph_utils.store.workflow_path(WorkflowDirs.PRUNED))
        quant_root = str(self.graph_utils.store.workflow_path(WorkflowDirs.QUANTIZED))
        source = self._resolve_source_workflow(graph_root, pruned_root)
        if source is None:
            logger.info("[QUANTIZE] No valid source round found")
            return
        target_round = self.graph_utils.store.next_round(quant_root)
        source_root = str(self.graph_utils.store.workflow_path(source.workflow_name))

        prompt_src, graph_src = self.graph_utils.read_graph_files(
            source.round_number,
            source_root,
        )
        Graph = self.graph_utils.load_graph(
            source.round_number,
            source_root,
        )
        self.graph = Graph(
            name=f"Quantized_Round{target_round}",
            dataset=self.dataset,
            llm_config=self.execute_llm_config,
        )
        self.round = target_round
        graphflow = self.graph.workflow_graph

        max_quantized = self._max_quantized_count(graphflow)
        if max_quantized == 0:
            logger.info("[QUANTIZE] No quantizable candidates found")
            return

        eval_dir = str(self.graph_utils.store.evaluation_path("quantize"))
        self.graph_utils.store.evaluation_path("quantize").mkdir(parents=True, exist_ok=True)
        evaluator = Evaluator(eval_path=eval_dir)

        baseline_score, baseline_avg_cost, baseline_total_cost = await evaluate_workflow_metrics(
            self,
            evaluator,
            eval_dir,
            self.validation_rounds,
            self.quantize_eval_samples,
        )
        quantized_score = baseline_score
        quantized_avg_cost = baseline_avg_cost
        quantized_total_cost = baseline_total_cost
        quantized_nodes: list[str] = []
        iteration_details: list[dict] = []

        for iter_count in range(1, max_quantized + 1):
            ranking = await build_quantize_ranking(
                self,
                graphflow,
                evaluator,
                eval_dir,
                excluded=set(quantized_nodes),
            )
            if not ranking.candidates:
                break

            accepted = await self._try_quantize_candidates(
                ranking.candidates[: self.max_candidates_per_iter],
                evaluator,
                eval_dir,
                baseline_score,
                iter_count,
            )
            iteration_details.append(
                {
                    **ranking.detail(
                        candidate=accepted["candidate"],
                        tried_candidates=accepted["tried_candidates"],
                    ),
                    "iter": iter_count,
                    "accepted": accepted["accepted"],
                    "threshold_value": accepted["threshold_value"],
                    "candidate_eval_score": accepted["score"],
                }
            )
            if not accepted["accepted"]:
                logger.info("[QUANTIZE] No acceptable candidates found. Stopping.")
                break

            candidate = accepted["candidate"]
            quantized_nodes.append(candidate)
            quantized_score = accepted["score"]
            quantized_avg_cost = accepted["avg_cost"]
            quantized_total_cost = accepted["total_cost"]
            set_workflow_graph(self.graph, graphflow)

        quantized_dir = self.graph_utils.create_round_directory(quant_root, self.round)
        graph_code = apply_quantization_to_code(
            graph_src,
            quantized_nodes,
            self.quantize_low_model_name,
        )
        self.graph_utils.write_workflow_files(quantized_dir, graph_code, prompt_src)
        self.data_utils.record_round_result(
            quant_root,
            self.round,
            quantized_score,
            quantized_avg_cost,
            quantized_total_cost,
        )
        metrics = self.data_utils.summarize_round_results(quant_root, self.round) or {
            "score": quantized_score,
            "avg_cost": quantized_avg_cost,
            "total_cost": quantized_total_cost,
            "evaluations": 1,
        }
        self.graph_utils.write_json(
            quantized_dir,
            "quantize_detail_info.json",
            {
                "source": source.to_manifest_ref(),
                "baseline_score": baseline_score,
                "final_score": quantized_score,
                "quantized_nodes": quantized_nodes,
                "total_quantized": len(quantized_nodes),
                "quantize_rate": self.quantize_rate,
                "quantize_threshold": self.quantize_threshold,
                "low_model_name": self.quantize_low_model_name,
                "iterations": iteration_details,
            },
        )
        write_round_manifest(
            self.graph_utils.store,
            WorkflowDirs.QUANTIZED,
            self.round,
            parent=source,
            source_selector=source.selector,
            metrics=metrics,
            summary={
                "baseline_score": baseline_score,
                "quantized_nodes": quantized_nodes,
                "quantized_count": len(quantized_nodes),
                "quantize_rate": self.quantize_rate,
                "quantize_threshold": self.quantize_threshold,
                "low_model_name": self.quantize_low_model_name,
            },
            artifacts=[
                "graph.py",
                "prompt.py",
                "quantize_detail_info.json",
                "manifest.json",
            ],
        )

    def _resolve_source_workflow(self, graph_root: str, pruned_root: str) -> WorkflowRoundRef | None:
        if self.initial_round is not None:
            return WorkflowRoundRef(
                stage="prune",
                workflow_name=WorkflowDirs.PRUNED,
                round_number=self.initial_round,
                selector="explicit_round",
            )

        pruned_round = self.graph_utils.latest_round(pruned_root)
        if pruned_round is not None and self.graph_utils.has_graph_files(pruned_root, pruned_round):
            return WorkflowRoundRef(
                stage="prune",
                workflow_name=WorkflowDirs.PRUNED,
                round_number=pruned_round,
                selector="latest_round",
            )

        round_number = select_best_graph_round(self.graph_utils, self.data_utils, graph_root, sample=1)
        if round_number is None:
            return None
        return WorkflowRoundRef(
            stage="mcts",
            workflow_name=WorkflowDirs.GRAPH,
            round_number=round_number,
            selector="mcts_best_score_fallback",
        )

    def _max_quantized_count(self, graphflow) -> int:
        candidates_count = len(mutable_node_ids(graphflow))
        if candidates_count == 0 or self.quantize_rate <= 0:
            return 0
        return min(candidates_count, max(1, int(candidates_count * self.quantize_rate)))

    async def _try_quantize_candidates(
        self,
        candidates: list[str],
        evaluator: Evaluator,
        eval_dir: str,
        baseline_score: float,
        iter_count: int,
    ):
        tried_candidates = []
        threshold = self.quantize_threshold * baseline_score

        for candidate in candidates:
            tried_candidates.append(candidate)
            original_config = self._node_config(candidate).copy()
            self._set_node_model(candidate, self.quantize_low_model_name)
            candidate_score, candidate_avg_cost, candidate_total_cost = await evaluate_workflow_metrics(
                self,
                evaluator,
                eval_dir,
                self.validation_rounds,
                self.quantize_eval_samples,
            )
            logger.info(
                f"[QUANTIZE] Iter {iter_count}: candidate {candidate} "
                f"score={candidate_score:.5f} threshold={threshold:.5f}"
            )

            if candidate_score >= threshold:
                return {
                    "accepted": True,
                    "candidate": candidate,
                    "score": candidate_score,
                    "avg_cost": candidate_avg_cost,
                    "total_cost": candidate_total_cost,
                    "tried_candidates": tried_candidates,
                    "threshold_value": threshold,
                }

            self._set_node_config(candidate, original_config)

        return {
            "accepted": False,
            "candidate": None,
            "score": None,
            "avg_cost": None,
            "total_cost": None,
            "tried_candidates": tried_candidates,
            "threshold_value": threshold,
        }

    def _node_config(self, node_id: str) -> dict:
        for node in self.graph.workflow_graph.nodes:
            if node.node_id == node_id:
                return node.node_llm_config
        raise ValueError(f"Node not found: {node_id}")

    def _set_node_model(self, node_id: str, model_name: str) -> None:
        config = self._node_config(node_id).copy()
        config["model"] = model_name
        self._set_node_config(node_id, config)

    def _set_node_config(self, node_id: str, config: dict) -> None:
        for node in self.graph.workflow_graph.nodes:
            if node.node_id == node_id:
                node.node_llm_config = config.copy()
                if hasattr(node, "node_llm_instance"):
                    node.node_llm_instance = None
                return
        raise ValueError(f"Node not found: {node_id}")
