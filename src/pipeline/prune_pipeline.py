import asyncio

from src.evaluation.evaluator import Evaluator
from src.optimizer.data_utils import DataUtils
from src.optimizer.evaluation_utils import EvaluationUtils
from src.optimizer.graph_transform import build_pruned_graphflow
from src.optimizer.graph_utils import GraphUtils
from src.optimizer.manifest_utils import WorkflowRoundRef, write_round_manifest
from src.optimizer.workspace import WorkflowDirs
from src.utils.logs import logger

from src.pipeline.prune_quantize_utils import (
    build_prune_ranking,
    evaluate_workflow_metrics,
    select_best_graph_round,
    set_workflow_graph,
)


class PrunePipeline:
    max_iters = 3
    max_candidates_per_iter = 2

    def __init__(
        self,
        workspace: str = "workspace",
        dataset: str = "MATH",
        exec_llm_config: dict | None = None,
        sample: int = 3,
        validation_rounds: int = 1,
        prune_threshold: float = 0.95,
        traverse_sample_size: int = 50,
        prune_eval_samples: int = None,
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
        self.execute_llm_config = exec_llm_config
        self.sample = sample
        self.validation_rounds = validation_rounds
        self.prune_threshold = prune_threshold
        self.traverse_sample_size = traverse_sample_size
        self.prune_eval_samples = prune_eval_samples
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
        loop.run_until_complete(self._prune())
        loop.close()

    async def _prune(self) -> None:
        graph_root = str(self.graph_utils.store.workflow_path(WorkflowDirs.GRAPH))
        pruned_root = str(self.graph_utils.store.workflow_path(WorkflowDirs.PRUNED))
        source = self._resolve_source_workflow(graph_root)
        if source is None:
            logger.info("[PRUNE] No valid source round found")
            return
        target_round = self.graph_utils.store.next_round(pruned_root)
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
            name=f"Pruned_Round{target_round}",
            dataset=self.dataset,
            llm_config=self.execute_llm_config,
        )
        self.round = target_round

        source_graphflow = self.graph.workflow_graph
        current_graphflow = source_graphflow
        current_code = graph_src
        pruned_nodes: list[str] = []
        iteration_details: list[dict] = []

        eval_dir = str(self.graph_utils.store.evaluation_path("prune"))
        self.graph_utils.store.evaluation_path("prune").mkdir(parents=True, exist_ok=True)
        evaluator = Evaluator(eval_path=eval_dir)

        baseline_score, baseline_avg_cost, baseline_total_cost = await evaluate_workflow_metrics(
            self,
            evaluator,
            eval_dir,
            self.validation_rounds,
            self.prune_eval_samples,
        )
        current_score = baseline_score
        current_avg_cost = baseline_avg_cost
        current_total_cost = baseline_total_cost

        for iter_count in range(1, self.max_iters + 1):
            ranking = await build_prune_ranking(
                self,
                current_graphflow,
                evaluator,
                eval_dir,
                excluded=set(pruned_nodes),
            )
            if not ranking.candidates:
                break

            accepted = await self._try_prune_candidates(
                ranking.candidates[: self.max_candidates_per_iter],
                pruned_nodes,
                source_graphflow,
                graph_src,
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
                set_workflow_graph(self.graph, current_graphflow)
                break

            candidate = accepted["candidate"]
            pruned_nodes.append(candidate)
            current_graphflow = accepted["graphflow"]
            current_code = accepted["graph_code"]
            current_score = accepted["score"]
            current_avg_cost = accepted["avg_cost"]
            current_total_cost = accepted["total_cost"]
            set_workflow_graph(self.graph, current_graphflow)

        final_round_dir = self.graph_utils.create_round_directory(pruned_root, self.round)
        self.graph_utils.write_workflow_files(final_round_dir, current_code, prompt_src)
        self.data_utils.record_round_result(
            pruned_root,
            self.round,
            current_score,
            current_avg_cost,
            current_total_cost,
        )
        metrics = self.data_utils.summarize_round_results(pruned_root, self.round) or {
            "score": current_score,
            "avg_cost": current_avg_cost,
            "total_cost": current_total_cost,
            "evaluations": 1,
        }
        self.graph_utils.write_json(
            final_round_dir,
            "prune_detail_info.json",
            {
                "source": source.to_manifest_ref(),
                "baseline_score": baseline_score,
                "final_score": current_score,
                "pruned_nodes": pruned_nodes,
                "pruned_number": len(pruned_nodes),
                "threshold": self.prune_threshold,
                "iterations": iteration_details,
            },
        )
        write_round_manifest(
            self.graph_utils.store,
            WorkflowDirs.PRUNED,
            self.round,
            parent=source,
            source_selector=source.selector,
            metrics=metrics,
            summary={
                "baseline_score": baseline_score,
                "pruned_nodes": pruned_nodes,
                "pruned_count": len(pruned_nodes),
                "threshold": self.prune_threshold,
            },
            artifacts=[
                "graph.py",
                "prompt.py",
                "prune_detail_info.json",
                "manifest.json",
            ],
        )

    def _resolve_source_workflow(self, graph_root: str) -> WorkflowRoundRef | None:
        if self.initial_round is not None:
            return WorkflowRoundRef(
                stage="mcts",
                workflow_name=WorkflowDirs.GRAPH,
                round_number=self.initial_round,
                selector="explicit_round",
            )

        round_number = select_best_graph_round(self.graph_utils, self.data_utils, graph_root, self.sample)
        if round_number is None:
            return None
        return WorkflowRoundRef(
            stage="mcts",
            workflow_name=WorkflowDirs.GRAPH,
            round_number=round_number,
            selector="best_score",
        )

    async def _try_prune_candidates(
        self,
        candidates: list[str],
        pruned_nodes: list[str],
        source_graphflow,
        graph_src: str,
        evaluator: Evaluator,
        eval_dir: str,
        baseline_score: float,
        iter_count: int,
    ):
        tried_candidates = []
        threshold = self.prune_threshold * baseline_score

        for candidate in candidates:
            tried_candidates.append(candidate)
            logger.info(f"[PRUNE] Iter {iter_count}: evaluating candidate {candidate}")
            candidate_graph, candidate_code = build_pruned_graphflow(
                source_graphflow,
                pruned_nodes + [candidate],
                graph_src,
            )
            set_workflow_graph(self.graph, candidate_graph)
            candidate_score, candidate_avg_cost, candidate_total_cost = await evaluate_workflow_metrics(
                self,
                evaluator,
                eval_dir,
                self.validation_rounds,
                self.prune_eval_samples,
            )
            logger.info(
                f"[PRUNE] Iter {iter_count}: candidate {candidate} "
                f"score={candidate_score:.5f} threshold={threshold:.5f}"
            )
            if candidate_score >= threshold:
                return {
                    "accepted": True,
                    "candidate": candidate,
                    "graphflow": candidate_graph,
                    "graph_code": candidate_code,
                    "score": candidate_score,
                    "avg_cost": candidate_avg_cost,
                    "total_cost": candidate_total_cost,
                    "tried_candidates": tried_candidates,
                    "threshold_value": threshold,
                }

        logger.info(f"[PRUNE] Iter {iter_count}: no acceptable candidates")
        return {
            "accepted": False,
            "candidate": None,
            "graphflow": None,
            "graph_code": None,
            "score": None,
            "avg_cost": None,
            "total_cost": None,
            "tried_candidates": tried_candidates,
            "threshold_value": threshold,
        }
