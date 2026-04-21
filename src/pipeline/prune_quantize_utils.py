from src.core.graphflow import GraphFlow
from src.evaluation.evaluator import Evaluator
from src.optimizer.manifest_utils import WorkflowRoundRef
from src.optimizer.pruners.betweenness import BetweennessPruner
from src.optimizer.pruners.degree import DegreePruner
from src.optimizer.pruners.traverse import TraversePruner
from src.optimizer.quantizers.betweenness import BetweennessQuantizer
from src.optimizer.quantizers.degree import DegreeQuantizer
from src.optimizer.quantizers.traverse import TraverseQuantizer
from src.utils.logs import logger
from src.utils.optimizer_utils.ranking import MetricRanking, RankingResult, reciprocal_rank_fusion

def set_workflow_graph(workflow, graphflow: GraphFlow) -> None:
    workflow.workflow_graph = graphflow
    workflow.executor.workflow_graph = graphflow


def select_best_graph_round(graph_utils, data_utils, graph_root: str, sample: int) -> int | None:
    valid_rounds = [
        item
        for item in data_utils.get_top_rounds(sample)
        if graph_utils.has_graph_files(graph_root, item["round"])
    ]
    if not valid_rounds:
        return None
    return int(max(valid_rounds, key=lambda item: item["score"])["round"])


async def evaluate_graph_variant(
    pipeline,
    evaluator: Evaluator,
    graphflow: GraphFlow,
    eval_dir: str,
    sample_size: int | None,
) -> tuple[float, float, float]:
    original_graphflow = pipeline.graph.workflow_graph
    set_workflow_graph(pipeline.graph, graphflow)
    try:
        return await evaluator.graph_evaluate(
            pipeline.dataset,
            pipeline.graph,
            {"dataset": pipeline.dataset, "llm_config": pipeline.execute_llm_config},
            eval_dir,
            is_test=False,
            sample_size=sample_size,
        )
    finally:
        set_workflow_graph(pipeline.graph, original_graphflow)


async def evaluate_workflow_metrics(
    pipeline,
    evaluator: Evaluator,
    eval_dir: str,
    validation_rounds: int,
    sample_size: int | None,
) -> tuple[float, float, float]:
    total_score = 0.0
    total_avg_cost = 0.0
    total_cost = 0.0

    for _ in range(validation_rounds):
        score, avg_cost, run_total_cost = await evaluator.graph_evaluate(
            pipeline.dataset,
            pipeline.graph,
            {"dataset": pipeline.dataset, "llm_config": pipeline.execute_llm_config},
            eval_dir,
            is_test=False,
            sample_size=sample_size,
        )
        total_score += score
        total_avg_cost += avg_cost
        total_cost += run_total_cost

    return (
        total_score / validation_rounds,
        total_avg_cost / validation_rounds,
        total_cost,
    )


async def build_prune_ranking(
    pipeline,
    graphflow: GraphFlow,
    evaluator: Evaluator,
    eval_dir: str,
    excluded: set[str] | None = None,
) -> RankingResult:
    excluded = excluded or set()
    degree_scores = DegreePruner(graphflow).identify_pruning_candidates()
    betweenness_scores = BetweennessPruner(graphflow).identify_pruning_candidates()
    traverse_graphs, _ = TraversePruner(graphflow).get_pruned()
    traverse_graphs = {
        node_id: candidate_graph
        for node_id, candidate_graph in traverse_graphs.items()
        if node_id not in excluded
    }
    traverse_scores, delta_costs = await _evaluate_variant_scores(
        pipeline,
        evaluator,
        graphflow,
        traverse_graphs,
        eval_dir,
        pipeline.traverse_sample_size,
    )

    ranking = reciprocal_rank_fusion(
        metrics=[
            MetricRanking("degree", degree_scores, reverse=True, missing_value=-1.0),
            MetricRanking("betweenness", betweenness_scores, reverse=False, missing_value=float("inf")),
            MetricRanking("traverse", traverse_scores, reverse=True, missing_value=float("-inf")),
            MetricRanking("delta", delta_costs, reverse=True, missing_value=float("-inf")),
        ],
        weights={
            "degree": pipeline.w1,
            "betweenness": pipeline.w2,
            "traverse": pipeline.w3,
            "delta": pipeline.w4,
        },
        excluded=excluded,
    )
    _log_ranking("[PRUNE]", ranking)
    return ranking


async def build_quantize_ranking(
    pipeline,
    graphflow: GraphFlow,
    evaluator: Evaluator,
    eval_dir: str,
    excluded: set[str] | None = None,
) -> RankingResult:
    excluded = excluded or set()
    degree_quantizer = DegreeQuantizer(graphflow)
    degree_quantizer.low_model_name = pipeline.quantize_low_model_name
    degree_scores = degree_quantizer.identify_quantization_candidates()

    betweenness_quantizer = BetweennessQuantizer(graphflow)
    betweenness_quantizer.low_model_name = pipeline.quantize_low_model_name
    betweenness_scores = betweenness_quantizer.identify_quantization_candidates()

    traverse_quantizer = TraverseQuantizer(graphflow)
    traverse_quantizer.low_model_name = pipeline.quantize_low_model_name
    traverse_graphs, _ = traverse_quantizer.get_quantized()
    traverse_graphs = {
        node_id: candidate_graph
        for node_id, candidate_graph in traverse_graphs.items()
        if node_id not in excluded
    }
    traverse_scores, delta_costs = await _evaluate_variant_scores(
        pipeline,
        evaluator,
        graphflow,
        traverse_graphs,
        eval_dir,
        pipeline.traverse_sample_size,
    )

    ranking = reciprocal_rank_fusion(
        metrics=[
            MetricRanking("degree", degree_scores, reverse=True, missing_value=-1.0),
            MetricRanking("betweenness", betweenness_scores, reverse=False, missing_value=float("inf")),
            MetricRanking("traverse", traverse_scores, reverse=True, missing_value=float("-inf")),
            MetricRanking("delta", delta_costs, reverse=True, missing_value=float("-inf")),
        ],
        weights={
            "degree": pipeline.w1,
            "betweenness": pipeline.w2,
            "traverse": pipeline.w3,
            "delta": pipeline.w4,
        },
        excluded=excluded,
    )
    _log_ranking("[QUANTIZE]", ranking)
    return ranking


async def _evaluate_variant_scores(
    pipeline,
    evaluator: Evaluator,
    baseline_graph: GraphFlow,
    candidate_graphs: dict[str, GraphFlow],
    eval_dir: str,
    sample_size: int | None,
) -> tuple[dict[str, float], dict[str, float]]:
    if not candidate_graphs:
        return {}, {}

    _, baseline_avg_cost, _ = await evaluate_graph_variant(
        pipeline,
        evaluator,
        baseline_graph,
        eval_dir,
        sample_size,
    )

    scores = {}
    delta_costs = {}
    total = len(candidate_graphs)
    for index, (node_id, candidate_graph) in enumerate(candidate_graphs.items(), start=1):
        logger.info(f"[RANKING] Traverse evaluation {index}/{total} on node {node_id}")
        score, avg_cost, _ = await evaluate_graph_variant(
            pipeline,
            evaluator,
            candidate_graph,
            eval_dir,
            sample_size,
        )
        scores[node_id] = score
        delta_costs[node_id] = baseline_avg_cost - avg_cost

    return scores, delta_costs


def _log_ranking(prefix: str, ranking: RankingResult) -> None:
    logger.info(f"{prefix} RRF fused: {ranking.fused}")
    for metric_name, ordered_nodes in ranking.rankings.items():
        metric_scores = ranking.scores[metric_name]
        logger.info(f"{prefix} {metric_name} order: {[(node, metric_scores.get(node)) for node in ordered_nodes]}")
