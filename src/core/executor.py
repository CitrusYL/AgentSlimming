import asyncio
import time
from typing import Any, ClassVar, Dict, Tuple

from pydantic import BaseModel

from src.core.graphflow import GraphFlow
from src.core.nodes.base import Node
from src.utils.cost_tracker import LLMCostTracker
from src.utils.logs import logger


class GraphExecutor(BaseModel):
    workflow_graph: GraphFlow
    ALLOWED_PROBLEM_CONTEXT_KEYS: ClassVar[set[str]] = {"entry_point", "question_id"}

    async def run(self, problem: Any) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        start_time = time.time()
        result: Dict[str, Any] = {}
        cost_tracker = LLMCostTracker()

        topo_order = self.workflow_graph.get_topological_order()
        logger.debug(f"[EXE] Topological order: {topo_order}")

        for node_id in topo_order:
            node: Node = self.workflow_graph.get_node(node_id)
            input_data = self._build_node_input(node, result, problem)
            try:
                if asyncio.iscoroutinefunction(node.execute):
                    node_result = await node.execute(input_data)
                else:
                    node_result = node.execute(input_data)

                self._record_llm_usage(cost_tracker, node_id, node,node_result)

            except Exception as e:
                node_result = {"success": False, "error": str(e), "node_id": node_id}

            result[node_id] = node_result

        execution_time = time.time() - start_time
        cost_summary = cost_tracker.get_cost_summary()

        logger.info(f"[COST] Total execution cost: ${cost_summary['total_cost_usd']:.6f}")
        logger.info(f"[COST] Total LLM calls: {cost_summary['total_calls']}")
        logger.info(f"[COST] Total tokens: {cost_summary['total_tokens']}")

        final_result = result.get(self.workflow_graph.final_node_id, {})
        return final_result, execution_time, cost_summary

    def _build_node_input(
        self,
        node: Node,
        previous_results: Dict[str, Any],
        problem: Any,
    ) -> Dict[str, Any]:
        problem_text, problem_context = self._normalize_problem(problem)
        input_data = {
            edge.input_name(): previous_results.get(edge.source, {})
            for edge in self.workflow_graph.incoming_edges(node.node_id)
        }
        if node.accepts_problem_input:
            input_data["problem"] = problem_text
        for key, value in problem_context.items():
            input_data.setdefault(key, value)
        return input_data

    @staticmethod
    def _normalize_problem(problem: Any) -> tuple[Any, Dict[str, Any]]:
        if not isinstance(problem, dict):
            return problem, {}

        unexpected_keys = sorted(
            key
            for key in problem
            if key not in {"problem", *GraphExecutor.ALLOWED_PROBLEM_CONTEXT_KEYS}
        )
        if unexpected_keys:
            raise ValueError(f"Unsupported problem context keys: {unexpected_keys}")

        problem_text = problem.get("problem", problem)
        problem_context = {
            key: value
            for key, value in problem.items()
            if key in GraphExecutor.ALLOWED_PROBLEM_CONTEXT_KEYS
        }
        return problem_text, problem_context

    def _record_llm_usage(
        self,
        cost_tracker: LLMCostTracker,
        node: Node,
        node_id: str,
        node_result: Any,
    ) -> None:
        if not isinstance(node_result, dict):
            return
        
        if not getattr(node, "count_towards_cost", True):
            return

        for usage in node_result.get("llm_usage") or []:
            prices = usage.get("prices") or {}
            cost_tracker.add_llm_call(
                node_id=node_id,
                model_name=usage.get("model", "unknown"),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                input_price=prices.get("input_price"),
                output_price=prices.get("output_price"),
            )
