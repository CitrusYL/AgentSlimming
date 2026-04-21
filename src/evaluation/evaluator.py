import inspect
from pathlib import Path
from typing import Any, Optional, Tuple

from src.catalog.datasets import DatasetType, dataset_file_path
from src.core.executor import GraphExecutor
from src.core.graphflow import GraphFlow
from src.core.workflow import OUTPUT_FIELDS
from src.evaluation.registry import get_evaluation_spec


class BenchmarkGraphRunner:
    def __init__(self, graph_instance: Any):
        self.graph_instance = graph_instance
        self.executor = GraphExecutor(workflow_graph=graph_instance) if isinstance(graph_instance, GraphFlow) else None
        self.accepts_extra_args = self._accepts_extra_args(graph_instance) if callable(graph_instance) else False

    async def __call__(self, problem: str, *args) -> tuple[str, float]:
        if callable(self.graph_instance):
            if args and not self.accepts_extra_args:
                raise TypeError(
                    f"{type(self.graph_instance).__name__} does not accept benchmark extra inputs: {args!r}"
                )
            result = await self.graph_instance(problem, *args) if args else await self.graph_instance(problem)
            return self._normalize_result(result)

        if self.executor is not None:
            final_result, _, cost_summary = await self.executor.run(problem)
            return self._normalize_output(final_result), cost_summary.get("total_cost_usd", 0.0)

        raise TypeError(f"Unsupported graph instance: {type(self.graph_instance)!r}")

    @classmethod
    def _normalize_result(cls, result: Any) -> tuple[str, float]:
        if isinstance(result, tuple) and len(result) == 2:
            output, cost = result
            return cls._normalize_output(output), float(cost or 0.0)
        return cls._normalize_output(result), 0.0

    @staticmethod
    def _normalize_output(result: Any) -> str:
        if not isinstance(result, dict):
            return str(result)
        for field in (*OUTPUT_FIELDS, "content"):
            value = result.get(field)
            if value is not None and str(value).strip():
                return str(value)
        raise ValueError(f"Graph result did not produce any of {(*OUTPUT_FIELDS, 'content')}")

    @staticmethod
    def _accepts_extra_args(graph_instance: Any) -> bool:
        signature = inspect.signature(graph_instance)
        parameters = list(signature.parameters.values())
        return any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters) or len(parameters) >= 2


class Evaluator:
    def __init__(self, eval_path: Optional[str] = None):
        self.eval_path = eval_path

    async def graph_evaluate(
        self,
        dataset: DatasetType,
        graph,
        params: dict,
        path: Optional[str] = None,
        is_test: bool = False,
        sample_size: int = None,
        data_path: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        eval_path = path or self.eval_path
        if eval_path is None:
            raise ValueError("Evaluation path is required")

        evaluation_spec = get_evaluation_spec(dataset)
        benchmark_data_path = data_path or str(dataset_file_path(dataset, "test" if is_test else "validate"))
        if not Path(benchmark_data_path).exists():
            raise FileNotFoundError(f"Data file not found: {benchmark_data_path}")

        benchmark = evaluation_spec.benchmark_class(
            name=dataset,
            file_path=benchmark_data_path,
            log_path=eval_path,
        )
        agent = self._build_agent(dataset, graph, params)
        sample_indices = list(range(sample_size)) if sample_size else None
        return await benchmark.run_evaluation(agent, sample_indices)

    def _build_agent(self, dataset: DatasetType, graph: Any, params: dict) -> BenchmarkGraphRunner:
        if isinstance(graph, type):
            graph = self._instantiate_graph(dataset, graph, params)
        return BenchmarkGraphRunner(graph)

    def _instantiate_graph(self, dataset: DatasetType, graph_class: type, params: dict) -> Any:
        dataset_name = params.get("dataset", dataset)
        llm_config = params.get("llm_config")
        return graph_class(name=dataset, llm_config=llm_config, dataset=dataset_name)
