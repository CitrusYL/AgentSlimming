from abc import ABC, abstractmethod
from typing import Any

from src.core.executor import GraphExecutor
from src.core.graphflow import GraphFlow
from src.utils.async_llm import resolve_llm_config


OUTPUT_FIELDS = ("output", "response", "answer", "code", "solution", "error")


def normalize_llm_config(llm_config: Any) -> dict[str, Any]:
    if llm_config is None:
        raise ValueError("Workflow llm_config is required and must come from config/config.yaml.")
    return resolve_llm_config(llm_config).to_dict()


class BaseWorkflow(ABC):
    def __init__(self, name: str, dataset: str, llm_config: Any) -> None:
        self.name = name
        self.dataset = dataset
        self.llm_config = normalize_llm_config(llm_config)
        self.workflow_graph = self._build_graph()
        self.executor = GraphExecutor(workflow_graph=self.workflow_graph)

    @abstractmethod
    def _build_graph(self) -> GraphFlow:
        raise NotImplementedError

    async def __call__(self, problem: str, entry_point: str | None = None):
        result, _, cost_summary = await self.executor.run(
            problem=self._executor_input(problem, entry_point)
        )
        return self._final_output(result), cost_summary.get("total_cost_usd", 0.0)

    def _executor_input(self, problem: str, entry_point: str | None) -> Any:
        if not entry_point:
            return problem
        return {"problem": problem, "entry_point": entry_point}

    @staticmethod
    def _final_output(result: Any) -> str:
        if not isinstance(result, dict):
            return str(result)
        for field in OUTPUT_FIELDS:
            value = result.get(field)
            if value is not None and str(value).strip():
                return str(value)
        return str(result)
