from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.core.nodes.node_util import build_prompt, compose_prompt, parse_response
from src.core.nodes.catalog import NodeDefinition, RuntimeNodeSpec, get_node_definition
from src.utils import async_llm


class Node(BaseModel, ABC):
    """Base class for workflow nodes."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    spec_name: ClassVar[str]

    node_id: str = ""
    node_description: Optional[str] = Field(default=None, alias="description")
    node_prompt: Optional[str] = None
    node_llm_config: Dict[str, Any] = Field(...)
    count_towards_cost: bool = Field(
        default=True,
        description="Whether LLM usage from this node should be included in final cost summary.",
    )
    
    @property
    def definition(self) -> NodeDefinition:
        return get_node_definition(self.spec_name)

    @property
    def spec(self) -> RuntimeNodeSpec:
        return self.definition.runtime_spec

    @property
    def accepts_problem_input(self) -> bool:
        return self.spec.include_problem

    def prompt_template(self) -> str:
        return compose_prompt(self.node_prompt, self.spec.default_prompt)

    def render_prompt(self, **inputs: Any) -> str:
        return build_prompt(
            self.node_id,
            self.spec,
            self.prompt_template(),
            **inputs,
        )

    def create_llm(self):
        return async_llm.create_llm_instance(self.node_llm_config)

    def parse_response(self, output: str, **context: Any) -> dict[str, Any]:
        return parse_response(
            output,
            self.spec,
            function_name=context.get("function_name"),
        )

    async def generate(self, **inputs: Any) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        prompt = self.render_prompt(**inputs)
        llm = self.create_llm()
        response = await llm(prompt)
        parsed = self.parse_response(response, **inputs)
        return response, parsed, self.collect_llm_usage(llm)

    def success(self, **payload: Any) -> Dict[str, Any]:
        return {"success": True, **payload}

    def failure(self, error: str, **payload: Any) -> Dict[str, Any]:
        return {"error": error, "node_id": self.node_id, "success": False, **payload}

    @staticmethod
    def collect_llm_usage(llm) -> list[dict[str, Any]]:
        return getattr(getattr(llm, "usage_tracker", None), "usage_history", [])

    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
