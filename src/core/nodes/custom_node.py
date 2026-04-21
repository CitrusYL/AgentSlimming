from typing import Any, ClassVar, Dict

from src.core.nodes.base import Node
from src.core.nodes.node_util import pick_text


class CustomNode(Node):
    """LLM reasoning node driven by a custom prompt."""

    spec_name: ClassVar[str] = "Custom"

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response, parsed, llm_usage = await self.generate(**inputs)
            parsed_text = pick_text(parsed, "response", "output")
            if parsed_text:
                return self.success(output=parsed_text, llm_usage=llm_usage)
            return self.success(
                output=response.strip() if isinstance(response, str) else str(response),
                llm_usage=llm_usage,
            )
        except Exception as e:
            return self.failure(str(e))
