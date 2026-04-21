from typing import Any, ClassVar, Dict

from src.core.nodes.base import Node
from src.core.nodes.node_util import pick_text, resolve_entry_point


class CustomCodeGenerateNode(Node):
    """Generates runnable Python code for code benchmarks."""

    spec_name: ClassVar[str] = "CustomCodeGenerate"

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            entry_point = resolve_entry_point(inputs)
            prompt_inputs = {**inputs, "function_name": entry_point}
            response, parsed, llm_usage = await self.generate(**prompt_inputs)
            code = pick_text(parsed, "code", "response", "output")

            if code:
                return self.success(
                    code=code,
                    output=code,
                    entry_point=entry_point,
                    llm_usage=llm_usage,
                )

            return self.success(
                output=response.strip() if isinstance(response, str) else str(response),
                entry_point=entry_point,
                llm_usage=llm_usage,
            )
        except Exception as e:
            return self.failure(str(e))
