from typing import Any, ClassVar, Dict

from src.core.nodes.base import Node
from src.core.nodes.node_util import pick_text


class AnswerGenerateNode(Node):
    """Built-in answer generation node."""

    spec_name: ClassVar[str] = "AnswerGenerate"

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response, parsed, llm_usage = await self.generate(**inputs)
            answer = pick_text(parsed, "answer")
            if answer:
                return self.success(output=answer, answer=answer, llm_usage=llm_usage)

            parsed_text = pick_text(parsed, "response", "output")
            if parsed_text:
                return self.success(output=parsed_text, llm_usage=llm_usage)

            return self.success(
                output=response.strip() if isinstance(response, str) else str(response),
                llm_usage=llm_usage,
            )
        except Exception as e:
            return self.failure(str(e))
