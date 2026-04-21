from typing import Any, ClassVar, Dict, List

from src.core.nodes.base import Node
from src.core.nodes.node_util import collect_texts


class ScEnsembleNode(Node):
    """Selects the most consistent answer from multiple upstream candidates."""

    spec_name: ClassVar[str] = "ScEnsemble"

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            solutions: List[str] = collect_texts(inputs)
            if not solutions:
                return self.failure("No valid upstream output received.", output="")

            answer_mapping = {}
            solution_text = ""
            for index, solution in enumerate(solutions):
                letter = chr(65 + index)
                answer_mapping[letter] = index
                solution_text += f"{letter}: \n{solution}\n\n"

            prompt_inputs = {
                "problem": inputs.get("problem", ""),
                "solutions": solution_text,
            }

            _, parsed, llm_usage = await self.generate(**prompt_inputs)
            answer = str(parsed.get("solution_letter") or "").strip().upper()

            if answer in answer_mapping:
                return self.success(
                    output=solutions[answer_mapping[answer]],
                    llm_usage=llm_usage,
                )

            return self.success(output=solutions[0], llm_usage=llm_usage)
        except Exception as e:
            return self.failure(str(e))
