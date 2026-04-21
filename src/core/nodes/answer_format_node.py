from typing import Any, ClassVar, Dict, Optional

from pydantic import Field

from src.catalog.answer_formats import get_answer_format_requirements
from src.core.nodes.base import Node
from src.core.nodes.node_util import DEFAULT_RESULT_FIELDS, first_text, pick_text


class AnswerFormatNode(Node):
    spec_name: ClassVar[str] = "AnswerFormat"
    dataset: Optional[str] = Field(
        default=None,
        alias="dataset_name",
        description="Target dataset name for formatting.",
    )

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            original = first_text(inputs)
            if original is None:
                original = self._recover_from_failed_inputs(inputs)
            if original is None:
                return self.failure("No answer found to format.")

            dataset_name = self.dataset or ""
            format_requirements = get_answer_format_requirements(dataset_name)

            prompt_inputs = {
                **inputs,
                "original_answer": original,
                "dataset_name": dataset_name,
                "format_requirements": format_requirements,
                "task_context": self._build_task_context(inputs),
            }
            response, parsed, llm_usage = await self.generate(**prompt_inputs)
            formatted = pick_text(parsed, "response", "output")
            if formatted:
                return self.success(output=formatted, llm_usage=llm_usage)

            output = (
                response.strip()
                if isinstance(response, str)
                else str(response)
            )
            return self.success(output=output, llm_usage=llm_usage)
        except Exception as e:
            return self.failure(f"AnswerFormatNode failed: {str(e)}")

    @staticmethod
    def _build_task_context(inputs: Dict[str, Any]) -> str:
        parts = []

        problem = inputs.get("problem")
        if isinstance(problem, str) and problem.strip():
            parts.append(f"Problem:\n{problem.strip()}")

        entry_point = inputs.get("entry_point")
        if isinstance(entry_point, str) and entry_point.strip():
            parts.append(f"Entry point:\n{entry_point.strip()}")

        question_id = inputs.get("question_id")
        if isinstance(question_id, str) and question_id.strip():
            parts.append(f"Question ID:\n{question_id.strip()}")

        return "\n\n".join(parts) if parts else "No additional context provided."

    @staticmethod
    def _recover_from_failed_inputs(inputs: Dict[str, Any]) -> Optional[str]:
        for value in inputs.values():
            if not isinstance(value, dict):
                continue
            for field in (*DEFAULT_RESULT_FIELDS, "solution", "error"):
                candidate = value.get(field)
                if candidate is not None and str(candidate).strip():
                    return str(candidate)
        return None
