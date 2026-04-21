import sys
import traceback
from typing import Any, ClassVar, Dict

from src.core.nodes.base import Node
from src.core.nodes.node_util import pick_text
from src.utils.code_execution import extract_python_code, run_python_solve_async
from src.utils.logs import logger


class ProgramNode(Node):
    """Generates Python code, executes it, and returns the final output."""

    spec_name: ClassVar[str] = "Programmer"

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            code = None
            output = None
            all_llm_usage: list[dict[str, Any]] = []

            for attempt in range(3):
                response, parsed, llm_usage = await self.generate(**inputs)
                all_llm_usage.extend(llm_usage)
                code = pick_text(parsed, "code", "response", "output")
                if not code and isinstance(response, str):
                    code = response
                if not code:
                    return self.failure("No code generated", llm_usage=all_llm_usage)

                code = extract_python_code(code)
                status, output = await run_python_solve_async(code)

                if status == "Success":
                    return self.success(
                        code=code,
                        output=output,
                        llm_usage=all_llm_usage,
                    )

                logger.warning(
                    f"[ProgramNode] attempt {attempt + 1} failed, output={output}"
                )

            return self.failure(
                "Generated code failed to execute.",
                code=code,
                output=output,
                llm_usage=all_llm_usage,
            )

        except Exception as e:
            tb_str = "".join(traceback.format_exception(*sys.exc_info()))
            return self.failure(str(e), traceback=tb_str)
