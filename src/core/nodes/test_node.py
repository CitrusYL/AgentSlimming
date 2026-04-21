import sys
import traceback
from typing import Any, ClassVar, Dict, Optional

from src.core.nodes.base import Node
from src.core.nodes.node_util import first_text, pick_text
from src.utils.code_execution import extract_python_code, run_python_solve_async


class TestNode(Node):
    """Executes generated code and asks the model to repair it on failure."""

    spec_name: ClassVar[str] = "Test"
    dataset: Optional[str] = None

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            problem = inputs.get("problem", "")
            code = first_text(inputs, fields=("code", "output", "response"))
            all_llm_usage = []

            if not code:
                return self.failure(
                    "No code found to test.",
                    result=False,
                    solution="",
                    llm_usage=all_llm_usage,
                )

            for _ in range(3):
                code = extract_python_code(code)
                status, output = await run_python_solve_async(code)

                if status == "Success":
                    return self.success(
                        result=True,
                        solution=code,
                        output=code,
                        llm_usage=all_llm_usage,
                    )

                err_info = (
                    f"Previous code failed.\n"
                    f"Code:\n{code}\n\n"
                    f"Status: {status}\nOutput: {output}"
                )

                response, parsed, llm_usage = await self.generate(
                    problem=problem,
                    error_info=err_info,
                )
                all_llm_usage.extend(llm_usage)
                new_code = pick_text(parsed, "solution", "code", "output", "response")
                if not new_code and isinstance(response, str):
                    new_code = response.strip()

                if not new_code:
                    break

                code = new_code

            code = extract_python_code(code)
            status, _ = await run_python_solve_async(code)
            if status == "Success":
                return self.success(
                    result=True,
                    solution=code,
                    output=code,
                    llm_usage=all_llm_usage,
                )

            return self.failure(
                "Code still failed after repair attempts.",
                result=False,
                solution=code,
                output=code,
                llm_usage=all_llm_usage,
            )

        except Exception as e:
            tb_str = "".join(traceback.format_exception(*sys.exc_info()))
            return self.failure(str(e), traceback=tb_str)
