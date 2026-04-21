import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from benchmarks.code_check import (
    CodeCheckRequest,
    TIMEOUT_FAILURE_MESSAGE,
    run_code_check,
)
from src.utils.logs import logger


class HumanEvalBenchmark(BaseBenchmark):
    REQUIRED_FIELDS = ("prompt", "entry_point", "canonical_solution", "test")

    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def check_solution(self, solution, test, entry_point):
        result = run_code_check(
            CodeCheckRequest(
                solution=solution,
                test=test,
                entry_point=entry_point,
                timeout=15,
                check_mode="candidate",
                prelude=self._special_case_prelude(entry_point),
            )
        )
        if result[0] == self.FAIL and result[1] != TIMEOUT_FAILURE_MESSAGE:
            with open("error.log", "a", encoding="utf-8") as log_file:
                log_file.write(result[1] + "\n")
        return result

    @staticmethod
    def _special_case_prelude(entry_point: str) -> str:
        if entry_point == "decode_cyclic":
            return """
def encode_cyclic(s: str):
    \"\"\"
    returns encoded string by cycling groups of three characters.
    \"\"\"
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)
""".strip()

        if entry_point == "decode_shift":
            return """
def encode_shift(s: str):
    \"\"\"
    returns encoded string by shifting every character by 5 in the alphabet.
    \"\"\"
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])
""".strip()

        if entry_point == "find_zero":
            return """
def poly(xs: list, x: float):
    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))
""".strip()

        return ""

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, prompt, entry_point):
        # Generate output with a timeout of 60 seconds
        return await asyncio.wait_for(graph(prompt, entry_point), timeout=60)

    async def evaluate_problem(self, data: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = data["prompt"]
        expected_output = (
            "\nCorrect Solution:\ndef "
            + data["entry_point"]
            + "(params you should put here):"
            + "\n\n"
            + data["canonical_solution"]
        )

        try:
            # Generate prediction using the graph function
            prediction, cost = await self._generate_output(graph, input_text, data["entry_point"])

            # Check the solution
            ret = self.check_solution(prediction, data["test"], data["entry_point"])
            test_case_details = ret[1]
            expected_output = test_case_details + expected_output

            # Calculate score based on the check result
            score = 1.0 if ret[0] == self.PASS else 0.0

            # Log mismatch if the score is 0
            if score == 0:
                self.log_mismatch(input_text, expected_output, prediction, score)

            return input_text, prediction, expected_output, score, cost

        except asyncio.TimeoutError:
            logger.info("Timeout error. Skipping this sample.")
            return input_text, "Timeout", expected_output, 0.0, 0.0

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        # The scoring logic for HumanEval is already implemented in evaluate_problem, this is just to conform to the interface
        return 0.0, prediction

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]
