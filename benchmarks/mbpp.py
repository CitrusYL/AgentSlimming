from typing import Any, Callable, Dict, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from benchmarks.code_check import (
    CodeCheckRequest,
    TIMEOUT_FAILURE_MESSAGE,
    run_code_check,
)
from src.utils.logs import logger

class MBPPBenchmark(BaseBenchmark):
    REQUIRED_FIELDS = ("prompt", "entry_point", "code", "test")

    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def check_solution(self, solution, test, entry_point):
        result = run_code_check(
            CodeCheckRequest(
                solution=solution,
                test=test,
                entry_point=entry_point,
                timeout=15,
                check_mode="no_args",
            )
        )
        if result[0] == self.FAIL and result[1] != TIMEOUT_FAILURE_MESSAGE:
            with open("error.log", "a", encoding="utf-8") as log_file:
                log_file.write(result[1] + "\n")
        return result

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, prompt, entry_point):
        return await graph(prompt, entry_point)

    async def evaluate_problem(self, data: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = data["prompt"]
        expected_output = "\nCorrect Solution:\ndef " + data["code"]

        try:
            # Generate prediction using the graph function
            prediction, cost = await self._generate_output(graph, input_text, data["entry_point"])

            # Check the solution
            ret = self.check_solution(prediction, data["test"], data["entry_point"])
            test_case_details = ret[1]
            expected_output = test_case_details + "\nCorrect Solution:" + data["code"]

            # Calculate score based on the check result
            score = 1.0 if ret[0] == self.PASS else 0.0

            # Log mismatch if the score is 0
            if score == 0:
                self.log_mismatch(input_text, expected_output, prediction, score)

            return input_text, prediction, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        # The scoring logic for MBPP is already implemented in evaluate_problem, this is just to conform to the interface
        return 0.0, prediction

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]
