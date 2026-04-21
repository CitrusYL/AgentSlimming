import asyncio
import inspect
import re
from math import isclose
from typing import Any, Callable, List, Tuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from src.utils.logs import logger


class AIMEBenchmark(BaseBenchmark):
    REQUIRED_FIELDS = ("problem", "solution")

    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_model_answer(self, text: str) -> str:
        if text is None:
            return ""
        text = str(text)

        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else text.strip()

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        predicted_answer = self.extract_model_answer(prediction)
        expected_answers = self.extract_reference_answers(expected_output)

        for expected_answer in expected_answers:
            if self.math_equal(predicted_answer, expected_answer):
                return 1, predicted_answer
        return 0, predicted_answer

    def extract_reference_answers(self, text: Any) -> List[str]:
        if text is None:
            return []

        raw_text = str(text).strip()
        if not raw_text:
            return []

        boxed_pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = [match.strip() for match in re.findall(boxed_pattern, raw_text, re.DOTALL) if match.strip()]
        if boxed_matches:
            return list(dict.fromkeys(boxed_matches))

        if re.search(r"\b(?:accepted|either|both|or|and/or)\b", raw_text, re.IGNORECASE):
            numeric_matches = [match.strip() for match in re.findall(r"-?\d+(?:\.\d+)?", raw_text) if match.strip()]
            if numeric_matches:
                return list(dict.fromkeys(numeric_matches))

        fallback = self.extract_model_answer(raw_text)
        return [fallback] if fallback else []

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        if str(prediction).strip() == str(reference).strip():
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction_v = self.parse_digits(prediction)
                reference_v = self.parse_digits(reference)
                return isclose(prediction_v, reference_v, abs_tol=1e-3)
        except Exception:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except Exception:
            pass

        return False

    def is_digit(self, num: Any) -> bool:
        return self.parse_digits(num) is not None

    def parse_digits(self, num: Any):
        num = regex.sub(",", "", str(num).strip())

        num = re.sub(r"^\$|\$$", "", num)
        num = re.sub(r"\\text\{([^}]*)\}", r"\1", num).strip()

        try:
            return float(num)
        except Exception:
            if num.endswith("%"):
                num2 = num[:-1].strip()
                if num2.endswith("\\"):
                    num2 = num2[:-1].strip()
                try:
                    return float(num2) / 100
                except Exception:
                    pass
        return None

    def symbolic_equal(self, a: Any, b: Any) -> bool:
        def _parse(s):
            s = str(s).strip()
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except Exception:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except Exception:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except Exception:
            pass

        return False

    def get_function_code(self, func):
        try:
            return inspect.getsource(func)
        except OSError:
            return "no code"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
        retry=retry_if_exception_type((Exception, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _generate_output(self, graph: Callable, input_text: str):
        return await asyncio.wait_for(graph(input_text), timeout=300)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, int, float]:
        input_text = problem["problem"]
        expected_output = problem["solution"]

        try:
            output, cost = await self._generate_output(graph, input_text)
            uni_score, extracted_output = self.calculate_score(expected_output, output)

            if uni_score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                    extract_answer_code=self.get_function_code(self.extract_model_answer),
                )

            return input_text, output, expected_output, uni_score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost"]
