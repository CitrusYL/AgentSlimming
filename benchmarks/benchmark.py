import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple

import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from src.utils.logs import logger
from src.utils.common import write_json_file


class BaseBenchmark(ABC):
    REQUIRED_FIELDS: tuple[str, ...] = ()

    def __init__(self, name: str, file_path: str, log_path: str):
        self.name = name
        self.file_path = file_path
        self.log_path = log_path

        Path(self.log_path).mkdir(parents=True, exist_ok=True)

    PASS = "PASS"
    FAIL = "FAIL"

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                sample = json.loads(line)
                self.validate_sample(sample)
                data.append(sample)
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    def validate_sample(self, sample: dict) -> None:
        missing = [field for field in self.REQUIRED_FIELDS if field not in sample]
        if missing:
            raise KeyError(f"{self.name} sample is missing required fields: {missing}")

    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]) -> tuple[float, float, float]:
        df = pd.DataFrame(results, columns=columns)
        avg_score = float(df["score"].mean()) if not df.empty else 0.0
        total_cost = float(df["cost"].sum()) if not df.empty else 0.0
        avg_cost = total_cost / len(df) if len(df) > 0 else 0.0

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{avg_score:.5f}_{current_time}.csv"
        output_file = Path(self.log_path) / filename
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return avg_score, avg_cost, total_cost

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
    ) -> None:
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
        log_file = Path(self.log_path) / "log.json"
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(log_data)
        write_json_file(log_file, data, encoding="utf-8", indent=4)

    @abstractmethod
    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        raise NotImplementedError

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        raise NotImplementedError

    async def evaluate_all_problems(self, data: List[dict], agent: Callable, max_concurrent_tasks: int = 16):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def sem_evaluate(problem):
            async with semaphore:
                return await self.evaluate_problem(problem, agent)

        tasks = [sem_evaluate(problem) for problem in data]
        return await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(data))

    def _save_and_log_results(
        self,
        results: List[Tuple[Any, ...]],
        columns: List[str],
        *,
        include_avg_cost: bool = False,
    ) -> tuple[float, float, float]:
        average_score, average_cost, total_cost = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        if include_avg_cost:
            logger.info(f"Avg Cost:{average_cost:.5f}")
        return average_score, average_cost, total_cost

    async def run_evaluation(self, agent: Callable, va_list: List[int] = None, max_concurrent_tasks: int = 16):
        data = await self.load_data(va_list)
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        return self._save_and_log_results(results, self.get_result_columns())

    async def run_baseline(self, agent: Callable, max_concurrent_tasks: int = 16):
        data = await self.load_data()
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        return self._save_and_log_results(
            results,
            self.get_result_columns(),
            include_avg_cost=True,
        )
