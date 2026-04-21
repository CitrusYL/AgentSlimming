import asyncio
import time
from typing import List, Literal

from src.utils.async_llm import create_llm_instance
from src.core.formatter import XmlFormatter, FormatError
from src.optimizer.data_utils import DataUtils
from src.optimizer.evaluation_utils import EvaluationUtils
from src.optimizer.graph_utils import GraphUtils
from src.optimizer.manifest_utils import WorkflowRoundRef, write_round_manifest
from src.optimizer.workspace import WorkflowDirs
from src.utils.logs import logger
from src.utils.optimizer_utils.experience_utils import ExperienceUtils
from src.utils.optimizer_utils.convergence_utils import ConvergenceUtils
from src.utils.optimizer_utils.response_parsing import extract_graph_optimize_fields
from src.utils.optimizer_utils.schemas import GraphOptimize

QuestionType = Literal["math", "code", "qa"]


class MCTSPipeline:
    def __init__(
        self,
        workspace: str = "workspace",
        dataset: str = "MATH",
        question_type: QuestionType = "math",
        opt_llm_config: dict | None = None,
        exec_llm_config: dict | None = None,
        operators: List[str] = None,
        sample: int = 3,
        check_convergence: bool = True,
        initial_round: int = 1,
        max_rounds: int = 20,
        validation_rounds: int = 1,
        mcts_eval_samples: int = None,
    ):
        self.workspace = workspace.replace("\\", "/").rstrip("/")
        self.dataset = dataset
        self.type = question_type
        if opt_llm_config is None:
            raise ValueError("opt_llm_config is required.")
        if exec_llm_config is None:
            raise ValueError("exec_llm_config is required.")
        self.optimize_llm_config = opt_llm_config
        self.optimize_llm = create_llm_instance(self.optimize_llm_config)
        self.execute_llm_config = exec_llm_config
        self.operators = operators or []
        self.sample = sample
        self.check_convergence = check_convergence
        self.initial_round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds
        self.mcts_eval_samples = mcts_eval_samples
        self.root_path = f"{self.workspace}/{self.dataset}"
        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)
        self.graph = None
        self.round = self.initial_round

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        graph_path = str(self.graph_utils.store.workflow_path(WorkflowDirs.GRAPH))
        data = self.data_utils.load_results(graph_path)

        while self.round <= self.max_rounds:
            if self.round == 1:
                directory = self.graph_utils.create_round_directory(graph_path, self.round)
                self.graph = self.graph_utils.load_graph(self.round, graph_path)
                avg_score = loop.run_until_complete(
                    self.evaluation_utils.evaluate_graph(
                        self,
                        directory,
                        self.validation_rounds,
                        data,
                        sample_size=self.mcts_eval_samples,
                        graph_path=graph_path,
                    )
                )
                self.experience_utils.update_experience(
                    directory,
                    self.experience_utils.create_experience_data({"round": self.round, "score": avg_score}, "initial graph"),
                    avg_score,
                )
                self._write_round_manifest(
                    parent=None,
                    source_selector="repository_seed",
                    summary={"modification": "initial graph"},
                    artifacts=[
                        "graph.py",
                        "prompt.py",
                        "experience.json",
                        "manifest.json",
                    ],
                )
                logger.info(f"Round {self.round} - Score: {avg_score}")
                self.round += 1
                time.sleep(1)
                continue

            score = loop.run_until_complete(self._optimize_graph(graph_path))
            if score is None:
                logger.warning(f"[SKIP] Round {self.round} produced no valid score. Skipping to next round.")
                self.round += 1
                time.sleep(1)
                continue

            logger.info(f"Round {self.round} - Score: {score}")
            converged, _, _ = self.convergence_utils.check_convergence(top_k=3)
            if converged and self.check_convergence:
                logger.info(f"[MCTS] Convergence detected at round {self.round}")
                break

            self.round += 1
            time.sleep(1)

        loop.close()

    async def _optimize_graph(self, graph_path: str):
        validation_n = self.validation_rounds
        data = self.data_utils.load_results(graph_path)
        directory = self.graph_utils.create_round_directory(graph_path, self.round)

        all_top_rounds = self.data_utils.get_top_rounds(self.sample)
        existing_rounds = []
        for item in all_top_rounds:
            if self.graph_utils.has_graph_files(graph_path, item["round"]):
                existing_rounds.append(item)

        if not existing_rounds:
            raise RuntimeError(f"No valid historical rounds found under {graph_path}")

        sample = self.data_utils.select_round(existing_rounds)

        round_number = int(sample["round"])
        prompt_src, graph_src = self.graph_utils.read_graph_files(round_number, graph_path)
        workflow_source = self.graph_utils.extract_workflow_source(graph_src)
        processed_experience = self.experience_utils.load_experience()
        experience = self.experience_utils.format_experience(processed_experience, sample["round"])
        operator_spec = self.graph_utils.format_operator_specs(self.operators)
        log_data = self.data_utils.load_log(sample["round"])

        graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
            experience=experience,
            score=sample["score"],
            graph=workflow_source,
            prompt=prompt_src,
            operator_spec=operator_spec,
            type=self.type,
            log_data=log_data,
        )

        self.graph_utils.write_text(directory, "full_context.txt", graph_optimize_prompt)

        try:
            formatter = XmlFormatter.from_model(GraphOptimize)
            response = await self.optimize_llm.call_with_format(graph_optimize_prompt, formatter)
        except FormatError:
            raw_response = await self.optimize_llm(graph_optimize_prompt)
            response = extract_graph_optimize_fields(raw_response)
            if not response:
                return None

        if not self.experience_utils.check_modification(processed_experience, response["modification"], sample["round"]):
            return None

        self.graph_utils.write_graph_files(directory, response)

        try:
            self.graph = self.graph_utils.load_graph(self.round, graph_path)
        except Exception as e:
            fail_entry = {"round": self.round, "status": "failed_to_load_graph", "error": str(e)}
            fail_log_path = self.graph_utils.store.root_path / "failed_rounds.jsonl"
            self.graph_utils.append_jsonl(str(fail_log_path), fail_entry)
            return None

        try:
            avg_score = await self.evaluation_utils.evaluate_graph(
                self,
                directory,
                validation_n,
                data,
                sample_size=self.mcts_eval_samples,
                graph_path=graph_path,
            )
        except Exception as e:
            fail_entry = {"round": self.round, "status": "failed_to_evaluate", "error": str(e)}
            fail_log_path = self.graph_utils.store.root_path / "failed_rounds.jsonl"
            self.graph_utils.append_jsonl(str(fail_log_path), fail_entry)
            return None

        self.experience_utils.update_experience(
            directory,
            self.experience_utils.create_experience_data(sample, response["modification"]),
            avg_score,
        )
        self._write_round_manifest(
            parent=WorkflowRoundRef(
                stage="mcts",
                workflow_name=WorkflowDirs.GRAPH,
                round_number=round_number,
                selector="score_sampling",
            ),
            source_selector="score_sampling",
            summary={"modification": response["modification"]},
            artifacts=[
                "graph.py",
                "prompt.py",
                "experience.json",
                "full_context.txt",
                "manifest.json",
            ],
        )
        return avg_score

    def _write_round_manifest(
        self,
        *,
        parent: WorkflowRoundRef | None,
        source_selector: str,
        summary: dict,
        artifacts: list[str],
    ) -> None:
        graph_root = str(self.graph_utils.store.workflow_path(WorkflowDirs.GRAPH))
        metrics = self.data_utils.summarize_round_results(graph_root, self.round) or {}
        write_round_manifest(
            self.graph_utils.store,
            WorkflowDirs.GRAPH,
            self.round,
            parent=parent,
            source_selector=source_selector,
            metrics=metrics,
            summary=summary,
            artifacts=artifacts,
        )
