import time
import asyncio
import shutil
from dataclasses import dataclass
from typing import Optional, List, Literal

from src.utils.async_llm import create_llm_instance
from src.core.formatter import XmlFormatter, FormatError
from src.optimizer.data_utils import DataUtils
from src.optimizer.evaluation_utils import EvaluationUtils
from src.optimizer.graph_utils import GraphUtils
from src.optimizer.manifest_utils import WorkflowRoundRef, read_round_manifest, write_round_manifest
from src.optimizer.workspace import WorkflowDirs
from src.utils.logs import logger
from src.utils.optimizer_utils.experience_utils import ExperienceUtils
from src.utils.optimizer_utils.response_parsing import extract_graph_optimize_fields
from src.utils.optimizer_utils.schemas import GraphOptimize
from src.prompts.finetune_prompt import WORKFLOW_INPUT, WORKFLOW_OPTIMIZE_PROMPT, WORKFLOW_PROMPT_USE

QuestionType = Literal["math", "code", "qa"]


@dataclass(frozen=True)
class FinetuneRunPlan:
    seed_round: int
    next_round: int
    target_round: int
    needs_seed_evaluation: bool
    seed_source: WorkflowRoundRef

    @property
    def has_new_rounds(self) -> bool:
        return self.next_round <= self.target_round


class FinetuneOptimizer:
    def __init__(
        self,
        workspace: str = "workspace",
        dataset: str = "MATH",
        question_type: QuestionType = "math",
        opt_llm_config: Optional[dict] = None,
        exec_llm_config: Optional[dict] = None,
        low_exec_model: Optional[str] = None,
        operators: Optional[List[str]] = None,
        sample: int = 3,
        validation_rounds: int = 5,
        max_rounds: int = 10,
        sample_size: Optional[int] = None,
    ):
        self.workspace = workspace.replace("\\", "/").rstrip("/")
        self.dataset = dataset
        self.type = question_type

        if opt_llm_config is None:
            raise ValueError("opt_llm_config is required.")
        if exec_llm_config is None:
            raise ValueError("exec_llm_config is required.")
        self.optimize_llm_config = opt_llm_config
        self.execute_llm_config = exec_llm_config
        self.low_exec_model = low_exec_model
        self.optimize_llm = create_llm_instance(self.optimize_llm_config)

        self.operators = operators or []
        self.sample = sample
        self.validation_rounds = validation_rounds
        self.max_rounds = max_rounds
        self.sample_size = sample_size

        self.root_path = f"{self.workspace}/{self.dataset}"
        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path, WorkflowDirs.FINETUNED)

        self.graph = None
        self.round = 1

    def run(self) -> None:
        finetune_root = str(self.graph_utils.store.workflow_path(WorkflowDirs.FINETUNED))
        quantized_root = str(self.graph_utils.store.workflow_path(WorkflowDirs.QUANTIZED))
        run_plan = self._build_run_plan(finetune_root, quantized_root)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        eval_dir = self._evaluation_dir()
        validation_data = self.data_utils.load_results(finetune_root)
        if run_plan.needs_seed_evaluation:
            self.round = run_plan.seed_round
            loop.run_until_complete(
                self.evaluation_utils.evaluate_initial_round(
                    optimizer=self,
                    graph_path=finetune_root,
                    directory=eval_dir,
                    validation_n=self.validation_rounds,
                    data=validation_data,
                    sample_size=self.sample_size,
                )
            )
            logger.info(
                f"[FINETUNE] Evaluated seed round_{run_plan.seed_round} "
                f"from {self._source_label(run_plan.seed_source)}."
            )
        else:
            logger.info(
                f"[FINETUNE] Reusing evaluated seed round_{run_plan.seed_round} "
                f"from {self._source_label(run_plan.seed_source)}."
            )
        self._ensure_seed_manifest(run_plan, finetune_root)

        completed_round = run_plan.seed_round
        if not run_plan.has_new_rounds:
            logger.info(
                f"[FINETUNE] No new rounds to generate. "
                f"Current seed round_{run_plan.seed_round} already meets target max_rounds={self.max_rounds}."
            )

        for round_number in range(run_plan.next_round, run_plan.target_round + 1):
            self.round = round_number
            score = loop.run_until_complete(self._optimize_graph(graph_path=finetune_root))
            if score is None:
                logger.warning(f"[FINETUNE][SKIP] Round {self.round} produced no valid score. Skipping to next round.")
                time.sleep(1)
                continue
            logger.info(f"[FINETUNE] Round {self.round} - Score: {score:.5f}")
            completed_round = self.round
            time.sleep(1)

        loop.close()
        logger.info(f"[FINETUNE] Completed finetuning through round {completed_round}")

    def _build_run_plan(self, finetune_root: str, quantized_root: str) -> FinetuneRunPlan:
        latest_finetune_round = self._latest_valid_round(finetune_root)
        if latest_finetune_round is not None:
            return FinetuneRunPlan(
                seed_round=latest_finetune_round,
                next_round=latest_finetune_round + 1,
                target_round=self.max_rounds,
                needs_seed_evaluation=not self.data_utils.has_round_results(
                    finetune_root, latest_finetune_round
                ),
                seed_source=WorkflowRoundRef(
                    stage="finetune",
                    workflow_name=WorkflowDirs.FINETUNED,
                    round_number=latest_finetune_round,
                    selector="latest_round",
                ),
            )

        latest_quantized_round = self._require_quantized_round(quantized_root)
        self._initialize_seed_round(finetune_root, quantized_root, latest_quantized_round)
        return FinetuneRunPlan(
            seed_round=1,
            next_round=2,
            target_round=self.max_rounds,
            needs_seed_evaluation=True,
            seed_source=WorkflowRoundRef(
                stage="quantize",
                workflow_name=WorkflowDirs.QUANTIZED,
                round_number=latest_quantized_round,
                selector="latest_round",
            ),
        )

    def _latest_valid_round(self, workflow_root: str) -> int | None:
        latest_round = self.graph_utils.latest_round(workflow_root)
        if latest_round is None:
            return None

        for round_number in range(latest_round, 0, -1):
            if self.graph_utils.has_graph_files(workflow_root, round_number):
                return round_number
        return None

    def _require_quantized_round(self, quantized_root: str) -> int:
        latest_quantized_round = self._latest_valid_round(quantized_root)
        if latest_quantized_round is None:
            raise RuntimeError(
                f"No quantized rounds found under {quantized_root}. Please run quantize_pipeline.py first."
            )
        return latest_quantized_round

    def _initialize_seed_round(
        self,
        finetune_root: str,
        quantized_root: str,
        quantized_round: int,
    ) -> None:
        prompt_src, graph_src = self.graph_utils.read_graph_files(
            quantized_round,
            quantized_root,
        )
        round1_dir = self.graph_utils.create_round_directory(finetune_root, 1)
        self.graph_utils.write_workflow_files(round1_dir, graph_src, prompt_src)
        logger.info(
            f"[FINETUNE] No existing finetuned rounds. "
            f"Initialized seed round_1 from quantized round_{quantized_round}."
        )

    async def _optimize_graph(self, graph_path: str):
        validation_n = self.validation_rounds
        data = self.data_utils.load_results(graph_path)

        directory = self.graph_utils.create_round_directory(graph_path, self.round)

        all_top_rounds = self.data_utils.get_top_rounds(self.sample, path=graph_path, mode="Finetune")
        existing_rounds = []
        for item in all_top_rounds:
            if self.graph_utils.has_graph_files(graph_path, item["round"]):
                existing_rounds.append(item)

        if not existing_rounds:
            raise RuntimeError(f"No valid finetuned rounds found under {graph_path}")

        sample = self.data_utils.select_round(existing_rounds)

        round_number = int(sample["round"])
        logger.info(f"[FINETUNE] Loading finetuned graph for round {round_number} from: {graph_path}")

        prompt_src, graph_src = self.graph_utils.read_graph_files(round_number, graph_path)
        workflow_source = self.graph_utils.extract_workflow_source(graph_src)

        processed_experience = self.experience_utils.load_experience()
        experience = self.experience_utils.format_experience(processed_experience, sample["round"])

        operator_spec = self.graph_utils.format_operator_specs(self.operators)
        log_file = str(self.graph_utils.store.log_file(graph_path, sample["round"]))
        log_data = self.data_utils.load_log(sample["round"], path=log_file, mode="Finetune")

        graph_optimize_prompt = self._create_graph_optimize_prompt(
            experience=experience,
            score=sample["score"],
            graph=workflow_source,
            prompt=prompt_src,
            operator_spec=operator_spec,
            type=self.type,
            log_data=log_data,
            low_model=self.low_exec_model
        )

        self.graph_utils.write_text(directory, "full_context.txt", graph_optimize_prompt)
        logger.info(f"[FINETUNE] Complete context saved to {directory}/full_context.txt")

        try:
            formatter = XmlFormatter.from_model(GraphOptimize)
            response = await self.optimize_llm.call_with_format(graph_optimize_prompt, formatter)
        except FormatError as e:
            logger.error(f"[FINETUNE] Format error: {str(e)}")
            raw_response = await self.optimize_llm(graph_optimize_prompt)
            response = extract_graph_optimize_fields(raw_response)
            if not response:
                logger.error("[FINETUNE] Failed to extract fields from raw response")
                self._record_fail("failed_to_format", str(e))
                shutil.rmtree(directory, ignore_errors=True)
                return None

        if not self.experience_utils.check_modification(processed_experience, response["modification"], sample["round"]):
            logger.warning("[FINETUNE] Modification rejected by experience filter.")
            shutil.rmtree(directory, ignore_errors=True)
            return None

        self.graph_utils.write_graph_files(directory, response)

        try:
            self.graph = self.graph_utils.load_graph(self.round, graph_path)
        except Exception as e:
            logger.error(f"[FINETUNE][SKIP] Failed to load generated graph in round {self.round}: {e}")
            self._record_fail("failed_to_load_graph", str(e))
            shutil.rmtree(directory, ignore_errors=True)
            return None

        try:
            avg_score = await self.evaluation_utils.evaluate_graph(
                optimizer=self,
                directory=self._evaluation_dir(),
                validation_n=validation_n,
                data=data,
                graph_path=graph_path,
                sample_size=self.sample_size
            )
        except Exception as e:
            logger.error(f"[FINETUNE][SKIP] Failed to evaluate graph in round {self.round}: {e}")
            self._record_fail("failed_to_evaluate", str(e))
            shutil.rmtree(directory, ignore_errors=True)
            return None

        self.experience_utils.update_experience(
            directory,
            self.experience_utils.create_experience_data(
                {"round": round_number, "score": sample["score"]},
                response["modification"],
            ),
            avg_score
        )
        self._write_round_manifest(
            parent=WorkflowRoundRef(
                stage="finetune",
                workflow_name=WorkflowDirs.FINETUNED,
                round_number=round_number,
                selector="score_sampling",
            ),
            source_selector="score_sampling",
            summary={
                "modification": response["modification"],
                "source_round": round_number,
            },
            artifacts=[
                "graph.py",
                "prompt.py",
                "experience.json",
                "full_context.txt",
                "manifest.json",
            ],
            graph_path=graph_path,
        )
        return avg_score

    def _evaluation_dir(self) -> str:
        eval_dir = self.graph_utils.store.evaluation_path("finetune")
        eval_dir.mkdir(parents=True, exist_ok=True)
        return str(eval_dir)

    def _ensure_seed_manifest(self, run_plan: FinetuneRunPlan, graph_path: str) -> None:
        existing_manifest = read_round_manifest(
            self.graph_utils.store,
            WorkflowDirs.FINETUNED,
            run_plan.seed_round,
        )
        if existing_manifest is not None:
            return

        parent = None
        source_selector = run_plan.seed_source.selector or "recovered_existing_round"
        summary = {"seed_round": True}
        if not (
            run_plan.seed_source.workflow_name == WorkflowDirs.FINETUNED
            and run_plan.seed_source.round_number == run_plan.seed_round
        ):
            parent = run_plan.seed_source
        else:
            summary["note"] = "Recovered existing finetune round before manifest support."

        write_round_manifest(
            self.graph_utils.store,
            WorkflowDirs.FINETUNED,
            run_plan.seed_round,
            parent=parent,
            source_selector=source_selector,
            metrics=self.data_utils.summarize_round_results(graph_path, run_plan.seed_round) or {},
            summary=summary,
            artifacts=[
                "graph.py",
                "prompt.py",
                "manifest.json",
            ],
        )

    def _write_round_manifest(
        self,
        *,
        parent: WorkflowRoundRef,
        source_selector: str,
        summary: dict,
        artifacts: list[str],
        graph_path: str,
    ) -> None:
        metrics = self.data_utils.summarize_round_results(graph_path, self.round) or {}
        write_round_manifest(
            self.graph_utils.store,
            WorkflowDirs.FINETUNED,
            self.round,
            parent=parent,
            source_selector=source_selector,
            metrics=metrics,
            summary=summary,
            artifacts=artifacts,
        )

    @staticmethod
    def _source_label(source: WorkflowRoundRef) -> str:
        return f"{source.stage} round_{source.round_number}"

    def _create_graph_optimize_prompt(self, experience: str, score: float, graph: str, prompt: str, operator_spec: str, type: str, log_data: str, low_model: str = None) -> str:
        if low_model:
            operator_spec += f"\n\nLow-cost model available: {low_model} (use this for less critical nodes to reduce cost)"

        graph_input = WORKFLOW_INPUT.format(
            experience=experience,
            score=score,
            graph=graph,
            prompt=prompt,
            operator_spec=operator_spec,
            type=type,
            log=log_data,
        )
        graph_system = WORKFLOW_OPTIMIZE_PROMPT.format(type=type)
        graph_build = WORKFLOW_PROMPT_USE.format(type=type)
        return graph_input + graph_build + graph_system

    def _record_fail(self, status: str, error: str):
        fail_entry = {
            "round": self.round,
            "status": status,
            "error": error,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        fail_log_path = self.graph_utils.store.root_path / "finetune_failed_rounds.jsonl"
        self.graph_utils.append_jsonl(str(fail_log_path), fail_entry)
