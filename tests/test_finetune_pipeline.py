import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.optimizer.workspace import WorkflowDirs, WorkspaceStore
from src.pipeline.finetune_pipeline import FinetuneOptimizer


class FinetuneRunPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.dataset = "MATH"
        self.root = self.workspace / self.dataset
        self.store = WorkspaceStore(self.root)

    def _optimizer(self, max_rounds: int) -> FinetuneOptimizer:
        with patch("src.pipeline.finetune_pipeline.create_llm_instance", return_value=object()):
            return FinetuneOptimizer(
                workspace=str(self.workspace),
                dataset=self.dataset,
                question_type="math",
                opt_llm_config={},
                exec_llm_config={},
                max_rounds=max_rounds,
            )

    def _write_round(self, workflow_name: str, round_number: int, with_files: bool = True) -> Path:
        directory = self.store.create_round_dir(self.store.workflow_path(workflow_name), round_number)
        if with_files:
            (directory / "graph.py").write_text("class Workflow: pass\n", encoding="utf-8")
            (directory / "prompt.py").write_text("PROMPT = 'x'\n", encoding="utf-8")
        return directory

    def _write_results(self, workflow_name: str, rows: list[dict]) -> None:
        result_path = self.store.results_file(self.store.workflow_path(workflow_name))
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(rows), encoding="utf-8")

    def test_build_run_plan_initializes_seed_from_quantized_round(self) -> None:
        self._write_round(WorkflowDirs.QUANTIZED, 3)
        optimizer = self._optimizer(max_rounds=5)

        plan = optimizer._build_run_plan(
            str(self.store.workflow_path(WorkflowDirs.FINETUNED)),
            str(self.store.workflow_path(WorkflowDirs.QUANTIZED)),
        )

        self.assertEqual(plan.seed_round, 1)
        self.assertEqual(plan.next_round, 2)
        self.assertEqual(plan.target_round, 5)
        self.assertTrue(plan.needs_seed_evaluation)
        self.assertTrue(
            self.store.has_graph_files(self.store.workflow_path(WorkflowDirs.FINETUNED), 1)
        )

    def test_build_run_plan_resumes_from_next_round_after_latest_result(self) -> None:
        self._write_round(WorkflowDirs.FINETUNED, 4)
        self._write_results(WorkflowDirs.FINETUNED, [{"round": 4, "score": 0.8}])
        optimizer = self._optimizer(max_rounds=6)

        plan = optimizer._build_run_plan(
            str(self.store.workflow_path(WorkflowDirs.FINETUNED)),
            str(self.store.workflow_path(WorkflowDirs.QUANTIZED)),
        )

        self.assertEqual(plan.seed_round, 4)
        self.assertEqual(plan.next_round, 5)
        self.assertEqual(plan.target_round, 6)
        self.assertFalse(plan.needs_seed_evaluation)

    def test_build_run_plan_marks_unevaluated_seed_round_for_recheck(self) -> None:
        self._write_round(WorkflowDirs.FINETUNED, 5)
        optimizer = self._optimizer(max_rounds=5)

        plan = optimizer._build_run_plan(
            str(self.store.workflow_path(WorkflowDirs.FINETUNED)),
            str(self.store.workflow_path(WorkflowDirs.QUANTIZED)),
        )

        self.assertEqual(plan.seed_round, 5)
        self.assertEqual(plan.next_round, 6)
        self.assertEqual(plan.target_round, 5)
        self.assertTrue(plan.needs_seed_evaluation)
        self.assertFalse(plan.has_new_rounds)

    def test_latest_valid_round_skips_incomplete_round_directories(self) -> None:
        self._write_round(WorkflowDirs.FINETUNED, 4)
        self._write_round(WorkflowDirs.FINETUNED, 5, with_files=False)
        optimizer = self._optimizer(max_rounds=6)

        latest_round = optimizer._latest_valid_round(
            str(self.store.workflow_path(WorkflowDirs.FINETUNED))
        )

        self.assertEqual(latest_round, 4)


if __name__ == "__main__":
    unittest.main()
