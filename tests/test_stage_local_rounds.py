import json
import tempfile
import unittest
from pathlib import Path

from src.optimizer.workspace import WorkflowDirs, WorkspaceStore
from src.pipeline.prune_pipeline import PrunePipeline
from src.pipeline.quantize_pipeline import QuantizePipeline


def _write_workflow_round(store: WorkspaceStore, workflow_name: str, round_number: int) -> None:
    directory = store.create_round_dir(store.workflow_path(workflow_name), round_number)
    (directory / "graph.py").write_text("class Workflow: pass\n", encoding="utf-8")
    (directory / "prompt.py").write_text("PROMPT = 'x'\n", encoding="utf-8")


def _write_results(store: WorkspaceStore, workflow_name: str, rows: list[dict]) -> None:
    path = store.results_file(store.workflow_path(workflow_name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


class StageLocalRoundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name) / "workspace" / "MATH"
        self.store = WorkspaceStore(self.root)

    def test_prune_selects_best_mcts_round_and_uses_local_next_round(self) -> None:
        _write_workflow_round(self.store, WorkflowDirs.GRAPH, 5)
        _write_workflow_round(self.store, WorkflowDirs.GRAPH, 9)
        _write_workflow_round(self.store, WorkflowDirs.PRUNED, 1)
        _write_workflow_round(self.store, WorkflowDirs.PRUNED, 2)
        _write_results(
            self.store,
            WorkflowDirs.GRAPH,
            [
                {"round": 5, "score": 0.51},
                {"round": 9, "score": 0.93},
            ],
        )

        pipeline = PrunePipeline(
            workspace=str(self.root.parent),
            dataset="MATH",
            exec_llm_config={},
        )
        source = pipeline._resolve_source_workflow(str(self.store.workflow_path(WorkflowDirs.GRAPH)))

        self.assertIsNotNone(source)
        self.assertEqual(source.stage, "mcts")
        self.assertEqual(source.round_number, 9)
        self.assertEqual(source.selector, "best_score")
        self.assertEqual(self.store.next_round(self.store.workflow_path(WorkflowDirs.PRUNED)), 3)

    def test_quantize_prefers_latest_prune_round_and_falls_back_to_mcts(self) -> None:
        _write_workflow_round(self.store, WorkflowDirs.PRUNED, 1)
        _write_workflow_round(self.store, WorkflowDirs.PRUNED, 2)
        _write_workflow_round(self.store, WorkflowDirs.GRAPH, 4)
        _write_results(
            self.store,
            WorkflowDirs.GRAPH,
            [{"round": 4, "score": 0.77}],
        )

        pipeline = QuantizePipeline(
            workspace=str(self.root.parent),
            dataset="MATH",
            exec_llm_config={},
            quantize_low_model_name="low-model",
        )
        source = pipeline._resolve_source_workflow(
            str(self.store.workflow_path(WorkflowDirs.GRAPH)),
            str(self.store.workflow_path(WorkflowDirs.PRUNED)),
        )
        self.assertIsNotNone(source)
        self.assertEqual(source.stage, "prune")
        self.assertEqual(source.round_number, 2)
        self.assertEqual(source.selector, "latest_round")

        fallback_root = Path(self.tempdir.name) / "workspace" / "GSM8K"
        fallback_store = WorkspaceStore(fallback_root)
        _write_workflow_round(fallback_store, WorkflowDirs.GRAPH, 7)
        _write_results(
            fallback_store,
            WorkflowDirs.GRAPH,
            [{"round": 7, "score": 0.81}],
        )
        fallback_pipeline = QuantizePipeline(
            workspace=str(fallback_root.parent),
            dataset="GSM8K",
            exec_llm_config={},
            quantize_low_model_name="low-model",
        )
        fallback_source = fallback_pipeline._resolve_source_workflow(
            str(fallback_store.workflow_path(WorkflowDirs.GRAPH)),
            str(fallback_store.workflow_path(WorkflowDirs.PRUNED)),
        )
        self.assertIsNotNone(fallback_source)
        self.assertEqual(fallback_source.stage, "mcts")
        self.assertEqual(fallback_source.round_number, 7)
        self.assertEqual(fallback_source.selector, "mcts_best_score_fallback")


if __name__ == "__main__":
    unittest.main()
