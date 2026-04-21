import tempfile
import unittest
from pathlib import Path

from src.optimizer.manifest_utils import WorkflowRoundRef, read_round_manifest, write_round_manifest
from src.optimizer.workspace import WorkflowDirs, WorkspaceStore


class ManifestUtilsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name) / "workspace" / "MATH"
        self.store = WorkspaceStore(self.root)

    def test_next_round_is_stage_local(self) -> None:
        pruned_root = self.store.workflow_path(WorkflowDirs.PRUNED)
        self.assertEqual(self.store.next_round(pruned_root), 1)
        self.store.create_round_dir(pruned_root, 1)
        self.store.create_round_dir(pruned_root, 2)
        self.assertEqual(self.store.next_round(pruned_root), 3)

    def test_round_manifest_records_full_lineage_chain(self) -> None:
        write_round_manifest(
            self.store,
            WorkflowDirs.GRAPH,
            9,
            source_selector="repository_seed",
            metrics={"score": 0.91},
            summary={"modification": "initial graph"},
            artifacts=["graph.py", "prompt.py", "manifest.json"],
        )
        write_round_manifest(
            self.store,
            WorkflowDirs.PRUNED,
            1,
            parent=WorkflowRoundRef("mcts", WorkflowDirs.GRAPH, 9, "best_score"),
            source_selector="best_score",
            metrics={"score": 0.88},
            summary={"pruned_nodes": ["Reasoner"]},
            artifacts=["graph.py", "prompt.py", "prune_detail_info.json", "manifest.json"],
        )
        write_round_manifest(
            self.store,
            WorkflowDirs.QUANTIZED,
            1,
            parent=WorkflowRoundRef("prune", WorkflowDirs.PRUNED, 1, "latest_round"),
            source_selector="latest_round",
            metrics={"score": 0.86},
            summary={"quantized_nodes": ["Reasoner"]},
            artifacts=["graph.py", "prompt.py", "quantize_detail_info.json", "manifest.json"],
        )

        manifest = read_round_manifest(self.store, WorkflowDirs.QUANTIZED, 1)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest["parent"]["stage"], "prune")
        self.assertEqual(manifest["parent"]["round"], 1)
        self.assertEqual(
            manifest["lineage"],
            [
                {
                    "stage": "prune",
                    "workflow": WorkflowDirs.PRUNED,
                    "round": 1,
                    "path": f"{WorkflowDirs.PRUNED}/round_1",
                },
                {
                    "stage": "mcts",
                    "workflow": WorkflowDirs.GRAPH,
                    "round": 9,
                    "path": f"{WorkflowDirs.GRAPH}/round_9",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
