import json
import re
from pathlib import Path
from typing import Any, Optional

from pydantic_core import to_jsonable_python


class WorkflowDirs:
    GRAPH = "workflows"
    PRUNED = "workflows_pruned"
    QUANTIZED = "workflows_quantized"
    FINETUNED = "workflows_finetuned"


class WorkspaceStore:
    """Path and file IO helpers for optimizer workspace artifacts."""

    ROUND_PATTERN = re.compile(r"^round_(\d+)$")

    def __init__(self, root_path: str | Path):
        self.root_path = Path(root_path)

    def workflow_path(self, name: str = WorkflowDirs.GRAPH) -> Path:
        return self.root_path / name

    def evaluation_path(self, pipeline_name: str) -> Path:
        return self.root_path / f"{pipeline_name}_evaluations"

    def next_round(self, workflow_path: str | Path) -> int:
        latest_round = self.latest_round(workflow_path)
        if latest_round is None:
            return 1
        return latest_round + 1

    def round_dir(
        self,
        workflow_path: str | Path,
        round_number: int,
        iter_number: Optional[int] = None,
    ) -> Path:
        name = (
            f"round_{round_number}_iter_{iter_number}"
            if iter_number is not None
            else f"round_{round_number}"
        )
        return Path(workflow_path) / name

    def create_round_dir(self, workflow_path: str | Path, round_number: int) -> Path:
        return self._mkdir(self.round_dir(workflow_path, round_number))

    def create_iter_dir(
        self,
        workflow_path: str | Path,
        round_number: int,
        iter_number: int,
    ) -> Path:
        return self._mkdir(self.round_dir(workflow_path, round_number, iter_number))

    def latest_round(self, workflow_path: str | Path) -> Optional[int]:
        root = Path(workflow_path)
        if not root.is_dir():
            return None

        rounds = []
        for path in root.iterdir():
            if not path.is_dir():
                continue
            match = self.ROUND_PATTERN.match(path.name)
            if match:
                rounds.append(int(match.group(1)))
        return max(rounds) if rounds else None

    def has_graph_files(
        self,
        workflow_path: str | Path,
        round_number: int,
        iter_number: Optional[int] = None,
    ) -> bool:
        directory = self.round_dir(workflow_path, round_number, iter_number)
        return (directory / "graph.py").is_file() and (directory / "prompt.py").is_file()

    def read_workflow_files(
        self,
        workflow_path: str | Path,
        round_number: int,
        iter_number: Optional[int] = None,
    ) -> tuple[str, str]:
        directory = self.round_dir(workflow_path, round_number, iter_number)
        return (
            (directory / "prompt.py").read_text(encoding="utf-8"),
            (directory / "graph.py").read_text(encoding="utf-8"),
        )

    def write_workflow_files(
        self,
        directory: str | Path,
        graph: str,
        prompt: str,
    ) -> None:
        directory = self._mkdir(directory)
        (directory / "graph.py").write_text(graph, encoding="utf-8")
        (directory / "prompt.py").write_text(prompt, encoding="utf-8")

    def write_text(self, directory: str | Path, filename: str, content: str) -> None:
        directory = self._mkdir(directory)
        (directory / filename).write_text(content, encoding="utf-8")

    def write_json(self, directory: str | Path, filename: str, data: Any) -> None:
        directory = self._mkdir(directory)
        with (directory / filename).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False, default=to_jsonable_python)

    def append_jsonl(self, path: str | Path, data: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, default=to_jsonable_python) + "\n")

    def results_file(self, workflow_path: str | Path) -> Path:
        return Path(workflow_path) / "results.json"

    def log_file(self, workflow_path: str | Path, round_number: int) -> Path:
        return self.round_dir(workflow_path, round_number) / "log.json"

    def module_name(
        self,
        workflow_path: str | Path,
        round_number: int,
        iter_number: Optional[int] = None,
        module: str = "graph",
    ) -> str:
        directory = self.round_dir(workflow_path, round_number, iter_number)
        try:
            relative = directory.resolve().relative_to(Path.cwd().resolve())
        except ValueError as exc:
            raise ValueError(f"Workflow path must be inside the current project: {directory}") from exc
        return ".".join((*relative.parts, module))

    @staticmethod
    def _mkdir(path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
