import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.optimizer.workspace import WorkspaceStore, WorkflowDirs
from src.utils.common import read_json_file


WORKFLOW_STAGES = {
    WorkflowDirs.GRAPH: "mcts",
    WorkflowDirs.PRUNED: "prune",
    WorkflowDirs.QUANTIZED: "quantize",
    WorkflowDirs.FINETUNED: "finetune",
}


@dataclass(frozen=True)
class WorkflowRoundRef:
    stage: str
    workflow_name: str
    round_number: int
    selector: str | None = None

    def to_manifest_ref(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "workflow": self.workflow_name,
            "round": self.round_number,
            "path": f"{self.workflow_name}/round_{self.round_number}",
        }


def stage_for_workflow(workflow_name: str) -> str:
    try:
        return WORKFLOW_STAGES[workflow_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported workflow name: {workflow_name}") from exc


def read_round_manifest(
    store: WorkspaceStore,
    workflow_name: str,
    round_number: int,
) -> dict[str, Any] | None:
    manifest_path = store.round_dir(store.workflow_path(workflow_name), round_number) / "manifest.json"
    if not manifest_path.exists():
        return None
    return read_json_file(str(manifest_path), encoding="utf-8")


def write_round_manifest(
    store: WorkspaceStore,
    workflow_name: str,
    round_number: int,
    *,
    parent: WorkflowRoundRef | None = None,
    source_selector: str | None = None,
    metrics: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    artifacts: list[str] | None = None,
) -> dict[str, Any]:
    parent_payload = parent.to_manifest_ref() if parent is not None else None
    manifest = {
        "stage": stage_for_workflow(workflow_name),
        "workflow": workflow_name,
        "round": round_number,
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "parent": parent_payload,
        "lineage": _build_lineage(store, parent_payload),
        "source_selector": source_selector,
        "metrics": metrics or {},
        "summary": summary or {},
        "artifacts": artifacts or [],
    }
    store.write_json(
        store.round_dir(store.workflow_path(workflow_name), round_number),
        "manifest.json",
        manifest,
    )
    return manifest


def _build_lineage(store: WorkspaceStore, parent_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if parent_payload is None:
        return []

    parent_manifest = read_round_manifest(
        store,
        parent_payload["workflow"],
        int(parent_payload["round"]),
    )
    if not parent_manifest:
        return [parent_payload]

    lineage = parent_manifest.get("lineage", [])
    if isinstance(lineage, list):
        return [parent_payload, *lineage]
    return [parent_payload]
