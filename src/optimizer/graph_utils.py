import re
import textwrap
from typing import Any

from src.core.nodes.catalog import build_node_imports, known_node_class_names
from src.prompts.optimize_prompt import (
    WORKFLOW_INPUT,
    WORKFLOW_OPTIMIZE_PROMPT,
    WORKFLOW_PROMPT_USE,
    WORKFLOW_TEMPLATE,
)
from src.prompts.operator_catalog import format_operator_specs as render_operator_specs
from src.optimizer.workspace import WorkspaceStore
from src.utils.logs import logger


def _extract_used_nodes_from_graph(graph_code: str) -> list[str]:
    pattern = r"\b([A-Z][A-Za-z0-9_]+)\b"
    matches = re.findall(pattern, graph_code)
    known_classes = known_node_class_names()
    return sorted({node for node in matches if node in known_classes})


def build_dynamic_imports(used_nodes: list[str]) -> str:
    return build_node_imports(used_nodes)


def clean_prompt_content(prompt: Any) -> str:
    if isinstance(prompt, str):
        content = prompt
    elif isinstance(prompt, dict):
        content = _prompt_text_from_dict(prompt)
    else:
        content = str(prompt)

    content = re.sub(r"class\s+\w+\s*[:].*?(?=^\S|\Z)", "", content, flags=re.S | re.M)
    return content.strip()


def _prompt_text_from_dict(prompt: dict[str, Any]) -> str:
    for key in ("content", "text", "prompt"):
        value = prompt.get(key)
        if isinstance(value, str):
            return value
    return str(prompt)


def format_graph_body(graph_code: str) -> str:
    body = _strip_code_fence(graph_code).strip()
    if not body:
        raise ValueError("Generated graph body is empty")
    forbidden = (
        r"\b("
        r"class\s+Workflow|def\s+_build_graph|def\s+__init__|"
        r"async\s+def\s+__call__|GraphExecutor"
        r")\b"
    )
    if re.search(forbidden, body) or re.search(r"(?m)^\s*(from|import)\s+", body):
        raise ValueError("Generated graph must contain only the Workflow._build_graph body")
    stale_fields = {
        "source_key": "Edge.key",
        "target_key": "Edge.key",
        "input_key": "Edge.key",
        "node_output": "node outputs are implicit and should not be declared",
        "node_role": "node type is determined by the class name",
    }
    for stale_name, replacement in stale_fields.items():
        if re.search(rf"\b{re.escape(stale_name)}\s*=", body):
            raise ValueError(
                f"Generated graph uses unsupported field '{stale_name}'. "
                f"Use {replacement} instead."
            )
    return textwrap.indent(textwrap.dedent(body).strip(), "        ")


def _strip_code_fence(code: str) -> str:
    code = code.strip()
    if not code.startswith("```"):
        return code

    lines = code.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


class GraphUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.store = WorkspaceStore(root_path)

    def create_round_directory(self, graph_path: str, round_number: int) -> str:
        return str(self.store.create_round_dir(graph_path, round_number))

    def create_iter_directory(self, graph_path: str, round_number: int, iter_number: int) -> str:
        return str(self.store.create_iter_dir(graph_path, round_number, iter_number))

    def latest_round(self, workflows_path: str) -> int | None:
        return self.store.latest_round(workflows_path)

    def has_graph_files(self, workflows_path: str, round_number: int, iter_number: int | None = None) -> bool:
        return self.store.has_graph_files(workflows_path, round_number, iter_number)

    def load_graph(self, round_number: int, workflows_path: str, iter_number: int | None = None):
        if not self.store.has_graph_files(workflows_path, round_number, iter_number):
            directory = self.store.round_dir(workflows_path, round_number, iter_number)
            raise FileNotFoundError(f"Workflow graph/prompt files not found: {directory}")

        graph_module_name = self.store.module_name(
            workflows_path,
            round_number,
            iter_number,
            module="graph",
        )
        try:
            graph_module = __import__(graph_module_name, fromlist=[""])
            graph_class = getattr(graph_module, "Workflow")
            return graph_class
        except ImportError as e:
            logger.error(f"Error loading graph for round {round_number} (iter {iter_number}): {e}")
            raise

    def read_graph_files(self, round_number: int, workflows_path: str, iter_number: int | None = None):
        try:
            prompt_content, graph_content = self.store.read_workflow_files(
                workflows_path,
                round_number,
                iter_number,
            )
        except FileNotFoundError as e:
            logger.error(f"Error: File not found for round {round_number}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt for round {round_number}: {e}")
            raise
        return prompt_content, graph_content

    def extract_workflow_source(self, graph_code: str) -> str:
        build_graph = re.search(
            (
                r"    def _build_graph\(self\).*?:\n"
                r"(?P<body>.*?)(?=\n    (?:async\s+def|def)\s|\nclass\s|\Z)"
            ),
            graph_code,
            flags=re.DOTALL,
        )
        if build_graph:
            return textwrap.dedent(build_graph.group("body")).strip()

        match = re.search(r"class Workflow:.+", graph_code, flags=re.DOTALL)
        return match.group(0) if match else ""

    def format_operator_specs(self, operators: list[str]) -> str:
        return render_operator_specs(operators)

    def create_graph_optimize_prompt(
        self,
        experience: str,
        score: float,
        graph: str,
        prompt: str,
        operator_spec: str,
        type: str,
        log_data: str,
    ) -> str:
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

    def write_graph_files(self, directory: str, response: dict) -> None:
        graph_code = response["graph"]
        prompt_obj = response["prompt"]
        used_nodes = _extract_used_nodes_from_graph(graph_code)
        dynamic_imports = build_dynamic_imports(used_nodes)
        graph = WORKFLOW_TEMPLATE.format(
            graph=format_graph_body(graph_code),
            dynamic_imports=dynamic_imports,
        )

        self.store.write_workflow_files(directory, graph, clean_prompt_content(prompt_obj))

    def write_workflow_files(self, directory: str, graph: str, prompt: str) -> None:
        self.store.write_workflow_files(directory, graph, prompt)

    def write_text(self, directory: str, filename: str, content: str) -> None:
        self.store.write_text(directory, filename, content)

    def write_json(self, directory: str, filename: str, data) -> None:
        self.store.write_json(directory, filename, data)

    def append_jsonl(self, path: str, data: dict) -> None:
        self.store.append_jsonl(path, data)
