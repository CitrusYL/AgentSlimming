from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel

from src.core.formatter import BaseFormatter, CodeFormatter, TextFormatter, XmlFormatter
from src.core.response_models import (
    AnswerGenerateResponseModel,
    CustomResponseModel,
    ScEnsembleResponseModel,
    TestResponseModel,
)

FormatterKind = Literal["none", "text", "xml", "code"]


@dataclass(frozen=True)
class RuntimeNodeSpec:
    key: str
    default_prompt: str = ""
    formatter_kind: FormatterKind = "text"
    response_model: type[BaseModel] | None = None
    include_problem: bool = True

    def create_formatter(self, *, function_name: str | None = None) -> Optional[BaseFormatter]:
        if self.formatter_kind == "none":
            return None
        if self.formatter_kind == "text":
            return TextFormatter()
        if self.formatter_kind == "xml":
            if self.response_model is None:
                raise ValueError(f"Node spec {self.key} is missing response_model")
            return XmlFormatter.from_model(self.response_model)
        if self.formatter_kind == "code":
            return CodeFormatter(function_name=function_name)
        raise ValueError(f"Unsupported formatter kind: {self.formatter_kind}")


@dataclass(frozen=True)
class OperatorSpec:
    key: str
    node_class_name: str
    description: str
    constructor: str
    outputs: tuple[str, ...]
    prompt_required: bool = False

    def format_operator(self, index: int) -> str:
        prompt_note = (
            "requires a prompt_custom constant"
            if self.prompt_required
            else "uses the built-in node prompt"
        )
        outputs = ", ".join(self.outputs)
        return (
            f"{index}. {self.key} -> {self.node_class_name}\n"
            f"   Description: {self.description}\n"
            f"   Constructor: {self.constructor}\n"
            f"   Outputs: {outputs}\n"
            f"   Prompt: {prompt_note}"
        )


@dataclass(frozen=True)
class NodeDefinition:
    key: str
    node_class_name: str
    module_name: str
    module_path: str
    runtime_spec: RuntimeNodeSpec
    operator_spec: OperatorSpec
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def import_path(self) -> str:
        return f"{self.module_path}.{self.module_name}"

    @property
    def import_statement(self) -> str:
        return f"from {self.import_path} import {self.node_class_name}"

    def load_class(self) -> type[Any]:
        module = import_module(self.import_path)
        return getattr(module, self.node_class_name)


def _node_definition(
    *,
    key: str,
    node_class_name: str,
    module_name: str,
    aliases: tuple[str, ...] = (),
    description: str,
    constructor: str,
    outputs: tuple[str, ...],
    prompt_required: bool = False,
    default_prompt: str = "",
    formatter_kind: FormatterKind = "text",
    response_model: type[BaseModel] | None = None,
    include_problem: bool = True,
) -> NodeDefinition:
    return NodeDefinition(
        key=key,
        node_class_name=node_class_name,
        module_name=module_name,
        module_path="src.core.nodes",
        aliases=aliases,
        runtime_spec=RuntimeNodeSpec(
            key=key,
            default_prompt=default_prompt,
            formatter_kind=formatter_kind,
            response_model=response_model,
            include_problem=include_problem,
        ),
        operator_spec=OperatorSpec(
            key=key,
            node_class_name=node_class_name,
            description=description,
            constructor=constructor,
            outputs=outputs,
            prompt_required=prompt_required,
        ),
    )


NODE_DEFINITIONS: dict[str, NodeDefinition] = {
    "Input": _node_definition(
        key="Input",
        node_class_name="InputNode",
        module_name="input_node",
        formatter_kind="none",
        include_problem=False,
        description="Workflow entry node. Use exactly one Input node with node_id=\"Input\".",
        constructor=(
            "InputNode(node_id=\"Input\", node_llm_config=self.llm_config, "
            "description=\"Graph input entry\", count_towards_cost=True)"
        ),
        outputs=("success",),
    ),

    "Custom": _node_definition(
        key="Custom",
        node_class_name="CustomNode",
        module_name="custom_node",
        default_prompt="""
            Information: {input}

            Based on the provided information above, provide the corresponding output directly.
            Return only the answer text with no XML, JSON, or field wrappers.
        """.strip(),
        formatter_kind="text",
        response_model=CustomResponseModel,
        description="Flexible LLM reasoning node driven by a custom prompt constant.",
        constructor=(
            "CustomNode(node_id=\"...\", node_prompt=prompt_custom.<PROMPT_NAME>, "
            "node_llm_config=self.llm_config, description=\"...\", count_towards_cost=True)"
        ),
        outputs=("output", "llm_usage"),
        prompt_required=True,
    ),

    "AnswerGenerate": _node_definition(
        key="AnswerGenerate",
        node_class_name="AnswerGenerateNode",
        module_name="answer_generate_node",
        default_prompt="""
            Think step by step and solve the problem.
            1. In the "thought" field, explain your thinking process in detail.
            2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.

            Your task: {input}
        """.strip(),
        formatter_kind="xml",
        response_model=AnswerGenerateResponseModel,
        description="Built-in answer generation node for direct step-by-step solving.",
        constructor=(
            "AnswerGenerateNode(node_id=\"...\", node_llm_config=self.llm_config, "
            "description=\"...\", count_towards_cost=True)"
        ),
        outputs=("output", "answer", "llm_usage"),
    ),

    "ScEnsemble": _node_definition(
        key="ScEnsemble",
        node_class_name="ScEnsembleNode",
        module_name="sc_ensemble_node",
        default_prompt="""
            Given the question described as follows: {problem}
            Several candidate solutions have been generated to address the given question. They are as follows:
            {solutions}

            Your task is to act as an expert evaluator. Carefully evaluate these solutions and identify the definitive answer that appears most frequently across them (Majority Consensus).
            Analyze the solutions to determine the most consistent and reliable outcome.
        """.strip(),
        formatter_kind="xml",
        response_model=ScEnsembleResponseModel,
        description="LLM self-consistency selector over multiple upstream candidate solutions.",
        constructor=(
            "ScEnsembleNode(node_id=\"...\", node_llm_config=self.llm_config, "
            "description=\"...\", count_towards_cost=True)"
        ),
        outputs=("output", "llm_usage"),
    ),

    "Programmer": _node_definition(
        key="Programmer",
        node_class_name="ProgramNode",
        module_name="program_node",
        aliases=("Program",),
        default_prompt="""
            You are a professional Python programmer. Your task is to write complete, self-contained code based on a given problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

            Problem: {problem}
            Resources: {input}

            Your code should:
            1. Implement the calculation steps described in the problem.
            2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
            3. `solve` function return the final calculation result.

            Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
        """.strip(),
        formatter_kind="code",
        description="Generates Python code, executes solve(), and returns the execution output.",
        constructor=(
            "ProgramNode(node_id=\"...\", node_llm_config=self.llm_config, "
            "description=\"...\", count_towards_cost=True)"
        ),
        outputs=("code", "output", "llm_usage"),
    ),

    "CustomCodeGenerate": _node_definition(
        key="CustomCodeGenerate",
        node_class_name="CustomCodeGenerateNode",
        module_name="custom_code_generate_node",
        default_prompt="""
            You are a professional Python programmer. Your task is to write complete, self-contained code based on a given problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

            Problem: {problem}
            Resources: {input}

            Your code should:
            1. Implement the calculation steps described in the problem.
            2. Define a function named `{function_name}` that performs the calculation and returns the result.
            3. Return only runnable Python code without explanations.
        """.strip(),
        formatter_kind="code",
        description="Generates Python code for code-generation benchmarks and preserves the entry point.",
        constructor=(
            "CustomCodeGenerateNode(node_id=\"...\", "
            "node_prompt=prompt_custom.<PROMPT_NAME>, "
            "node_llm_config=self.llm_config, description=\"...\", count_towards_cost=True)"
        ),
        outputs=("code", "output", "entry_point", "llm_usage"),
        prompt_required=True,
    ),

    "Test": _node_definition(
        key="Test",
        node_class_name="TestNode",
        module_name="test_node",
        default_prompt="""
            Given a problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution.
            Problem: {problem}

            Failure details:
            {error_info}

            Please provide a "reflection" field explaining the failed test cases and code solution, and a "solution" field containing a better code solution without any additional text or test cases.
        """.strip(),
        formatter_kind="xml",
        response_model=TestResponseModel,
        description="Executes upstream Python code and asks the LLM to repair it on failure.",
        constructor=(
            "TestNode(node_id=\"...\", node_llm_config=self.llm_config, "
            "description=\"...\", count_towards_cost=True)"
        ),
        outputs=("result", "solution", "output", "llm_usage"),
    ),

    "AnswerFormat": _node_definition(
        key="AnswerFormat",
        node_class_name="AnswerFormatNode",
        module_name="answer_format_node",
        aliases=("AnswerFormatter",),
        default_prompt="""
            You are an answer formatter for the {dataset_name} dataset.
            Task Context:
            {task_context}

            Format Requirements:
            {format_requirements}

            Original Answer:
            {original_answer}

            Return only the final formatted answer according to the requirements.
        """.strip(),
        formatter_kind="text",
        include_problem=True,
        description="Final node that formats the selected raw answer for the target dataset.",
        constructor=(
            "AnswerFormatNode(node_id=\"AnswerFormatter\", dataset_name=self.dataset, "
            "node_llm_config=self.llm_config, description=\"Format the final answer\", count_towards_cost=True)"
        ),
        outputs=("output", "llm_usage"),
    ),
}

NODE_CLASS_NAME_TO_KEY = {
    definition.node_class_name: key for key, definition in NODE_DEFINITIONS.items()
}

NODE_SPEC_ALIASES = {
    alias: definition.key
    for definition in NODE_DEFINITIONS.values()
    for alias in definition.aliases
}


def resolve_node_spec_name(name: str) -> str:
    return NODE_SPEC_ALIASES.get(name, name)


def get_node_definition(name: str) -> NodeDefinition:
    resolved_name = resolve_node_spec_name(name)
    try:
        return NODE_DEFINITIONS[resolved_name]
    except KeyError as exc:
        available = ", ".join(sorted(NODE_DEFINITIONS))
        raise ValueError(f"Unknown node spec: {name}. Available: {available}") from exc


def get_runtime_node_spec(name: str) -> RuntimeNodeSpec:
    return get_node_definition(name).runtime_spec


def get_operator_spec(name: str) -> OperatorSpec:
    return get_node_definition(name).operator_spec


def iter_node_definitions() -> Iterable[NodeDefinition]:
    return NODE_DEFINITIONS.values()


def known_node_class_names() -> set[str]:
    return set(NODE_CLASS_NAME_TO_KEY)


def get_node_module_name(class_name: str) -> str:
    try:
        key = NODE_CLASS_NAME_TO_KEY[class_name]
    except KeyError as exc:
        available = ", ".join(sorted(NODE_CLASS_NAME_TO_KEY))
        raise ValueError(f"Unknown node class: {class_name}. Available: {available}") from exc
    return NODE_DEFINITIONS[key].module_name


def get_node_import_path(class_name: str) -> str:
    try:
        key = NODE_CLASS_NAME_TO_KEY[class_name]
    except KeyError as exc:
        available = ", ".join(sorted(NODE_CLASS_NAME_TO_KEY))
        raise ValueError(f"Unknown node class: {class_name}. Available: {available}") from exc
    return NODE_DEFINITIONS[key].import_path


def load_node_class(name: str) -> type[Any]:
    return get_node_definition(name).load_class()


def build_node_imports(class_names: Iterable[str]) -> str:
    unique_class_names = sorted(set(class_names))
    return "\n".join(
        NODE_DEFINITIONS[NODE_CLASS_NAME_TO_KEY[class_name]].import_statement
        for class_name in unique_class_names
    )
