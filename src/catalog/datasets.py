from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from benchmarks.benchmark import BaseBenchmark


DatasetType = Literal[
    "AIME",
    "DROP",
    "GSM8K",
    "HotpotQA",
    "HumanEval",
    "LiveCode",
    "MATH",
    "MATHDEMO",
    "MBPP",
    "MusiqueAns",
]
QuestionType = Literal["qa", "math", "code"]

QA_OPERATORS = ("Custom", "AnswerGenerate", "ScEnsemble", "Input", "AnswerFormat")
MATH_OPERATORS = ("Custom", "ScEnsemble", "Programmer", "Input", "AnswerFormat")
CODE_OPERATORS = ("Custom", "CustomCodeGenerate", "ScEnsemble", "Test", "Input", "AnswerFormat")


@dataclass(frozen=True)
class DatasetConfig:
    name: DatasetType
    file_stem: str
    question_type: QuestionType
    operators: tuple[str, ...]
    benchmark_module: str
    benchmark_class_name: str


def _dataset(
    name: DatasetType,
    file_stem: str,
    question_type: QuestionType,
    operators: tuple[str, ...],
    benchmark_module: str,
    benchmark_class_name: str,
) -> DatasetConfig:
    return DatasetConfig(
        name=name,
        file_stem=file_stem,
        question_type=question_type,
        operators=operators,
        benchmark_module=benchmark_module,
        benchmark_class_name=benchmark_class_name,
    )


DATASET_CONFIGS: dict[DatasetType, DatasetConfig] = {
    "DROP": _dataset("DROP", "drop", "qa", QA_OPERATORS, "benchmarks.drop", "DROPBenchmark"),
    "HotpotQA": _dataset(
        "HotpotQA",
        "hotpotqa",
        "qa",
        QA_OPERATORS,
        "benchmarks.hotpotqa",
        "HotpotQABenchmark",
    ),
    "MusiqueAns": _dataset(
        "MusiqueAns",
        "musiqueans",
        "qa",
        QA_OPERATORS,
        "benchmarks.musique",
        "MusiqueAnsBenchmark",
    ),
    "GSM8K": _dataset("GSM8K", "gsm8k", "math", MATH_OPERATORS, "benchmarks.gsm8k", "GSM8KBenchmark"),
    "MATH": _dataset("MATH", "math", "math", MATH_OPERATORS, "benchmarks.math", "MATHBenchmark"),
    "MATHDEMO": _dataset(
        "MATHDEMO",
        "math",
        "math",
        MATH_OPERATORS,
        "benchmarks.math",
        "MATHBenchmark",
    ),
    "AIME": _dataset("AIME", "aime", "math", MATH_OPERATORS, "benchmarks.aime", "AIMEBenchmark"),
    "HumanEval": _dataset(
        "HumanEval",
        "humaneval",
        "code",
        CODE_OPERATORS,
        "benchmarks.humaneval",
        "HumanEvalBenchmark",
    ),
    "MBPP": _dataset("MBPP", "mbpp", "code", CODE_OPERATORS, "benchmarks.mbpp", "MBPPBenchmark"),
    "LiveCode": _dataset(
        "LiveCode",
        "livecode",
        "code",
        CODE_OPERATORS,
        "benchmarks.livecode",
        "LiveCodeBench",
    ),
}


def get_dataset_config(dataset: DatasetType | str) -> DatasetConfig:
    if dataset not in DATASET_CONFIGS:
        supported = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {supported}")
    return DATASET_CONFIGS[dataset]


def get_supported_dataset_names() -> tuple[str, ...]:
    return tuple(DATASET_CONFIGS)


@lru_cache(maxsize=None)
def load_benchmark_class(dataset: DatasetType | str) -> type["BaseBenchmark"]:
    config = get_dataset_config(dataset)
    module = import_module(config.benchmark_module)
    return getattr(module, config.benchmark_class_name)


def dataset_file_path(
    dataset: DatasetType | str,
    split: Literal["validate", "test"],
    root: str | Path = "data/datasets",
) -> Path:
    config = get_dataset_config(dataset)
    return Path(root) / f"{config.file_stem}_{split}.jsonl"
