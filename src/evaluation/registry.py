from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.catalog.datasets import DatasetType, get_dataset_config, load_benchmark_class

if TYPE_CHECKING:
    from benchmarks.benchmark import BaseBenchmark


@dataclass(frozen=True)
class EvaluationSpec:
    dataset: DatasetType
    benchmark_class: type["BaseBenchmark"]


def get_evaluation_spec(dataset: DatasetType | str) -> EvaluationSpec:
    config = get_dataset_config(dataset)
    return EvaluationSpec(
        dataset=config.name,
        benchmark_class=load_benchmark_class(config.name),
    )
