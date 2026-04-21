import argparse
from typing import Sequence

from src.catalog.datasets import get_dataset_config, get_supported_dataset_names
from src.pipeline.finetune_pipeline import FinetuneOptimizer
from src.pipeline.mcts_pipeline import MCTSPipeline
from src.pipeline.prune_pipeline import PrunePipeline
from src.pipeline.quantize_pipeline import QuantizePipeline
from src.utils.async_llm import LLMsConfig

PIPELINE_CHOICES = ("mcts", "prune", "quantize", "finetune")
DEFAULT_PIPELINES = ("mcts", "prune", "quantize")
DEFAULT_PIPELINES_TEXT = ",".join(DEFAULT_PIPELINES)
OPTIMIZER_PIPELINES = {"mcts", "finetune"}


def parse_pipelines(value: str) -> tuple[str, ...]:
    pipelines = tuple(item.strip() for item in value.split(",") if item.strip())
    if not pipelines:
        raise argparse.ArgumentTypeError("At least one pipeline is required.")

    unknown = sorted(set(pipelines) - set(PIPELINE_CHOICES))
    if unknown:
        choices = ", ".join(PIPELINE_CHOICES)
        raise argparse.ArgumentTypeError(f"Unknown pipeline(s): {', '.join(unknown)}. Choices: {choices}.")

    duplicates = sorted({pipeline for pipeline in pipelines if pipelines.count(pipeline) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(f"Duplicate pipeline(s): {', '.join(duplicates)}.")

    return pipelines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GraphFlow optimizer runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(get_supported_dataset_names()),
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--pipelines",
        type=parse_pipelines,
        default=DEFAULT_PIPELINES_TEXT,
        metavar="LIST",
        help=f"Comma-separated pipeline list. Choices: {', '.join(PIPELINE_CHOICES)}",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace root. Defaults to workspace.",
    )
    parser.add_argument("--config", default=None, help="Path to LLM config YAML")

    parser.add_argument("--sample", type=int, default=4, help="Sample count per optimization round")
    parser.add_argument(
        "--initial_round",
        type=int,
        default=0,
        help=(
            "Resume/source round. For mcts this is the starting local round. "
            "For prune and quantize this selects the source round from the previous stage. "
            "0 means pipeline default."
        ),
    )
    parser.add_argument("--max_rounds", type=int, default=20, help="Maximum optimization rounds")
    parser.add_argument("--check_convergence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation_rounds", type=int, default=1, help="Evaluation repetitions per round")

    parser.add_argument("--opt_model_name", default=None, help="Optimizer model name from config/config.yaml")
    parser.add_argument("--exec_model_name", default=None, help="Execution model name from config/config.yaml")
    parser.add_argument("--quantize_low_model_name", default=None, help="Low-cost quantization model name from config/config.yaml")

    parser.add_argument("--mcts_eval_samples", type=int, default=None, help="MCTS evaluation sample size")
    parser.add_argument("--prune_eval_samples", type=int, default=None, help="Pruning evaluation sample size")
    parser.add_argument("--quantize_eval_samples", type=int, default=None, help="Quantization evaluation sample size")
    parser.add_argument("--finetune_eval_samples", type=int, default=None, help="Finetune evaluation sample size")
    parser.add_argument("--traverse_sample_size", type=int, default=50, help="Traverse evaluation sample size")

    parser.add_argument("--prune_threshold", type=float, default=0.95, help="Pruning score threshold")
    parser.add_argument("--quantize_threshold", type=float, default=0.95, help="Quantization score threshold")
    parser.add_argument("--quantize_rate", type=float, default=0.4, help="Quantization rate")

    args = parser.parse_args(argv)
    if isinstance(args.pipelines, str):
        args.pipelines = parse_pipelines(args.pipelines)
    if args.initial_round < 0:
        parser.error("--initial_round must be >= 0")
    if args.exec_model_name is None:
        parser.error("--exec_model_name is required")
    if any(pipeline in OPTIMIZER_PIPELINES for pipeline in args.pipelines) and args.opt_model_name is None:
        parser.error("--opt_model_name is required for mcts and finetune")
    if "quantize" in args.pipelines and args.quantize_low_model_name is None:
        parser.error("--quantize_low_model_name is required for quantize")
    return args


def workspace_for(args: argparse.Namespace, pipeline: str) -> str:
    if args.workspace:
        return args.workspace
    return "workspace"


def source_round(args: argparse.Namespace) -> int | None:
    return args.initial_round or None


def optimizer_round(args: argparse.Namespace) -> int:
    return args.initial_round or 1


def build_pipeline(
    pipeline: str,
    args: argparse.Namespace,
    experiment,
    opt_llm_config: dict | None,
    exec_llm_config: dict,
):
    workspace = workspace_for(args, pipeline)
    dataset = args.dataset

    if pipeline == "mcts":
        return MCTSPipeline(
            workspace=workspace,
            dataset=dataset,
            question_type=experiment.question_type,
            opt_llm_config=opt_llm_config,
            exec_llm_config=exec_llm_config,
            operators=list(experiment.operators),
            sample=args.sample,
            check_convergence=args.check_convergence,
            initial_round=optimizer_round(args),
            max_rounds=args.max_rounds,
            validation_rounds=args.validation_rounds,
            mcts_eval_samples=args.mcts_eval_samples,
        )

    if pipeline == "prune":
        return PrunePipeline(
            workspace=workspace,
            dataset=dataset,
            exec_llm_config=exec_llm_config,
            sample=args.sample,
            validation_rounds=args.validation_rounds,
            prune_threshold=args.prune_threshold,
            traverse_sample_size=args.traverse_sample_size,
            prune_eval_samples=args.prune_eval_samples,
            initial_round=source_round(args),
        )

    if pipeline == "quantize":
        return QuantizePipeline(
            workspace=workspace,
            dataset=dataset,
            exec_llm_config=exec_llm_config,
            validation_rounds=args.validation_rounds,
            traverse_sample_size=args.traverse_sample_size,
            quantize_eval_samples=args.quantize_eval_samples,
            quantize_rate=args.quantize_rate,
            quantize_threshold=args.quantize_threshold,
            quantize_low_model_name=args.quantize_low_model_name,
            initial_round=source_round(args),
        )

    if pipeline == "finetune":
        return FinetuneOptimizer(
            workspace=workspace,
            dataset=dataset,
            question_type=experiment.question_type,
            opt_llm_config=opt_llm_config,
            exec_llm_config=exec_llm_config,
            operators=list(experiment.operators),
            sample=args.sample,
            validation_rounds=args.validation_rounds,
            max_rounds=args.max_rounds,
            sample_size=args.finetune_eval_samples,
        )

    raise ValueError(f"Unsupported pipeline: {pipeline}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    experiment = get_dataset_config(args.dataset)
    models_config = LLMsConfig.default(args.config)
    exec_llm_config = models_config.get(args.exec_model_name)

    opt_llm_config = None
    if any(pipeline in OPTIMIZER_PIPELINES for pipeline in args.pipelines):
        opt_llm_config = models_config.get(args.opt_model_name)

    for pipeline_name in args.pipelines:
        pipeline = build_pipeline(pipeline_name, args, experiment, opt_llm_config, exec_llm_config)
        pipeline.run()


if __name__ == "__main__":
    main()
