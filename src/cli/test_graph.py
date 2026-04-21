import argparse
import asyncio
import importlib
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Type

from src.catalog.datasets import get_supported_dataset_names
from src.evaluation.evaluator import Evaluator
from src.utils.async_llm import LLMsConfig
from src.utils.common import write_json_file
from src.utils.logs import logger

def _module_name_from_file_path(file_path: str) -> str:
    abs_path = Path(file_path).resolve()
    candidates = [Path.cwd(), *[Path(p) for p in sys.path if isinstance(p, str) and p]]

    for base in candidates:
        try:
            rel = abs_path.relative_to(base.resolve())
        except Exception:
            continue

        if str(rel) == ".":
            continue

        module_name = ".".join(rel.with_suffix("").parts)
        if module_name.endswith(".__init__"):
            module_name = module_name[: -len(".__init__")]
        return module_name

    raise ValueError(f"Cannot resolve module path for graph file: {file_path}")


def load_graph_class(graph_ref: str) -> Type:
    """Load Workflow class from a python module path or filesystem path."""
    file_path = None
    module_name = None

    graph_path = Path(graph_ref)
    if graph_path.exists():
        file_path = graph_ref
        if graph_path.is_dir():
            graph_file_path = graph_path / "graph.py"
            if not graph_file_path.exists():
                raise FileNotFoundError(f"graph.py not found in directory: {file_path}")
            file_path = str(graph_file_path)

        module_name = _module_name_from_file_path(file_path)
    else:
        module_name = graph_ref
        if module_name.endswith(".py"):
            module_name = module_name[:-3]
        module_name = module_name.replace("/", ".").replace("\\", ".")

    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        if file_path:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load graph module from {file_path}") from e
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        else:
            raise

    if not hasattr(module, "Workflow"):
        raise ImportError(f"Class 'Workflow' not found in {module_name}")

    return getattr(module, "Workflow")

async def main():
    parser = argparse.ArgumentParser(description="Test a specified graph on the test set.")
    parser.add_argument("--graph", type=str, required=True, help="Python module path (preferred) or path to graph.py / its directory.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(get_supported_dataset_names()),
        default="MATH",
        help="Dataset name.",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Path to the test data file. If not provided, defaults to standard path.")
    parser.add_argument("--sample", type=int, default=None, help="Evaluate the first N samples from the selected split.")
    parser.add_argument("--model", type=str, required=True, help="Execution model name from config/config.yaml.")
    parser.add_argument("--config", type=str, default=None, help="Path to LLM config YAML.")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save results.")

    args = parser.parse_args()

    # Load Graph Class
    try:
        GraphClass = load_graph_class(args.graph)
        logger.info(f"Successfully loaded graph from {args.graph}")
    except Exception as e:
        logger.error(f"Failed to load graph: {e}")
        return

    # Configure LLM
    models_config = LLMsConfig.default(args.config)
    exec_llm_config = models_config.get(args.model)
    evaluator = Evaluator(eval_path=args.output_dir)

    logger.info("Starting evaluation...")

    try:
        score, avg_cost, total_cost = await evaluator.graph_evaluate(
            args.dataset,
            GraphClass,
            {"dataset": args.dataset, "llm_config": exec_llm_config},
            path=args.output_dir,
            is_test=True,
            sample_size=args.sample,
            data_path=args.data_path,
        )

        logger.info(f"Test Complete.")
        logger.info(f"Score: {score}")
        logger.info(f"Avg Cost: {avg_cost}")
        logger.info(f"Total Cost: {total_cost}")

        # Save results to JSON
        current_time = datetime.now().isoformat()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        result_data = {
            "score": score,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
            "time": current_time,
        }

        json_path = Path(args.output_dir) / f"{args.dataset}_test_{timestamp_str}.json"
        write_json_file(json_path, result_data)

        logger.info(f"Test summary saved to {json_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
