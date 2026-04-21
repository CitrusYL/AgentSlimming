# Demo Workspace And Evaluation

This page covers the tracked `MATHDEMO` workspace and standalone workflow evaluation. Installation/configuration live in [setup.md](setup.md), and pipeline flags live in [running.md](running.md).

## Demo Workspace

The repository includes a tracked prune and quantize demo seed at:

```text
workspace/MATHDEMO/workflows/round_2/graph.py
workspace/MATHDEMO/workflows/round_2/prompt.py
```

This lives alongside the minimal format example in the same tracked demo dataset:

- `workspace/MATHDEMO/workflows/round_1` is the simplest round-1 example
- `workspace/MATHDEMO/workflows/round_2` is a runnable workflow seed intended to skip MCTS and demonstrate pruning and quantization directly

`MATHDEMO` is a repository-local alias of the real `MATH` benchmark. It uses the same evaluator and dataset files, but keeps the tracked demo workspace distinct from experiment directories under `workspace/MATH`.

Example wrappers live under `src/cli/examples/`:

- `src/cli/examples/math_demo_prune.sh`
- `src/cli/examples/math_demo_quantize.sh`

Run them from the repository root:

```bash
EXEC_MODEL=<executor_model> bash src/cli/examples/math_demo_prune.sh
EXEC_MODEL=<executor_model> LOW_MODEL=<low_cost_model> bash src/cli/examples/math_demo_quantize.sh
```

Both wrappers default to:

- `WORKSPACE=workspace`
- `DATASET=MATHDEMO`
- `INITIAL_ROUND=2`

Optional evaluation sample controls:

- `PRUNE_EVAL_SAMPLES` for `math_demo_prune.sh`
- `QUANTIZE_EVAL_SAMPLES` for `math_demo_quantize.sh`

## Testing A Workflow

Evaluate an existing workflow graph on a test split:

```bash
python -m src.cli.test_graph \
  --config config/config.yaml \
  --graph workspace.MATH.workflows.round_1.graph \
  --dataset MATH \
  --data_path data/datasets/math_test.jsonl \
  --model gpt-4.1-mini \
  --sample 20 \
  --output_dir test_results
```

`--graph` accepts either a module path or a filesystem path to `graph.py` or its containing directory.

For code tasks, the workflow runtime passes `entry_point` through the executor when the benchmark provides it. `BaseWorkflow` and `GraphExecutor` handle that payload; `graph.py` does not need custom call logic.

## Safety Notes

Code-generation benchmarks execute model-produced Python code through the local runner. Do not run untrusted workflows outside a restricted environment.
