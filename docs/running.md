# Running Pipelines

This page covers the CLI entrypoints and stage behavior. Workflow artifact layout and node-extension details live in [workflows.md](workflows.md).

The main CLI is:

```bash
python -m src.cli.run
```

Supported `--dataset` values come directly from `src/catalog/datasets.py`. That catalog is also used by the evaluator, so CLI choices and benchmark dispatch stay aligned.

## Important Constraints

- `--opt_model_name` is required when `--pipelines` includes `mcts` or `finetune`.
- `--quantize_low_model_name` is required when `--pipelines` includes `quantize`.
- The default pipeline chain is `mcts,prune,quantize`. `finetune` is available but is not part of the default chain.

## Pipeline Semantics

- `mcts`: graph/prompt search over executable workflows
- `prune`: multi-agent workflow pruning, not neural network pruning
- `quantize`: swapping selected workflow nodes to a cheaper model, not true weight quantization
- `finetune`: another MCTS-like search stage starting from quantized workflows, not base-model finetuning

All stages now use stage-local `round_<n>` directories. Cross-stage lineage is tracked in each round's `manifest.json`, not in the directory name.

For `finetune`, `--max_rounds` is the target final round number in `workflows_finetuned/`. If `round_4` already exists, the next generated round is `round_5`; the seed round is reused or re-evaluated, but it is not overwritten.

## Example Commands

Run one pipeline:

```bash
python -m src.cli.run \
  --config config/config.yaml \
  --dataset MATH \
  --pipelines mcts \
  --workspace workspace \
  --max_rounds 10 \
  --sample 4 \
  --opt_model_name gpt-4.1 \
  --exec_model_name gpt-4.1-mini
```

`mcts` starts from `round_1` by default. `prune` chooses its source round from `--initial_round` if provided; otherwise it selects the best-scoring available MCTS workflow round. `quantize` chooses its source round from `--initial_round` if provided; otherwise it prefers the latest accepted prune round and falls back to the top-scoring MCTS round when no pruned round exists.

Run pruning from an explicit round:

```bash
python -m src.cli.run \
  --config config/config.yaml \
  --dataset MATH \
  --pipelines prune \
  --workspace workspace \
  --initial_round 9 \
  --traverse_sample_size 16 \
  --prune_eval_samples 70 \
  --prune_threshold 0.95 \
  --exec_model_name gpt-4.1-mini
```

Run quantization from a specific accepted prune round:

```bash
python -m src.cli.run \
  --config config/config.yaml \
  --dataset MATH \
  --pipelines quantize \
  --workspace workspace \
  --initial_round 2 \
  --traverse_sample_size 16 \
  --quantize_eval_samples 70 \
  --quantize_threshold 0.95 \
  --quantize_rate 0.4 \
  --exec_model_name gpt-4.1-mini \
  --quantize_low_model_name gpt-4.1-nano
```

Run the default pipeline chain (this is also what you get if you omit `--pipelines`):

```bash
python -m src.cli.run \
  --config config/config.yaml \
  --dataset MATH \
  --pipelines mcts,prune,quantize \
  --workspace workspace \
  --max_rounds 10 \
  --sample 4 \
  --opt_model_name gpt-4.1 \
  --exec_model_name gpt-4.1-mini \
  --quantize_low_model_name gpt-4.1-nano
```

Because this chain includes both `mcts` and `quantize`, it requires all three model flags:
`--opt_model_name`, `--exec_model_name`, and `--quantize_low_model_name`.

Run finetuning from the latest quantized round:

```bash
python -m src.cli.run \
  --config config/config.yaml \
  --dataset MATH \
  --pipelines finetune \
  --workspace workspace \
  --sample 4 \
  --max_rounds 5 \
  --validation_rounds 1 \
  --finetune_eval_samples 20 \
  --opt_model_name gpt-4.1 \
  --exec_model_name gpt-4.1-mini
```

If `workflows_finetuned/round_<n>` already exists, `finetune` resumes from that round as the seed and writes any new outputs to `round_<n+1>` through `round_<max_rounds>`.

## Wrapper Scripts

Useful scripts under `src/cli/` are thin wrappers around `python -m src.cli.run`:

```bash
PYTHON=.venv/bin/python OPT_MODEL=<optimizer_model> EXEC_MODEL=<executor_model> DATASET=AIME MAX_ROUNDS=10 src/cli/mcts.sh
EXEC_MODEL=<executor_model> DATASET=MATH INITIAL_ROUND=9 src/cli/prune.sh
EXEC_MODEL=<executor_model> LOW_MODEL=<low_cost_model> DATASET=MATH INITIAL_ROUND=2 src/cli/quantize.sh
OPT_MODEL=<optimizer_model> EXEC_MODEL=<executor_model> DATASET=MATH MAX_ROUNDS=10 src/cli/finetune.sh
```

For the full current flag surface, run:

```bash
python -m src.cli.run -h
```
