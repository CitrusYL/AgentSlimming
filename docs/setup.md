# Setup

This page covers installation, local model config, and repository layout. Dataset download/reporting policy lives in [data.md](data.md), and CLI usage lives in [running.md](running.md).

## Repository Layout

```text
src/core/                 GraphFlow, Edge, GraphExecutor, BaseWorkflow, node runtime
src/catalog/              Dataset registry, answer-format registry, dataset-specific format prompts
src/evaluation/           Benchmark registry and evaluation runner
src/optimizer/            Workflow pruning/quantization heuristics, graph transforms, workspace and graph IO
src/pipeline/             MCTS, pruning, quantization, finetuning
src/prompts/              Optimizer prompts, workflow template, operator catalog
src/utils/                LLM config/loading, cost tracking, code execution, workspace IO
src/utils/optimizer_utils Ranking, experience, convergence, and optimizer parsing helpers
benchmarks/               Dataset-specific benchmark runners
data/                     Dataset builders and local dataset directory
config/                   Example model configuration
workspace/MATHDEMO        Tracked demo dataset with two example rounds
```

Most experiment outputs are intentionally ignored by git. The tracked `workspace/MATHDEMO` directory serves as a minimal runnable example.

## Environment

Option 1: `conda`

```bash
conda create -n agentslimming python=3.11
conda activate agentslimming
pip install -r requirements.txt
```

Option 2: `venv`

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Model Config

Create a local config:

```bash
cp config/config.example.yaml config/config.yaml
```

Minimal example:

```yaml
models:
  "gpt-4.1-mini":
    api_type: "openai"
    base_url: "<your base url>"
    api_key: "<your api key>"
    temperature: 0
    top_p: 1
    input_price: 0.0004
    output_price: 0.0016
```

Required fields:

- `api_type`
- `base_url`
- `api_key` or `key`
- `temperature`
- `input_price`
- `output_price`

Defaults:

- `model`: defaults to the enclosing model entry name if omitted
- `top_p`: defaults to `1.0` if omitted

There is no hardcoded pricing fallback in the code. If a workflow uses a model without `input_price` or `output_price`, execution fails early.

Config resolution order:

1. `--config /path/to/config.yaml`
2. `AGENT_SLIMMING_CONFIG=/path/to/config.yaml`
3. `<repo_root>/config/config.yaml`
4. `<cwd>/config/config.yaml`
5. `<cwd>/config.yaml`

When you run commands from the repository root, `<repo_root>/config/config.yaml` and `<cwd>/config/config.yaml` usually resolve to the same file.

## Environment Variables

- `AGENT_SLIMMING_CONFIG`: optional override for the default model config path. Used when you do not want to pass `--config` on every command.
- `AGENT_SLIMMING_LOG_LEVEL`: optional log verbosity override for the repository logger. Typical values are `DEBUG`, `INFO`, `WARNING`, and `ERROR`. The default is `INFO`.

Dataset- and benchmark-specific variables such as `HF_TOKEN` and `AGENT_SLIMMING_LIVECODE_USE_PRIVATE_TESTS` are documented in [data.md](data.md).
