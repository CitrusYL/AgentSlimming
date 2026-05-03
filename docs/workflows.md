# Workflows And Nodes

This page covers generated workflow artifacts, node registration, and workspace layout. Command flags and pipeline-selection details live in [running.md](running.md).

## Workflow Format

Generated `graph.py` files are intentionally small. They import node classes and implement only `Workflow._build_graph(self)`. Runtime behavior such as executor setup, `__call__`, problem payload normalization, final output extraction, and cost aggregation is handled by `src/core/workflow.py`.

Example:

```python
from src.core.graphflow import GraphFlow
from src.core.edge import Edge
from src.core.workflow import BaseWorkflow
from src.core.nodes.answer_format_node import AnswerFormatNode
from src.core.nodes.custom_node import CustomNode
from src.core.nodes.input_node import InputNode
from . import prompt as prompt_custom


class Workflow(BaseWorkflow):
    def _build_graph(self) -> GraphFlow:
        Input = InputNode(
            node_id="Input",
            node_llm_config=self.llm_config,
            description="Graph input entry.",
        )
        Reasoner = CustomNode(
            node_id="Reasoner",
            node_prompt=prompt_custom.PROMPT_REASONING,
            node_llm_config=self.llm_config,
            description="Reason over the problem.",
        )
        AnswerFormatter = AnswerFormatNode(
            node_id="AnswerFormatter",
            dataset_name=self.dataset,
            node_llm_config=self.llm_config,
            description="Format the final answer.",
        )
        return GraphFlow(
            nodes=[Input, Reasoner, AnswerFormatter],
            edges=[
                Edge(source="Input", target="Reasoner"),
                Edge(source="Reasoner", target="AnswerFormatter"),
            ],
            entry_node_ids=["Input"],
            final_node_id="AnswerFormatter",
        )
```
    Note on cost counting:

    - Each node instance supports an optional `count_towards_cost` flag (defaults to `True`).
    - When set to `False`, the executor will skip adding that node's LLM usage to the final cost summary.
    - Example: exclude an internal debugging node from cost accounting:

    ```python
    Reasoner = CustomNode(
      node_id="Reasoner",
      node_prompt=prompt_custom.PROMPT_REASONING,
      node_llm_config=self.llm_config,
      description="Reason over the problem.",
      count_towards_cost=False,  # do not include this node in cost totals
    )
    ```

    The executor checks `node.count_towards_cost` when recording usage and will only add calls to the `LLMCostTracker` if the flag is `True`.

The adjacent `prompt.py` file contains only custom prompt constants used by prompt-driven nodes such as `CustomNode` or `CustomCodeGenerateNode`.

## Node Registration

Node registration is centralized in `src/core/nodes/catalog.py`.

The compatibility modules:

- `src/core/nodes/runtime_specs.py`
- `src/prompts/operator_catalog.py`

read from that catalog instead of maintaining separate copies.

Together they define:

- available node types
- node aliases and import paths
- default prompts
- formatter and response parsing behavior
- operator descriptions injected into optimizer prompts

## Extending Nodes

To add a new node type without changing the optimizer/runtime plumbing:

1. Implement the node class under `src/core/nodes/` by subclassing `Node` and keeping task-specific behavior inside `execute()`.
2. Add one `NodeDefinition` entry in `src/core/nodes/catalog.py`.
3. Fill in the node key, aliases if any, class name, module file, runtime prompt and formatter settings, and optimizer-facing operator description in that single definition.
4. Add the node key to the allowed operator tuple for the target datasets in `src/catalog/datasets.py` if you want MCTS to use it.
5. If the node needs custom prompt constants, keep them in the generated `prompt.py` artifact and mark the operator as `prompt_required=True` in the catalog.

`NodeDefinition` is the single source of truth for:

- runtime prompt and formatter behavior
- optimizer operator descriptions
- node aliases
- generated import statements for `graph.py`
- dynamic class loading helpers

Normal node extensions should not require separate edits in `runtime_specs.py`, `operator_catalog.py`, or `graph_utils.py`.

## Workspace Layout

The optimizer reads and writes workflow artifacts under the selected workspace root:

```text
workspace/<dataset>/
  workflows/
    round_1/
      graph.py
      prompt.py
      manifest.json
    results.json
  workflows_pruned/
    round_<n>/
      graph.py
      prompt.py
      manifest.json
  workflows_quantized/
    round_<n>/
      graph.py
      prompt.py
      manifest.json
  workflows_finetuned/
    round_<n>/
      graph.py
      prompt.py
      manifest.json
  prune_evaluations/
  quantize_evaluations/
  finetune_evaluations/
```

Pipeline behavior:

- `mcts` reads and writes `workflows/`
- `prune` reads from `workflows/` and writes stage-local accepted outputs to `workflows_pruned/round_<n>/`
- `quantize` reads from `workflows_pruned/` when available, otherwise falls back to `workflows/`, and writes stage-local accepted outputs to `workflows_quantized/round_<n>/`
- `finetune` uses the latest valid `workflows_finetuned/round_<n>` as the seed round if it exists, otherwise initializes `workflows_finetuned/round_1` from the latest quantized round
- resumed `finetune` runs always write new artifacts to the next round number (`round_<n+1>` onward); they do not overwrite the seed round directory
- every accepted round writes its lineage and summary metrics to `manifest.json`
- for `finetune`, `--max_rounds` means the target final round number under `workflows_finetuned/`

Prune and quantize keep internal candidate search details inside the accepted round directory, instead of creating extra `iter` subdirectories.
