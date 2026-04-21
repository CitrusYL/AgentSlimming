WORKFLOW_OPTIMIZE_PROMPT = """You are building and optimizing a GraphFlow workflow for {type} problems.
Referring to the given graph and prompt, reconstruct and improve them. You may add, modify, or delete nodes, edges,
parameters, or prompts. Return only the formatter fields: modification, graph, and prompt.

The workflow may use critical-thinking patterns such as review, revise, self-ask, decomposition, parallel candidate
generation, self-consistency, or verifier-then-refine structures. In GraphFlow, express these behaviors with DAG
structure: branches, merges, skip connections, and intermediate aggregator or formatter nodes. A single modification
means one coherent optimization strategy, but that strategy may require multiple coordinated code edits such as adding
several nodes and edges together.

Graph output contract:
1. The <graph> field must contain only the Python body of `Workflow._build_graph(self)`.
2. Do not output imports, `class Workflow`, `__init__`, `__call__`, `GraphExecutor`, or markdown fences.
3. Instantiate node variables, define `edges`, and end with `return GraphFlow(...)`.
4. Use only the current implementation field names:
   - Node fields: `node_id`, `node_prompt`, `node_llm_config`, `description`
   - Edge fields: `source`, `target`, optional `key`, optional `description`, optional `as_candidate`
   - GraphFlow fields: `nodes`, `edges`, `entry_node_ids`, `final_node_id`, optional `description`
5. Never use stale or invented fields such as `source_key`, `target_key`, `input_key`, `node_output`, or `node_role`.
6. Use `self.llm_config`, `self.dataset`, and `prompt_custom.<NAME>` when needed.
7. The graph must start with `InputNode(node_id="Input", ...)`, use `entry_node_ids=["Input"]`, and end at an `AnswerFormatNode`.
8. Every non-entry node must be reachable from `Input`, node ids must be unique, and edge input names for the same target must be unique.
9. Use only node classes listed in the operator specification.
10. Keep the graph executable, compact, and within 10 nodes.
11. Ensure all prompt_custom constants required by the current graph are included, and exclude unused prompt_custom constants.
12. The generated prompt must not contain unresolved placeholders.
13. Never output `None` for required graph fields.

Prompt output contract:
1. The <prompt> field must contain only Python constant definitions used by `prompt_custom.<NAME>` in CustomNode or CustomCodeGenerateNode.
2. Do not include prompt definitions for built-in nodes.
3. If no custom prompts are needed, return an empty prompt section rather than unrelated content.
"""


WORKFLOW_INPUT = """
Here is a graph and the corresponding CustomNode prompt definitions that performed well in a previous iteration. You must
optimize and improve this workflow further. The modified graph must differ from the sample, and the exact change must be
described in <modification>.\n
<sample>
    <experience>{experience}</experience>
    <modification>(such as:add /delete /modify /restructure /parallelize /ensemble /revise ...)</modification>
    <score>{score}</score>
    <graph>{graph}</graph>
    <prompt>{prompt}</prompt>
    <operator_spec>{operator_spec}</operator_spec>
</sample>
You can create chains, branches, merges, and skip connections, but the final graph must remain a valid DAG.

Below are logs from this high-performing workflow when it still made mistakes. Use them as concrete references for what
to improve:
{log}

Optimization guidance:
1. Make one coherent improvement per iteration. That improvement may involve multiple coordinated node, edge, and prompt changes if they serve the same idea.
2. Large structural improvements are allowed when justified, for example introducing parallel candidate generators plus a merge node for self-consistency.
3. Prefer meaningful reconstruction over tiny cosmetic edits. Do not repeat past failed modifications from experience.
4. If the sample graph includes a full `Workflow` class, treat it only as reference. Your <graph> output must still be only the body of `_build_graph(self)`.
5. Use the operator specification actively. If relevant operators are available but unused, consider incorporating them.
6. Preserve enough information flow so downstream nodes receive the context they need; avoid adding branches that never influence the final answer.
"""

WORKFLOW_PROMPT_USE = """\nUse this graph-body pattern:
Input = InputNode(
    node_id="Input",
    node_llm_config=self.llm_config,
    description="Graph input entry",
)
Reasoner = CustomNode(
    node_id="Reasoner",
    node_prompt=prompt_custom.PROMPT_REASONING,
    node_llm_config=self.llm_config,
    description="Solve the problem step by step.",
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
    description="...",
)

Parallel self-consistency pattern example:
Input = InputNode(
    node_id="Input",
    node_llm_config=self.llm_config,
    description="Graph input entry",
)
ReasonerA = CustomNode(
    node_id="ReasonerA",
    node_prompt=prompt_custom.PROMPT_REASONING_A,
    node_llm_config=self.llm_config,
    description="Generate candidate solution A.",
)
ReasonerB = CustomNode(
    node_id="ReasonerB",
    node_prompt=prompt_custom.PROMPT_REASONING_B,
    node_llm_config=self.llm_config,
    description="Generate candidate solution B.",
)
Consensus = ScEnsembleNode(
    node_id="Consensus",
    node_llm_config=self.llm_config,
    description="Select the most reliable candidate.",
)
AnswerFormatter = AnswerFormatNode(
    node_id="AnswerFormatter",
    dataset_name=self.dataset,
    node_llm_config=self.llm_config,
    description="Format the final answer.",
)
return GraphFlow(
    nodes=[Input, ReasonerA, ReasonerB, Consensus, AnswerFormatter],
    edges=[
        Edge(source="Input", target="ReasonerA"),
        Edge(source="Input", target="ReasonerB"),
        Edge(source="ReasonerA", target="Consensus", key="candidate_a"),
        Edge(source="ReasonerB", target="Consensus", key="candidate_b"),
        Edge(source="Consensus", target="AnswerFormatter"),
    ],
    entry_node_ids=["Input"],
    final_node_id="AnswerFormatter",
    description="Parallel candidate generation followed by self-consistency.",
)

Multiple upstream inputs are allowed. When one target node needs multiple named inputs, use `key`, for example:
edges = [
    Edge(source="ReasonerA", target="Verifier", key="draft_answer"),
    Edge(source="Programmer", target="Verifier", key="tool_answer"),
]
Do not use `source_key` or any other unsupported edge field.

For CustomNode prompts, write only Python constants in the <prompt> field, for example:
PROMPT_REASONING = "Analyze the problem step by step and produce a concise answer."

Introducing multiple operators at appropriate points can improve performance. If relevant operators are available but
not yet used in the graph, consider incorporating them.
"""

WORKFLOW_TEMPLATE = """from src.core.graphflow import GraphFlow
from src.core.edge import Edge
from src.core.workflow import BaseWorkflow
{dynamic_imports}
from . import prompt as prompt_custom


class Workflow(BaseWorkflow):
    def _build_graph(self) -> GraphFlow:
{graph}
"""
