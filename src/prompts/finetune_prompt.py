WORKFLOW_OPTIMIZE_PROMPT = """You are optimizing a GraphFlow workflow for {type}, with cost-aware finetuning allowed.
Return only the fields requested by the formatter: modification, graph, and prompt.

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
9. You may use a lower-cost model for less critical nodes by setting `node_llm_config={{"model": "<low_model>", "temperature": 0.0}}` when the caller provides one.

Prompt output contract:
1. The <prompt> field must contain only Python constant definitions used by `prompt_custom.<NAME>` in CustomNode or CustomCodeGenerateNode.
2. Do not include prompt definitions for built-in nodes.
3. Do not include unresolved placeholders.
"""


WORKFLOW_INPUT = """
Here is a previous high-performing workflow and its custom prompt definitions. Use it as a reference and make one clear improvement. The modified graph must differ from the sample, and the difference must be described in <modification>.\n
<sample>
    <experience>{experience}</experience>
    <modification>(such as:add /delete /modify/ ...)</modification>
    <score>{score}</score>
    <graph>{graph}</graph>
    <prompt>{prompt}</prompt>
    <operator_spec>{operator_spec}</operator_spec>
</sample>
You can create chains, branches, merges, and skip connections, but the final graph must remain a valid DAG.

Below are the logs of some results with the aforementioned Graph that performed well but encountered errors, which can be used as references for optimization:
{log}

Optimization Constraints:

1. Make one atomic improvement per iteration: update one prompt, tune one node, change one node's model config, or add one node plus its necessary edges and prompt.
2. Keep the generated graph executable and compact.
3. If the sample graph includes a full `Workflow` class, treat it only as reference. Your <graph> output must still be only the body of `_build_graph(self)`.
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

If one target node needs multiple named inputs, use `key`, for example:
edges = [
    Edge(source="ReasonerA", target="Scorer", key="candidate_a"),
    Edge(source="ReasonerB", target="Scorer", key="candidate_b"),
]
Do not use `source_key` or any other unsupported edge field.

For CustomNode or CustomCodeGenerateNode prompts, write only Python constants in the <prompt> field, for example:
PROMPT_REASONING = "Analyze the problem step by step and produce a concise answer."
"""
