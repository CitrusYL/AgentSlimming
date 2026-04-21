from src.core.edge import Edge
from src.core.graphflow import GraphFlow
from src.core.nodes.answer_format_node import AnswerFormatNode
from src.core.nodes.custom_node import CustomNode
from src.core.nodes.input_node import InputNode
from src.core.workflow import BaseWorkflow
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
            description="Reason over the problem and produce a concise answer.",
        )
        AnswerFormatter = AnswerFormatNode(
            node_id="AnswerFormatter",
            dataset_name=self.dataset,
            node_llm_config=self.llm_config,
            description="Format the final answer for the target dataset.",
        )
        return GraphFlow(
            nodes=[Input, Reasoner, AnswerFormatter],
            edges=[
                Edge(source="Input", target="Reasoner"),
                Edge(source="Reasoner", target="AnswerFormatter"),
            ],
            entry_node_ids=["Input"],
            final_node_id="AnswerFormatter",
            description="Minimal MATHDEMO round-1 workflow.",
        )
