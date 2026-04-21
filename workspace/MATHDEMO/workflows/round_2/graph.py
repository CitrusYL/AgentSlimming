from src.core.edge import Edge
from src.core.graphflow import GraphFlow
from src.core.nodes.answer_format_node import AnswerFormatNode
from src.core.nodes.custom_node import CustomNode
from src.core.nodes.input_node import InputNode
from src.core.nodes.sc_ensemble_node import ScEnsembleNode
from src.core.workflow import BaseWorkflow
from . import prompt as prompt_custom


class Workflow(BaseWorkflow):
    def _build_graph(self) -> GraphFlow:
        Input = InputNode(
            node_id="Input",
            node_llm_config=self.llm_config,
            description="Graph input entry.",
        )
        SolveConcise = CustomNode(
            node_id="SolveConcise",
            node_prompt=prompt_custom.CONCISE_SOLUTION_PROMPT,
            node_llm_config=self.llm_config,
            description="Produce a concise mathematical solution.",
        )
        SolveDetailed = CustomNode(
            node_id="SolveDetailed",
            node_prompt=prompt_custom.DETAILED_SOLUTION_PROMPT,
            node_llm_config=self.llm_config,
            description="Produce a detailed mathematical solution.",
        )
        SolveAlternative = CustomNode(
            node_id="SolveAlternative",
            node_prompt=prompt_custom.ALTERNATIVE_SOLUTION_PROMPT,
            node_llm_config=self.llm_config,
            description="Produce an alternative mathematical solution path.",
        )
        SelectConsensus = ScEnsembleNode(
            node_id="SelectConsensus",
            node_llm_config=self.llm_config,
            description="Choose the most reliable candidate solution.",
        )
        AnswerFormatter = AnswerFormatNode(
            node_id="AnswerFormatter",
            dataset_name=self.dataset,
            node_llm_config=self.llm_config,
            description="Format the final answer for the target dataset.",
        )
        return GraphFlow(
            nodes=[
                Input,
                SolveConcise,
                SolveDetailed,
                SolveAlternative,
                SelectConsensus,
                AnswerFormatter,
            ],
            edges=[
                Edge(source="Input", target="SolveConcise"),
                Edge(source="Input", target="SolveDetailed"),
                Edge(source="Input", target="SolveAlternative"),
                Edge(
                    source="SolveConcise",
                    target="SelectConsensus",
                    key="candidate_a",
                    as_candidate=True,
                ),
                Edge(
                    source="SolveDetailed",
                    target="SelectConsensus",
                    key="candidate_b",
                    as_candidate=True,
                ),
                Edge(
                    source="SolveAlternative",
                    target="SelectConsensus",
                    key="candidate_c",
                    as_candidate=True,
                ),
                Edge(source="SelectConsensus", target="AnswerFormatter"),
            ],
            entry_node_ids=["Input"],
            final_node_id="AnswerFormatter",
            description="Tracked MATHDEMO round-2 workflow for prune and quantize examples.",
        )
