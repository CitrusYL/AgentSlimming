from pydantic import BaseModel, Field


class GraphOptimize(BaseModel):
    modification: str = Field(
        default="",
        description="Describe key modifications in this round",
    )
    graph: str = Field(
        default="",
        description=(
            "Python body of Workflow._build_graph using only current fields. "
            "Allowed edge fields are source, target, key, description, and as_candidate."
        ),
    )
    prompt: str = Field(
        default="",
        description="Python prompt constants referenced by prompt_custom.<NAME>",
    )
