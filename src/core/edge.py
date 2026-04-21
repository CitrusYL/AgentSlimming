from typing import Optional

from pydantic import BaseModel, ConfigDict


class Edge(BaseModel):
    """Directed dependency between workflow nodes."""
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    key: Optional[str] = None
    description: Optional[str] = None
    as_candidate: bool = False

    def input_name(self) -> str:
        return self.key or self.source
