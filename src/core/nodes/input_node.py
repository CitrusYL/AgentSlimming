from typing import Any, ClassVar, Dict

from src.core.nodes.base import Node


class InputNode(Node):
    """Workflow entry node. It provides graph structure but emits no business output."""

    spec_name: ClassVar[str] = "Input"

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.success()
