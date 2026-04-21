from src.core.nodes.catalog import (
    NODE_DEFINITIONS,
    NODE_SPEC_ALIASES,
    RuntimeNodeSpec,
    get_runtime_node_spec,
    resolve_node_spec_name,
)

NODE_RUNTIME_SPECS: dict[str, RuntimeNodeSpec] = {
    key: definition.runtime_spec for key, definition in NODE_DEFINITIONS.items()
}

__all__ = [
    "NODE_RUNTIME_SPECS",
    "NODE_SPEC_ALIASES",
    "RuntimeNodeSpec",
    "get_runtime_node_spec",
    "resolve_node_spec_name",
]
