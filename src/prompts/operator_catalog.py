from src.core.nodes.catalog import (
    NODE_DEFINITIONS,
    OperatorSpec,
    get_operator_spec,
)

OPERATOR_SPECS: dict[str, OperatorSpec] = {
    key: definition.operator_spec for key, definition in NODE_DEFINITIONS.items()
}


def format_operator_specs(operator_names: list[str]) -> str:
    return "\n\n".join(
        get_operator_spec(name).format_operator(index)
        for index, name in enumerate(operator_names, start=1)
    )


__all__ = [
    "OPERATOR_SPECS",
    "OperatorSpec",
    "format_operator_specs",
    "get_operator_spec",
]
