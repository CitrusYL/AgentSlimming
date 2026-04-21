import re


def extract_xml_fields(response: str, fields: list[str]) -> dict[str, str] | None:
    result = {field: "" for field in fields}
    for field in fields:
        match = re.search(rf"<{field}>(.*?)</{field}>", response, re.DOTALL)
        if match:
            result[field] = match.group(1).strip()
    return result if any(result.values()) else None


def extract_graph_optimize_fields(response: str) -> dict[str, str] | None:
    return extract_xml_fields(response, ["modification", "graph", "prompt"])
