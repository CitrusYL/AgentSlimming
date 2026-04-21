import json
import re
from string import Template
from typing import Any, Optional

from src.core.nodes.runtime_specs import RuntimeNodeSpec
from src.utils.logs import logger

DEFAULT_RESULT_FIELDS = ("output", "response", "answer", "solution", "code")


def compose_prompt(custom_prompt: str | None, default_prompt: str) -> str:
    custom = (custom_prompt or "").strip()
    default = default_prompt.strip()

    if custom and default:
        return f"{custom}\n\nHere are more detailed requirements:\n{default}"
    return custom or default


def build_prompt(
    node_id: str,
    spec: RuntimeNodeSpec,
    prompt_template: str,
    **inputs: Any,
) -> str:
    formatted_prompt = render_template(node_id, prompt_template, inputs)
    formatter = spec.create_formatter(function_name=inputs.get("function_name"))
    return formatter.prepare_prompt(formatted_prompt) if formatter else formatted_prompt


def render_template(node_id: str, prompt: str, inputs: dict[str, Any]) -> str:
    placeholders = list(dict.fromkeys(re.findall(r"\{(\w+)\}", prompt)))
    if not placeholders:
        parts = _format_inputs(inputs)
        return "\n".join(parts + [prompt]) if parts else prompt

    pending_inputs = _normalize_template_inputs(inputs)
    template_values = {
        placeholder: pending_inputs.pop(placeholder, "")
        for placeholder in placeholders
    }

    if "input" in placeholders and not template_values["input"]:
        formatted = _format_inputs(pending_inputs)
        template_values["input"] = "\n".join(formatted) if formatted else "None"

    missing = [
        placeholder
        for placeholder, value in template_values.items()
        if not str(value).strip()
    ]
    log_missing_placeholders(node_id, missing)

    template_prompt = prompt
    for placeholder in placeholders:
        template_prompt = template_prompt.replace(f"{{{placeholder}}}", f"${placeholder}")

    return Template(template_prompt).safe_substitute(**template_values)


def parse_response(
    output: str,
    spec: RuntimeNodeSpec,
    *,
    function_name: str | None = None,
) -> dict[str, Any]:
    formatter = spec.create_formatter(function_name=function_name)
    if formatter is not None:
        try:
            is_valid, result = formatter.validate_response(output)
            if is_valid and isinstance(result, dict):
                return result
        except Exception as exc:
            logger.warning(f"[PARSE] {spec.key} formatter failed: {exc}")

    parsed = _parse_json(output)
    if parsed:
        return parsed

    parsed = _parse_xml(output)
    if parsed:
        return parsed

    logger.warning(f"[PARSE] Failed to parse response for node spec {spec.key}")
    return {}


def log_missing_placeholders(node_id: str, missing: list[str]) -> None:
    if missing:
        logger.warning(
            f"[PROMPT] Node {node_id} missing placeholder values: {', '.join(sorted(missing))}"
        )


def extract_text(
    value: Any,
    *,
    fields: tuple[str, ...] = DEFAULT_RESULT_FIELDS,
    allow_plain_string: bool = False,
) -> Optional[str]:
    if isinstance(value, dict):
        if value.get("success", True) is False:
            return None
        for field in fields:
            candidate = value.get(field)
            if candidate is not None:
                return str(candidate)
        return None

    if allow_plain_string and isinstance(value, str):
        stripped = value.strip()
        return stripped or None

    return None


def first_text(
    inputs: dict[str, Any],
    *,
    ignore_keys: tuple[str, ...] = ("problem", "entry_point"),
    fields: tuple[str, ...] = DEFAULT_RESULT_FIELDS,
) -> Optional[str]:
    for key, value in inputs.items():
        if key in ignore_keys:
            continue
        text = extract_text(value, fields=fields)
        if text:
            return text
    return None


def collect_texts(
    inputs: dict[str, Any],
    *,
    ignore_keys: tuple[str, ...] = ("problem", "entry_point"),
    fields: tuple[str, ...] = DEFAULT_RESULT_FIELDS,
) -> list[str]:
    texts = []
    for key, value in inputs.items():
        if key in ignore_keys:
            continue
        text = extract_text(value, fields=fields)
        if text:
            texts.append(text)
    return texts


def pick_text(parsed: dict[str, Any], *fields: str) -> Optional[str]:
    for field in fields:
        value = parsed.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def resolve_entry_point(inputs: dict[str, Any], default: str = "solve") -> str:
    explicit = inputs.get("entry_point")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    for value in inputs.values():
        if isinstance(value, dict):
            candidate = value.get("entry_point")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

    return default


def _normalize_template_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(inputs)

    if "problem" in normalized:
        problem_text = extract_text({"problem": normalized["problem"]}, fields=("problem",))
        if problem_text:
            normalized.setdefault("question", problem_text)

    return normalized


def _format_inputs(inputs: dict[str, Any]) -> list[str]:
    parts = []
    for key, value in inputs.items():
        text = extract_text(
            value,
            allow_plain_string=(key == "problem"),
        )
        if text is not None:
            label = "Problem" if key == "problem" else key
            parts.append(f"{label}: {text}")
    return parts


def _parse_json(output: str) -> dict[str, Any]:
    try:
        result = json.loads(output)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


def _parse_xml(output: str) -> dict[str, Any]:
    matches = re.findall(r"<(\w+)>(.*?)</\1>", output, re.DOTALL)
    return {key: value.strip() for key, value in matches}
