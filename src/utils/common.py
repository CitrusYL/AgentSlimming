import json
from pathlib import Path
from typing import Any

from pydantic_core import to_jsonable_python


def read_json_file(json_file: str | Path, encoding: str = "utf-8") -> Any:
    path = Path(json_file)
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    with path.open("r", encoding=encoding) as fin:
        try:
            return json.load(fin)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to read JSON file: {path}") from exc


def write_json_file(
    json_file: str | Path,
    data: Any,
    encoding: str = "utf-8",
    indent: int = 4,
) -> None:
    path = Path(json_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=indent, default=to_jsonable_python)
