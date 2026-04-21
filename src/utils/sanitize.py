"""
Small AST-based code sanitizer for benchmark code-generation outputs.

Inspired by evalplus sanitize, but intentionally avoids tree-sitter so this
utility has no parser dependency beyond Python's stdlib.
"""

import ast
import re
import traceback
from typing import Optional


def syntax_check(code: str, verbose: bool = False) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def code_extract(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    lines = text.splitlines()
    best_start, best_end, best_length = 0, len(lines), 0
    for start in range(len(lines)):
        for end in range(start + 1, len(lines) + 1):
            candidate = "\n".join(lines[start:end])
            if not syntax_check(candidate):
                continue
            length = sum(1 for line in lines[start:end] if line.strip())
            if length > best_length:
                best_start, best_end, best_length = start, end, length

    return "\n".join(lines[best_start:best_end]).strip()


def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    code = code_extract(code)
    try:
        return _sanitize_with_ast(code, entrypoint)
    except Exception:
        return code


def _sanitize_with_ast(code: str, entrypoint: Optional[str] = None) -> str:
    tree = ast.parse(code)
    imports: list[str] = []
    definitions: list[tuple[str, str]] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.append((node.name, ast.unparse(node)))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions.append((target.id, ast.unparse(node)))

    if entrypoint:
        definitions = _reachable_definitions(definitions, entrypoint)

    return "\n".join([*imports, *(source for _, source in definitions)])


def _reachable_definitions(
    definitions: list[tuple[str, str]],
    entrypoint: str,
) -> list[tuple[str, str]]:
    dependency_graph = {name: set[str]() for name, _ in definitions}
    for name, source in definitions:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in dependency_graph and node.id != name:
                dependency_graph[name].add(node.id)

    reachable: set[str] = set()
    stack = [entrypoint]
    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        reachable.add(name)
        stack.extend(dependency_graph.get(name, []))

    return [(name, source) for name, source in definitions if name in reachable]
