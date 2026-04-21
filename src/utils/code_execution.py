import ast
import atexit
import asyncio
import concurrent.futures
import re
import sys
import traceback
from concurrent.futures.process import BrokenProcessPool
from typing import Iterable

from src.utils.logs import logger

DEFAULT_DISALLOWED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "multiprocessing",
    "matplotlib",
    "seaborn",
    "plotly",
    "bokeh",
    "ggplot",
    "pylab",
    "tkinter",
    "PyQt5",
    "wx",
    "pyglet",
}


def extract_python_code(text: str) -> str:
    """Return the first fenced Python block, or the stripped input text."""
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (code_blocks[0] if code_blocks else text).strip()


def _import_root(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def find_disallowed_import(code: str, disallowed_imports: Iterable[str]) -> str | None:
    disallowed = set(disallowed_imports)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _import_root(alias.name) in disallowed:
                    return _import_root(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if _import_root(node.module) in disallowed:
                return _import_root(node.module)
    return None


def run_python_solve(
    code: str,
    required_function: str = "solve",
    disallowed_imports: Iterable[str] = DEFAULT_DISALLOWED_IMPORTS,
) -> tuple[str, str]:
    """Execute generated Python code and call its required entrypoint."""
    blocked_import = find_disallowed_import(code, disallowed_imports)
    if blocked_import:
        return "Error", f"Prohibited import: {blocked_import}"

    try:
        namespace = {}
        logger.debug("\n[CODE]\n" + code + "\n[CODE END]\n")
        exec(code, namespace)

        entrypoint = namespace.get(required_function)
        if not callable(entrypoint):
            return "Error", f"Function '{required_function}' not found"

        result = entrypoint()
        return "Success", str(result)

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        return "Error", f"Execution error: {str(e)}\n{tb_str}"


class PythonSolveExecutor:
    """Shared process pool for generated Python code execution."""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers
        self._pool: concurrent.futures.ProcessPoolExecutor | None = None
        atexit.register(self.shutdown)

    def _ensure_pool(self) -> concurrent.futures.ProcessPoolExecutor:
        if self._pool is None:
            self._pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        return self._pool

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    async def run(self, code: str, timeout: int = 30) -> tuple[str, str]:
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(self._ensure_pool(), run_python_solve, code)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            future.cancel()
            return "Error", "Code execution timed out"
        except BrokenProcessPool:
            self.shutdown()
            return "Error", "Process pool broken, try again"
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            return "Error", f"Execution error: {str(e)}\n{tb_str}"


_PYTHON_SOLVE_EXECUTOR = PythonSolveExecutor()


async def run_python_solve_async(code: str, timeout: int = 30) -> tuple[str, str]:
    return await _PYTHON_SOLVE_EXECUTOR.run(code, timeout=timeout)
