from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.utils.sanitize import sanitize

CheckMode = Literal["candidate", "no_args"]
PASS = "PASS"
FAIL = "FAIL"
TIMEOUT_FAILURE_MESSAGE = (
    "Execution timed out. Please check if your solution contains infinite loops "
    "or overly time-consuming operations."
)


@dataclass(frozen=True)
class CodeCheckRequest:
    solution: str
    test: str
    entry_point: str
    timeout: int = 15
    check_mode: CheckMode = "candidate"
    prelude: str = ""


def run_code_check(request: CodeCheckRequest) -> tuple[str, str]:
    ctx = _multiprocessing_context()
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_run_code_check_worker, args=(request, result_queue))
    process.start()
    process.join(request.timeout)

    if process.is_alive():
        process.kill()
        process.join(timeout=1)
        return FAIL, TIMEOUT_FAILURE_MESSAGE

    try:
        result = result_queue.get_nowait()
    except Empty:
        return FAIL, f"Code check process exited unexpectedly with code {process.exitcode}."

    return result


def _run_code_check_worker(request: CodeCheckRequest, result_queue: Any) -> None:
    try:
        _execute_check(request)
        result_queue.put((PASS, "The solution passed all test cases."))
    except Exception as exc:
        prepared_solution = _prepare_solution(request)
        error_message = (
            f"Error: {str(exc)}.\n Solution: {prepared_solution}.\n Test: {request.test}"
        )
        result_queue.put((FAIL, error_message))


def _execute_check(request: CodeCheckRequest) -> None:
    prepared_solution = _prepare_solution(request)
    global_dict = _build_exec_globals()

    exec(prepared_solution, global_dict)

    if request.entry_point not in global_dict:
        raise ValueError(f"Function {request.entry_point} is not defined in the solution.")

    exec(request.test, global_dict)
    check = global_dict["check"]

    if request.check_mode == "candidate":
        check(global_dict[request.entry_point])
        return

    if request.check_mode == "no_args":
        check()
        return

    raise ValueError(f"Unsupported check mode: {request.check_mode}")


def _prepare_solution(request: CodeCheckRequest) -> str:
    solution = sanitize(code=request.solution, entrypoint=request.entry_point)
    if request.prelude:
        return request.prelude + "\n\n" + solution
    return solution


def _build_exec_globals() -> Dict[str, Any]:
    return {
        "math": __import__("math"),
        "hashlib": __import__("hashlib"),
        "re": __import__("re"),
        "List": List,
        "Dict": Dict,
        "Tuple": Tuple,
        "Optional": Optional,
        "Any": Any,
    }


def _multiprocessing_context() -> mp.context.BaseContext:
    methods = mp.get_all_start_methods()
    if "fork" in methods:
        return mp.get_context("fork")
    return mp.get_context("spawn")
