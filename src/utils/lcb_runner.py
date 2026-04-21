import ast
import json
import sys
import faulthandler
import platform
from datetime import datetime
import signal
from io import StringIO
from unittest.mock import patch, mock_open
from types import ModuleType
from enum import Enum
from decimal import Decimal
import time
import io
from typing import Any, Optional, Tuple, List, Dict
import inspect

import_string = (
    "from string import *\n"
    "from re import *\n"
    "from datetime import *\n"
    "from collections import *\n"
    "from heapq import *\n"
    "from bisect import *\n"
    "from copy import *\n"
    "from math import *\n"
    "from random import *\n"
    "from statistics import *\n"
    "from itertools import *\n"
    "from functools import *\n"
    "from operator import *\n"
    "from io import *\n"
    "from sys import *\n"
    "from json import *\n"
    "from builtins import *\n"
    "from typing import *\n"
    "import string\n"
    "import re\n"
    "import datetime\n"
    "import collections\n"
    "import heapq\n"
    "import bisect\n"
    "import copy\n"
    "import math\n"
    "import random\n"
    "import statistics\n"
    "import itertools\n"
    "import functools\n"
    "import operator\n"
    "import io\n"
    "import sys\n"
    "import json\n"
    "sys.setrecursionlimit(50000)\n"
)

def truncatefn(s, length=300):
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("timeout occured: alarm went off")
    raise TimeoutException


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda *args, **kwargs: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class DynamicStdinWithBuffer:
    def __init__(self, inputs: str = ""):
        self.set_inputs(inputs)

    def set_inputs(self, inputs: Any):
        if isinstance(inputs, list):
            inputs = "\n".join(str(x) for x in inputs)
        if inputs is None:
            inputs = ""
        inputs = str(inputs)
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self._bytesio = io.BytesIO(inputs.encode("utf-8"))
        self.buffer = self._bytesio

    def read(self, *args):
        return self._stringio.read(*args)

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self._stringio.readlines(*args)

    def __iter__(self):
        return iter(self._stringio)

    def __getattr__(self, name):
        return getattr(self._stringio, name)


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        if not astree.body:
            return code
        last_block = astree.body[-1]
        if not isinstance(last_block, ast.If):
            return code

        cond_str = ast.unparse(last_block.test).strip()
        patterns = {
            "__name__ == '__main__'",
            '__name__ == "__main__"',
            "__name__=='__main__'",
            '__name__=="__main__"',
            "globals().get('__name__', '__main__') == '__main__'",
            'globals().get("__name__", "__main__") == "__main__"',
            "globals().get('__name__','__main__')=='__main__'",
            'globals().get("__name__","__main__")=="__main__"',
        }
        if cond_str in patterns:
            new_body = astree.body[:-1]
            code = ast.unparse(new_body) + "\n"
    except Exception:
        pass
    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
            + "\n"
        )
        return main_code
    except Exception:
        return code


def call_method(method, stdin_obj: DynamicStdinWithBuffer):
    inputs = stdin_obj.inputs
    with patch("builtins.open", mock_open(read_data=inputs)):
        with patch("sys.stdin", stdin_obj), patch("sys.__stdin__", stdin_obj):
            try:
                return method()
            except SystemExit:
                return None


def get_function(compiled_sol, fn_name: str):
    try:
        if hasattr(compiled_sol, fn_name):
            fn = getattr(compiled_sol, fn_name)
            if callable(fn):
                return fn
    except Exception:
        pass
    return None


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
        assert compiled_sol is not None
    finally:
        signal.alarm(0)
    return compiled_sol


def convert_line_to_decimals(line: str) -> Tuple[bool, List[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str) -> List[str]:
    val = val.strip()
    return [val_line.strip() for val_line in val.split("\n")]


def _infer_nargs(func) -> Optional[int]:
    try:
        sig = inspect.signature(func)
        params = []
        for p in sig.parameters.values():
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                params.append(p)
            elif p.kind == p.VAR_POSITIONAL:
                return None
        required = [p for p in params if p.default is inspect._empty]
        return len(required)
    except Exception:
        return None


def _parse_call_args(inputs_str: str, expected_nargs: Optional[int]) -> List[Any]:
    s = (inputs_str or "").strip()
    if not s:
        return []

    if "\n" not in s:
        try:
            obj = json.loads(s)
        except Exception:
            obj = None

        if obj is not None:
            if isinstance(obj, list):
                if expected_nargs == 1:
                    return [obj]
                if expected_nargs is not None and len(obj) == expected_nargs:
                    return obj
                return [obj]
            return [obj]

    args: List[Any] = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        args.append(json.loads(line))
    return args


def grade_call_based(code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int):
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return [-4], {"error_code": -4, "error_message": "Compile returned None"}

    method = get_function(compiled_sol, fn_name)
    if method is None:
        return [-4], {"error_code": -4, "error_message": f"Function not found: {fn_name}"}

    expected_nargs = _infer_nargs(method)
    parsed_inputs = [_parse_call_args(inp, expected_nargs) for inp in all_inputs]

    parsed_outputs = []
    for output in all_outputs:
        try:
            parsed_outputs.append(json.loads(output))
        except Exception:
            parsed_outputs.append(output)

    total_execution = 0.0
    all_results: List[Any] = []

    for gt_inp, gt_out in zip(parsed_inputs, parsed_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)

            if isinstance(prediction, tuple):
                prediction = list(prediction)

            ok = (prediction == gt_out)
            all_results.append(ok)

            if not ok:
                return all_results, {
                    "output": truncatefn(prediction),
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }

        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                return all_results, {
                    "error": repr(e),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            else:
                all_results.append(-4)
                return all_results, {
                    "error": repr(e),
                    "error_code": -4,
                    "error_message": "Runtime Error",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
        finally:
            signal.alarm(0)
            faulthandler.disable()

    return all_results, {"execution time": total_execution}


def grade_stdio(code: str, all_inputs: list, all_outputs: list, timeout: int):
    code = clean_if_name(code)

    stdin_obj = DynamicStdinWithBuffer("")

    old_stdin = sys.stdin
    old__stdin = getattr(sys, "__stdin__", None)
    try:
        sys.stdin = stdin_obj
        sys.__stdin__ = stdin_obj  # type: ignore
        compiled_sol = compile_code(code, timeout)
    finally:
        sys.stdin = old_stdin
        if old__stdin is not None:
            sys.__stdin__ = old__stdin  # type: ignore

    if compiled_sol is None:
        return [-4], {"error_code": -4, "error_message": "Compile returned None"}

    method = None
    if hasattr(compiled_sol, "solve") and callable(getattr(compiled_sol, "solve")):
        method = getattr(compiled_sol, "solve")
    else:
        method = get_function(compiled_sol, "wrapped_function")

    if method is None:
        return [-4], {
            "error_code": -4,
            "error_message": "STDIN mode requires solve() or wrapped_function(), but neither was found",
        }

    all_results: List[Any] = []
    total_execution_time = 0.0

    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        stdin_obj.set_inputs(gt_inp)

        signal.alarm(timeout)
        faulthandler.enable()

        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, stdin_obj)
                total_execution_time += time.time() - start
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
                else:
                    all_results.append(-4)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -4,
                        "error_message": "Runtime Error",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]

        pred_lines = get_stripped_lines(prediction)
        gt_lines = get_stripped_lines(gt_out)

        wa_meta = {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
        }

        if len(pred_lines) != len(gt_lines):
            all_results.append(-2)
            wa_meta["error_message"] = "Wrong answer: mismatched output length"
            return all_results, wa_meta

        for idx, (pl, gl) in enumerate(zip(pred_lines, gt_lines)):
            if pl == gl:
                continue

            wa_meta["error_message"] = f"Wrong answer at output_line_idx={idx}: {truncatefn(pl)} != {truncatefn(gl)}"

            ok1, dec_pred = convert_line_to_decimals(pl)
            ok2, dec_gt = convert_line_to_decimals(gl)
            if not (ok1 and ok2):
                all_results.append(-2)
                return all_results, wa_meta

            if dec_pred != dec_gt:
                all_results.append(-2)
                return all_results, wa_meta

        all_results.append(True)

    return all_results, {"execution time": total_execution_time}


def reliability_guard(maximum_memory_bytes=None):
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.quit = None

    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None  # type: ignore

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def run_test(sample, test=None, debug=False, timeout=6):
    signal.signal(signal.SIGALRM, timeout_handler)
    reliability_guard()

    if debug:
        print(f"start = {datetime.now().time()}")

    try:
        in_outs = json.loads(sample.get("input_output", "{}"))
        if not isinstance(in_outs, dict):
            in_outs = {}
    except ValueError as e:
        raise e

    fn_name = in_outs.get("fn_name", None)
    if fn_name is None:
        which_type = CODE_TYPE.standard_input
        method_name = None
    else:
        which_type = CODE_TYPE.call_based
        method_name = fn_name

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if test is None:
        return [-4], {"error_code": -4, "error_message": "No test code provided"}

    if debug:
        print(f"loading test code = {datetime.now().time()}")

    if which_type == CODE_TYPE.call_based:
        signal.alarm(timeout)
        try:
            tmp = grade_call_based(
                code=test,
                all_inputs=in_outs.get("inputs", []) or [],
                all_outputs=in_outs.get("outputs", []) or [],
                fn_name=method_name,
                timeout=timeout,
            )
            if tmp is None:
                return [-4], {"error_code": -4, "error_message": "grade_call_based returned None"}
            results, metadata = tmp
            return results, metadata
        except Exception as e:
            return [-4], {"error_code": -4, "error_message": f"Error during testing: {e}"}
        finally:
            signal.alarm(0)

    signal.alarm(timeout)
    try:
        tmp = grade_stdio(
            code=test,
            all_inputs=in_outs.get("inputs", []) or [],
            all_outputs=in_outs.get("outputs", []) or [],
            timeout=timeout,
        )
        if tmp is None:
            return [-4], {"error_code": -4, "error_message": "grade_stdio returned None"}
        results, metadata = tmp
        return results, metadata
    except Exception as e:
        return [-4], {"error_code": -4, "error_message": f"Error during testing: {e}"}
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    sample = {
        "input_output": json.dumps({
            "fn_name": "add",
            "inputs": ["1\n2", "3\n4"],
            "outputs": ["3", "7"]
        })
    }
    
    test_code = """
def add(a, b):
    return a + b
"""
    
    results, metadata = run_test(sample, test_code, debug=True, timeout=10)
    print("Results:", results)
    print("Metadata:", metadata)
