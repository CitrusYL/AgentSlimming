import os
import re
import json
import base64
import zlib
import pickle
import asyncio
import multiprocessing
from typing import Any, Callable, Dict, List, Tuple

import aiofiles
import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from src.utils.logs import logger
from src.utils.lcb_runner import run_test

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CODE_FENCE_RE = re.compile(r"```(?:[\w+-]+)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
LIVECODE_PRIVATE_TESTS_ENV = "AGENT_SLIMMING_LIVECODE_USE_PRIVATE_TESTS"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}

_MAIN_GUARD_PATTERNS = [
    re.compile(
        r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*\n\s*solve\(\)\s*$',
        re.MULTILINE
    ),
    re.compile(
        r'^\s*if\s+globals\(\)\.get\(\s*[\'"]__name__[\'"]\s*,\s*[\'"]__main__[\'"]\s*\)\s*==\s*[\'"]__main__[\'"]\s*:\s*\n\s*solve\(\)\s*$',
        re.MULTILINE
    ),
]


def extractCode(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    blocks = CODE_FENCE_RE.findall(s)
    code = blocks[-1] if blocks else s
    return code.strip() + "\n"


def _normalize_io_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def _strip_autorun_for_stdin(code: str) -> str:
    out = code
    for pat in _MAIN_GUARD_PATTERNS:
        out = pat.sub("", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    return out


def decodeTests(blob: Any) -> List[Dict[str, Any]]:
    if not blob:
        return []
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        if isinstance(blob.get("tests"), list):
            return blob["tests"]
        return [blob]
    if not isinstance(blob, str):
        return []

    s = blob.strip()

    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                if isinstance(obj.get("tests"), list):
                    return obj["tests"]
                return [obj]
        except Exception:
            pass

    try:
        raw = zlib.decompress(base64.b64decode(s.encode("utf-8")))
        obj = pickle.loads(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if isinstance(obj.get("tests"), list):
                return obj["tests"]
            return [obj]
        if isinstance(obj, (bytes, bytearray)):
            obj2 = json.loads(obj.decode("utf-8"))
            return obj2 if isinstance(obj2, list) else [obj2]
        if isinstance(obj, str):
            obj2 = json.loads(obj)
            return obj2 if isinstance(obj2, list) else [obj2]
    except Exception:
        pass

    return []


def normPass(x: Any) -> bool:
    if isinstance(x, np.ndarray):
        x = x.item(0)
    if isinstance(x, np.bool_):
        x = bool(x)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return int(x) == 1
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "pass", "passed", "ok", "success"}
    if isinstance(x, dict):
        if "passed" in x:
            return bool(x["passed"])
        if "status" in x:
            return str(x["status"]).strip().upper() in {"PASS", "PASSED", "OK", "SUCCESS"}
    return False


def worker(smp, gen, out_res, out_meta, t):
    ret = run_test(smp, test=gen, debug=False, timeout=t)
    if ret is None:
        out_res.append([-4])
        out_meta.append({"error_code": -4, "error_message": "run_test returned None"})
        return
    res, meta = ret
    out_res.append(res)
    out_meta.append(meta)


def runTestInProc(sample: Dict[str, Any], code: str, timeout: int) -> Tuple[List[Any], Dict[str, Any]]:
    mgr = multiprocessing.Manager()
    out_res = mgr.list()
    out_meta = mgr.list()

    p = multiprocessing.Process(target=worker, args=(sample, code, out_res, out_meta, timeout))
    p.start()

    try:
        io_obj = json.loads(sample.get("input_output", "{}"))
        n_cases = max(1, len(io_obj.get("inputs", []) or []))
    except Exception:
        n_cases = 1

    p.join(timeout=(timeout + 1) * n_cases + 5)
    if p.is_alive():
        p.kill()
        p.join(timeout=1)

    if not out_res:
        return ([-1] * n_cases, {"error_code": -1, "error_message": "GlobalTimeout"})

    res0 = out_res[0]
    res_list = list(res0) if isinstance(res0, list) else [res0]
    meta0 = out_meta[0] if out_meta else {}
    return res_list, meta0


def cleanFnName(io_obj: Dict[str, Any]) -> Dict[str, Any]:
    fn = io_obj.get("fn_name", None)
    if fn is None:
        io_obj.pop("fn_name", None)
        return io_obj
    if isinstance(fn, str) and fn.strip() == "":
        io_obj.pop("fn_name", None)
        return io_obj
    if isinstance(fn, str) and fn.strip().lower() == "none":
        io_obj.pop("fn_name", None)
        return io_obj
    return io_obj


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _select_livecode_tests(
    source: Dict[str, Any],
    *,
    allow_private_tests: bool = False,
) -> List[Dict[str, Any]]:
    public_tests = decodeTests(source.get("public_test_cases"))
    if allow_private_tests and _env_flag(LIVECODE_PRIVATE_TESTS_ENV):
        return decodeTests(source.get("private_test_cases")) or public_tests
    return public_tests


class LiveCodeBench(BaseBenchmark):
    REQUIRED_FIELDS = ("question", "question_id", "input_output", "metadata")

    def __init__(self, name: str, file_path: str, log_path: str, timeout: int = 6):
        super().__init__(name, file_path, log_path)
        self.timeout = timeout

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "evaluation_details", "cost"]

    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        return 0.0, prediction

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        raw: List[dict] = []
        async with aiofiles.open(self.file_path, "r", encoding="utf-8") as f:
            async for line in f:
                raw.append(json.loads(line))

        data: List[dict] = []
        for item in raw:
            try:
                if "meta" in item and "problem" in item:
                    q = item.get("problem", "") or ""
                    meta = item.get("meta", {}) or {}
                    qid = str(meta.get("question_id") or meta.get("id") or "unknown")
                    level = item.get("level", "unknown")
                    platform = meta.get("platform", "unknown")
                    starter = meta.get("starter_code", "") or ""
                    tests = _select_livecode_tests(meta, allow_private_tests=True)
                    meta_blob = meta.get("metadata", None)
                else:
                    q = item.get("question_content", item.get("question", "")) or ""
                    qid = str(item.get("question_id", "unknown"))
                    level = item.get("difficulty", "unknown")
                    platform = item.get("platform", "unknown")
                    starter = item.get("starter_code", item.get("canonical_solution", "")) or ""
                    tests = _select_livecode_tests(item, allow_private_tests=True)
                    meta_blob = item.get("metadata", None)

                if not tests:
                    continue

                inputs = [_normalize_io_str(t.get("input", "")) for t in tests]
                outputs = [_normalize_io_str(t.get("output", "")) for t in tests]

                fn_name = None
                try:
                    meta_obj = json.loads(meta_blob) if isinstance(meta_blob, str) else meta_blob
                    if isinstance(meta_obj, dict):
                        fn_name = meta_obj.get("func_name") or meta_obj.get("fn_name")
                except Exception:
                    fn_name = None

                io_obj: Dict[str, Any] = {"inputs": inputs, "outputs": outputs}
                if isinstance(fn_name, str) and fn_name.strip() and fn_name.strip().lower() != "none":
                    io_obj["fn_name"] = fn_name.strip()

                data.append(
                    {
                        "question": q,
                        "question_id": qid,
                        "canonical_solution": starter,
                        "input_output": json.dumps(io_obj, ensure_ascii=False),
                        "metadata": {"difficulty": level, "platform": platform, "original_data": item},
                    }
                )
                self.validate_sample(data[-1])
            except Exception as e:
                logger.warning(f"[LCB] skip one item: {e}")

        if specific_indices is not None:
            return [data[i] for i in specific_indices if i < len(data)]
        return data

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception), reraise=True)
    async def generate_output(self, agent: Callable, prompt: str, entry_point: str, question_id: str = "") -> Tuple[str, float]:
        try:
            return await asyncio.wait_for(agent(prompt, entry_point, question_id), timeout=120)
        except TypeError:
            try:
                return await asyncio.wait_for(agent(prompt, entry_point), timeout=120)
            except TypeError:
                return await asyncio.wait_for(agent(prompt), timeout=120)

    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        q = problem["question"]
        qid = problem["question_id"]

        io_obj = json.loads(problem["input_output"])
        if not isinstance(io_obj, dict):
            raise ValueError("LiveCode input_output must decode to a dict")

        io_obj = cleanFnName(io_obj)
        fn_name = io_obj.get("fn_name", None)
        entry_point = fn_name or ""

        try:
            pred_raw, cost = await self.generate_output(agent, q, entry_point, qid)
            code = extractCode(pred_raw)

            if not fn_name:
                code = _strip_autorun_for_stdin(code)

            if ("AnswerFormatNode" in code and "执行失败" in code) or ("error" in code.lower() and "node_id" in code):
                details = {
                    "question_id": qid,
                    "fn_name": fn_name,
                    "execution_success": False,
                    "runner_meta": {"error_code": -10, "error_message": "formatter_failed", "raw": code[:500]},
                    "difficulty": problem.get("metadata", {}).get("difficulty", "unknown"),
                    "platform": problem.get("metadata", {}).get("platform", "unknown"),
                }
                expected = json.dumps(
                    {"question_id": qid, "difficulty": details["difficulty"], "platform": details["platform"], "fn_name": fn_name},
                    ensure_ascii=False,
                )
                self.log_mismatch(q, expected, code, code, extract_answer_code="LCB_FORMATTER_ERROR")
                return (q, code, expected, 0.0, details, float(cost or 0.0))

            sample = {
                "question": q,
                "question_id": qid,
                "input_output": json.dumps(io_obj, ensure_ascii=False),
            }

            case_results, runner_meta = runTestInProc(sample, code, self.timeout)
            ok_list = [normPass(x) for x in case_results]
            passed = (len(ok_list) > 0) and all(ok_list)
            score = 1.0 if passed else 0.0

            details = {
                "question_id": qid,
                "fn_name": fn_name,
                "case_results": case_results,
                "execution_success": passed,
                "runner_meta": runner_meta,
                "difficulty": problem.get("metadata", {}).get("difficulty", "unknown"),
                "platform": problem.get("metadata", {}).get("platform", "unknown"),
            }
            expected = json.dumps(
                {"question_id": qid, "difficulty": details["difficulty"], "platform": details["platform"], "fn_name": fn_name},
                ensure_ascii=False,
            )

            if not passed:
                logger.warning(f"[LCB] FAIL qid={qid} fn_name={fn_name} runner_meta={runner_meta} case_results={case_results}")
                self.log_mismatch(q, expected, code, code, extract_answer_code="LCB_RUN_TEST")

            return (q, code, expected, score, details, float(cost or 0.0))

        except Exception as e:
            details = {"question_id": qid, "execution_success": False, "runner_meta": {"error_code": -99, "error_message": repr(e)}}
            return (q, f"Evaluation error: {repr(e)}", "", 0.0, details, 0.0)
