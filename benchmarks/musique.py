import inspect
import json
import re
import string
from typing import Any, Callable, Dict, List, Tuple
from benchmarks.benchmark import BaseBenchmark

class MusiqueAnsBenchmark(BaseBenchmark):
    REQUIRED_FIELDS = ("problem", "solution", "meta")

    def __init__(self, file_path: str, log_path: str, name: str = "MuSiQue-Ans"):
        super().__init__(name=name, file_path=file_path, log_path=log_path)

    def get_result_columns(self) -> List[str]:
        return [
            "id",
            "level",
            "score",
            "em",
            "cost",
            "matched_gold",
            "gold_all",
            "pred_answer",
            "raw_prediction",
        ]

    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        pid = self._safe_str(self._get(problem, ["meta", "id"], default=""))
        level = self._safe_str(problem.get("level", "unknown"))
        prob_text = self._safe_str(problem.get("problem", ""))
        expected = self._safe_str(problem.get("solution", ""))
        aliases = self._get(problem, ["meta", "answer_aliases"], default=[])
        golds = self._build_golds(expected, aliases)

        raw_pred, cost = await self._call_agent(agent, problem, prob_text)
        pred_answer = self._extract_answer(raw_pred)

        score, info = self.calculate_score({"golds": golds}, pred_answer)
        em = float(info.get("em", 0))
        matched_gold = self._safe_str(info.get("matched_gold", ""))

        if int(em) == 0:
            self.log_mismatch(
                problem=prob_text,
                expected_output=golds[0] if golds else "",
                prediction=self._safe_str(raw_pred),
                extracted_output=pred_answer,
                extract_answer_code=info.get("extract_answer_code", "auto"),
            )

        return (
            pid or problem.get("idx", ""),
            level,
            float(score),
            float(em),
            float(cost),
            matched_gold,
            json.dumps(golds, ensure_ascii=False),
            pred_answer,
            self._safe_str(raw_pred),
        )

    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        golds = expected_output.get("golds", []) if isinstance(expected_output, dict) else []
        pred = self._safe_str(prediction)

        best_em = 0
        best_gold = golds[0] if golds else ""

        for g in golds:
            em = 1 if self._normalize(pred) == self._normalize(g) else 0
            if em > best_em:
                best_em = em
                best_gold = g
                if best_em == 1:
                    break

        info = {
            "em": best_em,
            "matched_gold": best_gold,
            "extract_answer_code": "auto",
        }
        return float(best_em), info

    async def _call_agent(self, agent: Callable, problem_obj: dict, problem_text: str) -> Tuple[str, float]:
        async def _await_if_needed(x: Any) -> Any:
            if inspect.isawaitable(x):
                return await x
            return x

        ret = await _await_if_needed(agent(problem_text))

        if isinstance(ret, (tuple, list)) and len(ret) >= 1:
            pred = ret[0]
            cost = ret[1] if len(ret) >= 2 else 0.0
            return self._safe_str(pred), self._safe_float(cost)

        if isinstance(ret, dict):
            cost = ret.get("cost", ret.get("total_cost", ret.get("price", 0.0)))
            pred = (
                ret.get("answer")
                or ret.get("prediction")
                or ret.get("output")
                or ret.get("text")
                or ret.get("response")
                or ""
            )
            return self._safe_str(pred), self._safe_float(cost)

        return self._safe_str(ret), 0.0

    def _extract_answer(self, raw: Any) -> str:
        s = self._safe_str(raw).strip()
        if not s:
            return ""

        js = self._try_parse_json(s)
        if isinstance(js, dict):
            ans = js.get("answer", js.get("final_answer", js.get("prediction", "")))
            return self._postprocess_answer(self._safe_str(ans))

        if "\n" in s:
            first = s.splitlines()[0].strip()
            cand = first if first else s.strip()
        else:
            cand = s

        cand = re.sub(r"^(final answer|answer)\s*[:：]\s*", "", cand.strip(), flags=re.IGNORECASE)
        return self._postprocess_answer(cand)

    def _postprocess_answer(self, s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def _build_golds(self, expected: str, aliases: Any) -> List[str]:
        golds: List[str] = []
        if expected.strip():
            golds.append(expected.strip())
        if isinstance(aliases, list):
            for a in aliases:
                if isinstance(a, str) and a.strip():
                    golds.append(a.strip())
        seen = set()
        out: List[str] = []
        for g in golds:
            k = self._normalize(g)
            if k and k not in seen:
                seen.add(k)
                out.append(g)
        return out if out else [""]

    def _normalize(self, s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        s = " ".join(s.split())
        return s

    def _try_parse_json(self, s: str) -> Any:
        ss = s.strip()
        if not ss:
            return None
        if not ((ss.startswith("{") and ss.endswith("}")) or (ss.startswith("[") and ss.endswith("]"))):
            return None
        try:
            return json.loads(ss)
        except Exception:
            return None

    def _get(self, obj: Any, path: List[str], default: Any = None) -> Any:
        cur = obj
        for k in path:
            if not isinstance(cur, dict):
                return default
            if k not in cur:
                return default
            cur = cur[k]
        return cur

    def _safe_str(self, x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        try:
            return str(x)
        except Exception:
            return ""

    def _safe_float(self, x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0
