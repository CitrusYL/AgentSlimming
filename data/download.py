import csv
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import hf_hub_download, snapshot_download

from src.catalog.datasets import dataset_file_path

DATA_ROOT = Path("data/datasets")
TMP_ROOT = DATA_ROOT / "_tmp_download"
HF_CACHE_DIR = TMP_ROOT / "_hf_cache"

DERIVED_VALIDATE_RATIO = 0.3
DERIVED_SPLIT_SEED = 42
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}

AIME_REPO = "gneubig/aime-1983-2024"
LIVECODE_REPO = "livecodebench/code_generation_lite"
LIVECODE_FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl")
LIVECODE_PRIVATE_TESTS_ENV = "AGENT_SLIMMING_LIVECODE_USE_PRIVATE_TESTS"

MUSIQUE_REPO = "dgslibisey/MuSiQue"
MUSIQUE_FILES = (
    "musique_ans_v1.0_train.jsonl",
    "musique_ans_v1.0_dev.jsonl",
    "musique_ans_v1.0_test.jsonl",
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _is_retryable_hf_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    retry_signals = [
        "eof occurred in violation of protocol",
        "ssl",
        "tls",
        "connection reset by peer",
        "connection aborted",
        "connection broken",
        "temporary failure in name resolution",
        "read timed out",
        "max retries exceeded",
        "503",
        "502",
        "504",
        "connection error",
        "proxy error",
    ]
    return any(s in msg for s in retry_signals)


def safe_hf_hub_download(
    repo_id: str,
    filename: str,
    repo_type: str = "dataset",
    local_dir: str | None = None,
    retries: int = 8,
) -> Path:
    last_err: Exception | None = None
    ensure_dir(HF_CACHE_DIR)

    for attempt in range(1, retries + 1):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                local_dir=local_dir,
                cache_dir=str(HF_CACHE_DIR),
                token=os.environ.get("HF_TOKEN"),
                force_download=False,
                etag_timeout=60,
            )
            return Path(path)
        except Exception as e:
            last_err = e
            if attempt == retries or not _is_retryable_hf_error(e):
                raise
            sleep_s = min(60, 2 ** attempt)
            print(
                f"[RETRY] hf_hub_download failed for {repo_id}/{filename}: "
                f"{e} | retry in {sleep_s}s ({attempt}/{retries})"
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Unreachable state in safe_hf_hub_download: {last_err}")


def safe_snapshot_download(
    repo_id: str,
    repo_type: str = "dataset",
    local_dir: str | None = None,
    retries: int = 6,
) -> Path:
    last_err: Exception | None = None
    ensure_dir(HF_CACHE_DIR)

    for attempt in range(1, retries + 1):
        try:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=local_dir,
                cache_dir=str(HF_CACHE_DIR),
                token=os.environ.get("HF_TOKEN"),
                max_workers=2,
                etag_timeout=60,
            )
            return Path(path)
        except Exception as e:
            last_err = e
            
            if attempt == retries or not _is_retryable_hf_error(e):
                raise
            sleep_s = min(60, 2 ** attempt)
            print(
                f"[RETRY] snapshot_download failed for {repo_id}: "
                f"{e} | retry in {sleep_s}s ({attempt}/{retries})"
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Unreachable state in safe_snapshot_download: {last_err}")


def outputs_ready(dataset: str) -> bool:
    validate_path = dataset_file_path(dataset, "validate")
    test_path = dataset_file_path(dataset, "test")
    return (
        validate_path.exists()
        and test_path.exists()
        and validate_path.stat().st_size > 0
        and test_path.stat().st_size > 0
    )


def write_dataset_splits(
    dataset: str,
    validate_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    if not validate_rows:
        raise ValueError(f"{dataset} validate split is empty")
    if not test_rows:
        raise ValueError(f"{dataset} test split is empty")

    validate_path = dataset_file_path(dataset, "validate")
    test_path = dataset_file_path(dataset, "test")
    write_jsonl(validate_path, validate_rows)
    write_jsonl(test_path, test_rows)
    return validate_path, test_path


def split_derived_validate(
    rows: list[dict[str, Any]],
    ratio: float = DERIVED_VALIDATE_RATIO,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 2:
        raise ValueError("Need at least two rows to derive validate/test splits")
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")

    shuffled = list(rows)
    random.Random(DERIVED_SPLIT_SEED).shuffle(shuffled)
    validate_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * ratio))))
    validate_rows = shuffled[:validate_count]
    test_rows = shuffled[validate_count:]
    return validate_rows, test_rows


def try_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "{[" or text[-1] not in "}]":
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUTHY_ENV_VALUES


def as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def maybe_skip(dataset: str) -> tuple[bool, Path, Path]:
    validate_path = dataset_file_path(dataset, "validate")
    test_path = dataset_file_path(dataset, "test")
    if outputs_ready(dataset):
        print(f"[SKIP] {dataset} -> {validate_path.name} | {test_path.name}")
        return True, validate_path, test_path
    return False, validate_path, test_path


def download_aime() -> tuple[Path, Path]:
    skipped, validate_path, test_path = maybe_skip("AIME")
    if skipped:
        return validate_path, test_path

    snapshot_dir = TMP_ROOT / "aime"
    remove_tree(snapshot_dir)
    ensure_dir(snapshot_dir)

    safe_snapshot_download(
        repo_id=AIME_REPO,
        repo_type="dataset",
        local_dir=str(snapshot_dir),
    )

    csv_files = sorted(snapshot_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {snapshot_dir}")

    rows: list[dict[str, Any]] = []
    with csv_files[0].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            year = as_int(row.get("Year"))
            rows.append(
                {
                    "problem": str(row.get("Question", "")).strip(),
                    "level": f"AIME {year}" if year else "AIME",
                    "type": "AIME",
                    "solution": str(row.get("Answer", "")).strip(),
                    "meta": {
                        "id": str(row.get("ID", "")).strip(),
                        "year": year,
                        "problem_number": as_int(row.get("Problem Number")),
                        "part": row.get("Part"),
                        "source_hf": AIME_REPO,
                        "source_split": "full_corpus",
                        "split_policy": "derived_validate_from_full_corpus",
                    },
                }
            )

    validate_rows, test_rows = split_derived_validate(rows)
    result = write_dataset_splits("AIME", validate_rows, test_rows)
    remove_tree(snapshot_dir)
    return result


def convert_livecode_example(example: dict[str, Any], source_file: str) -> dict[str, Any]:
    title = str(example.get("question_title") or "").strip()
    content = str(example.get("question_content") or "").strip()
    problem = f"{title}\n\n{content}".strip() if title else content
    private_tests = try_json_loads(example.get("private_test_cases", ""))
    meta = {
        "question_id": example.get("question_id"),
        "platform": example.get("platform"),
        "contest_id": example.get("contest_id"),
        "contest_date": example.get("contest_date"),
        "starter_code": example.get("starter_code"),
        "public_test_cases": try_json_loads(example.get("public_test_cases", "")),
        "metadata": try_json_loads(example.get("metadata", "")),
        "source_hf": LIVECODE_REPO,
        "source_file": source_file,
        "source_split": "official_test_release",
        "split_policy": "derived_validate_from_release_tests",
    }
    if private_tests:
        meta["private_test_cases"] = private_tests
    return {
        "problem": problem,
        "type": "LiveCodeBench-CodeGen",
        "level": example.get("difficulty", "unknown"),
        "solution": "",
        "meta": meta,
    }


def download_livecode() -> tuple[Path, Path]:
    skipped, validate_path, test_path = maybe_skip("LiveCode")
    if skipped:
        return validate_path, test_path

    raw_dir = TMP_ROOT / "livecode"
    remove_tree(raw_dir)
    ensure_dir(raw_dir)

    rows: list[dict[str, Any]] = []
    for filename in LIVECODE_FILES:
        local_path = safe_hf_hub_download(
            repo_id=LIVECODE_REPO,
            repo_type="dataset",
            filename=filename,
            local_dir=str(raw_dir),
        )
        with local_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    rows.append(convert_livecode_example(json.loads(text), local_path.name))

    validate_rows, test_rows = split_derived_validate(rows)
    result = write_dataset_splits("LiveCode", validate_rows, test_rows)
    remove_tree(raw_dir)
    return result


def musique_hops(example_id: str) -> str:
    if example_id.startswith("2hop__"):
        return "2-hop"
    if example_id.startswith("3hop__"):
        return "3-hop"
    if example_id.startswith("4hop__"):
        return "4-hop"
    return "unknown"


def build_musique_problem(question: str, paragraphs: Any) -> str:
    lines = [question.strip(), "", "Context:"]
    for index, paragraph in enumerate(paragraphs if isinstance(paragraphs, list) else []):
        if not isinstance(paragraph, dict):
            continue
        title = str(paragraph.get("title", "")).strip()
        text = str(paragraph.get("paragraph_text", "")).strip()
        tag = paragraph.get("idx", index)
        header = f"[{tag}] {title}".strip()
        if header:
            lines.append(header)
        if text:
            lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def download_musique_ans() -> tuple[Path, Path]:
    skipped, validate_path, test_path = maybe_skip("MusiqueAns")
    if skipped:
        return validate_path, test_path

    raw_dir = TMP_ROOT / "musiqueans"
    remove_tree(raw_dir)
    ensure_dir(raw_dir)

    local_files: list[Path] = []
    for filename in MUSIQUE_FILES:
        try:
            local_path = safe_hf_hub_download(
                repo_id=MUSIQUE_REPO,
                repo_type="dataset",
                filename=filename,
                local_dir=str(raw_dir),
            )
            local_files.append(local_path)
        except Exception:
            continue

    if not local_files:
        raise FileNotFoundError(f"No MuSiQue-Ans files downloaded from {MUSIQUE_REPO}.")

    rows: list[dict[str, Any]] = []
    for file_path in local_files:
        source_split = (
            "train" if "train" in file_path.name
            else "dev" if "dev" in file_path.name
            else "test" if "test" in file_path.name
            else "unknown"
        )

        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    example = json.loads(text)
                except Exception:
                    continue

                example_id = str(example.get("id", "")).strip()
                paragraphs = example.get("paragraphs", [])
                rows.append(
                    {
                        "problem": build_musique_problem(str(example.get("question", "")).strip(), paragraphs),
                        "type": "MuSiQue-Ans",
                        "level": musique_hops(example_id),
                        "solution": str(example.get("answer", "")).strip(),
                        "meta": {
                            "id": example_id,
                            "answer_aliases": example.get("answer_aliases", []),
                            "answerable": example.get("answerable", True),
                            "question_decomposition": example.get("question_decomposition", []),
                            "paragraphs": paragraphs,
                            "source_hf": MUSIQUE_REPO,
                            "source_file": file_path.name,
                            "source_split": source_split,
                        },
                    }
                )

    validate_rows, test_rows = split_derived_validate(rows)
    result = write_dataset_splits("MusiqueAns", validate_rows, test_rows)
    remove_tree(raw_dir)
    return result


def main() -> None:
    ensure_dir(DATA_ROOT)
    ensure_dir(TMP_ROOT)
    ensure_dir(HF_CACHE_DIR)

    try:
        aime_validate, aime_test = download_aime()
        print(f"[OK] AIME -> {aime_validate.name} | {aime_test.name}")

        livecode_validate, livecode_test = download_livecode()
        print(f"[OK] LiveCode -> {livecode_validate.name} | {livecode_test.name}")

        musique_validate, musique_test = download_musique_ans()
        print(f"[OK] MuSiQue-Ans -> {musique_validate.name} | {musique_test.name}")

    finally:
        remove_tree(TMP_ROOT)


if __name__ == "__main__":
    main()
