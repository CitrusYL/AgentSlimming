# Data

This page is the source of truth for dataset files, downloader behavior, split
policy, and reporting caveats. Legal/provenance notices for the git-tracked
fixtures live in [../DATASET_NOTICES.md](../DATASET_NOTICES.md).

## Expected Files

Evaluation expects JSONL files under `data/datasets/`:

```text
data/datasets/<dataset_lowercase>_validate.jsonl
data/datasets/<dataset_lowercase>_test.jsonl
```

Examples:

```text
data/datasets/math_validate.jsonl
data/datasets/math_test.jsonl
data/datasets/musiqueans_validate.jsonl
data/datasets/musiqueans_test.jsonl
```

Dataset path resolution is centralized in `src/catalog/datasets.py`.

## Naming Notes

- CLI dataset names come directly from `src/catalog/datasets.py`.
- The CLI key is `MusiqueAns`; this document also uses the upstream benchmark spelling `MuSiQue-Ans` when discussing source data and reporting caveats.
- `MATHDEMO` is a repository-local alias of `MATH`; it reuses the same `math_*.jsonl` files and only changes the tracked demo workspace under `workspace/MATHDEMO/`.

## Provisioning Model

The repository uses two dataset paths:

- Small benchmark files are tracked directly in git under `data/datasets/`.
- Large benchmark files are downloaded from Hugging Face and converted into the repository JSONL format by `data/download.py`.

`.gitignore` mirrors this policy by ignoring `data/datasets/*` by default and only allowlisting the git-managed fixtures listed below.

### Git-Managed Benchmark Assets

These files are expected to be committed and reviewed in git. They are
repository-local experiment fixtures inherited from the earlier AFlow
evaluation setup, kept for fair comparison with prior AFlow-style experiments.
They are not the full original upstream datasets, and the selection/splitting
rules were not newly defined by AgentSlimming. See
[../DATASET_NOTICES.md](../DATASET_NOTICES.md) for dataset-specific provenance
and license notices.

- `DROP`: `drop_validate.jsonl`, `drop_test.jsonl`
- `GSM8K`: `gsm8k_validate.jsonl`, `gsm8k_test.jsonl`
- `HotpotQA`: `hotpotqa_validate.jsonl`, `hotpotqa_test.jsonl`
- `MATH`: `math_validate.jsonl`, `math_test.jsonl`
- `MATHDEMO`: reuses the `MATH` files above; it does not introduce separate dataset JSONL files
- `HumanEval`: `humaneval_validate.jsonl`, `humaneval_test.jsonl`, `humaneval_public_test.jsonl`
- `MBPP`: `mbpp_validate.jsonl`, `mbpp_test.jsonl`, `mbpp_public_test.jsonl`

### Downloaded Benchmark Assets

These files are materialized locally by `python -m data.download` and should stay out of git:

- `AIME`: `aime_validate.jsonl`, `aime_test.jsonl`
- `LiveCode`: `livecode_validate.jsonl`, `livecode_test.jsonl`
- `MusiqueAns` (`MuSiQue-Ans`): `musiqueans_validate.jsonl`, `musiqueans_test.jsonl`

Currently, the downloader only handles:

- `AIME`
- `LiveCode`
- `MusiqueAns`

Run it from the repository root:

```bash
python -m data.download
```

Downloader behavior:

- Skips a dataset when both target JSONL files already exist.
- Uses a temporary cache under `data/datasets/_tmp_download/`.
- Respects `HF_TOKEN` for authenticated Hugging Face access.

If you add a new benchmark fixture, update both this document and `.gitignore` in the same change so the publication policy stays explicit.

## Large-Dataset Split Policy

- `AIME`: derives `validate`/`test` from the downloaded full corpus with a deterministic split. This is a repository-local split, not an official benchmark split.
- `LiveCode`: derives `validate`/`test` from the published release test files with a deterministic split. This is a repository-local split, not the official LiveCodeBench reporting protocol. The derived rows retain `public_test_cases` and `private_test_cases` when the source provides them, but evaluation uses public tests by default. Set `AGENT_SLIMMING_LIVECODE_USE_PRIVATE_TESTS=1` only for local/internal workflows that intentionally switch to private tests, not for public reporting.
- `MusiqueAns` (`MuSiQue-Ans`): downloads the available train/dev/test JSONL files from Hugging Face, converts them into the repository format, and derives `validate`/`test` with a deterministic split over the downloaded rows.

If you need a different split policy, edit `data/download.py` or prepare the target JSONL files manually.

## Relevant Environment Variables

- `HF_TOKEN`: optional Hugging Face token used by `python -m data.download` when authenticated dataset access is needed.
- `AGENT_SLIMMING_LIVECODE_USE_PRIVATE_TESTS=1`: when a LiveCode row contains both `public_test_cases` and `private_test_cases`, evaluation uses the private tests instead of the public tests. The default is unset/off, which keeps evaluation on public tests. Do not enable this for public reporting or leaderboard-style comparisons.

## Public Reporting Guidance

The following integrations are convenient local experiment paths. They should not be presented as official benchmark-compatible results unless you rerun the official evaluation pipeline for the underlying benchmark.

- `AIME`: the Hugging Face source used here is a single corpus, not an official train/dev/test benchmark split. This repository derives deterministic repository-local `validate`/`test` files from that corpus.
- `MusiqueAns` (`MuSiQue-Ans`): the official benchmark already ships train/dev/test splits and an official evaluation script. This repository currently downloads those files, converts them into the local schema, and then derives repository-local `validate`/`test` files for the optimizer pipeline. Numbers from this path are not directly comparable to official MuSiQue reports or leaderboard submissions.
- `LiveCode`: official LiveCodeBench reporting uses benchmark releases, release-date windows, and `pass@1` / `pass@5` metrics. This repository instead derives repository-local `validate`/`test` files from released data and evaluates persisted public tests by default. These results are not directly comparable to the official LiveCodeBench leaderboard.

If you publish numbers produced by this repository:

- explicitly label repository-local splits or repository-local protocols;
- do not compare `AIME`, `MusiqueAns` / `MuSiQue-Ans`, or `LiveCode` numbers here against official papers or leaderboards without rerunning the official benchmark setup;
- cite the upstream benchmark sources and explain any protocol differences.

Official references used to verify the guidance above:

- `AIME` source corpus used here: <https://huggingface.co/datasets/gneubig/aime-1983-2024>
- `MuSiQue`: paper <https://doi.org/10.1162/tacl_a_00475>, official repo <https://github.com/stonybrooknlp/musique>
- `LiveCodeBench`: official repo <https://github.com/LiveCodeBench/LiveCodeBench>, leaderboard <https://livecodebench.github.io/leaderboard.html>

## Adding A Benchmark

To add a new dataset integration, keep the benchmark implementation, dataset catalog,
and publication policy in sync:

1. Implement a benchmark class under `benchmarks/` by subclassing `BaseBenchmark`.
2. Set `REQUIRED_FIELDS` and implement `evaluate_problem()`, `calculate_score()`, and `get_result_columns()`.
3. Override `load_data()` only when the raw dataset schema needs adaptation before evaluation.
4. Register the dataset in `src/catalog/datasets.py` with its file stem, question type, operators, and benchmark module/class.
5. Add the dataset files or downloader logic under `data/`, then update this document and `.gitignore` in the same change.
6. Add or update tests for any publication-policy, split-policy, or benchmark-specific behavior.

Use `benchmarks/drop.py` as the simplest reference for a standard JSONL-backed
benchmark class. Use `benchmarks/livecode.py` as the reference when a benchmark
needs custom `load_data()` logic or schema conversion before evaluation.
