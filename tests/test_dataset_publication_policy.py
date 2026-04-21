import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_NOTICES_PATH = REPO_ROOT / "DATASET_NOTICES.md"

GIT_MANAGED_DATASET_FILES = (
    "data/datasets/drop_validate.jsonl",
    "data/datasets/drop_test.jsonl",
    "data/datasets/gsm8k_validate.jsonl",
    "data/datasets/gsm8k_test.jsonl",
    "data/datasets/hotpotqa_validate.jsonl",
    "data/datasets/hotpotqa_test.jsonl",
    "data/datasets/math_validate.jsonl",
    "data/datasets/math_test.jsonl",
    "data/datasets/humaneval_validate.jsonl",
    "data/datasets/humaneval_test.jsonl",
    "data/datasets/humaneval_public_test.jsonl",
    "data/datasets/mbpp_validate.jsonl",
    "data/datasets/mbpp_test.jsonl",
    "data/datasets/mbpp_public_test.jsonl",
)

DOWNLOADED_DATASET_FILES = (
    "data/datasets/aime_validate.jsonl",
    "data/datasets/aime_test.jsonl",
    "data/datasets/livecode_validate.jsonl",
    "data/datasets/livecode_test.jsonl",
    "data/datasets/musiqueans_validate.jsonl",
    "data/datasets/musiqueans_test.jsonl",
    "data/datasets/_tmp_download/demo.jsonl",
)


class DatasetPublicationPolicyTests(unittest.TestCase):
    def assert_ignore_state(self, path: str, should_ignore: bool) -> None:
        result = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=REPO_ROOT,
            check=False,
        )
        expected_code = 0 if should_ignore else 1
        state = "ignored" if should_ignore else "tracked-eligible"
        self.assertEqual(result.returncode, expected_code, f"{path} should be {state}")

    def test_git_managed_dataset_files_are_not_ignored(self) -> None:
        for path in GIT_MANAGED_DATASET_FILES:
            with self.subTest(path=path):
                self.assert_ignore_state(path, should_ignore=False)

    def test_downloaded_dataset_files_are_ignored(self) -> None:
        for path in DOWNLOADED_DATASET_FILES:
            with self.subTest(path=path):
                self.assert_ignore_state(path, should_ignore=True)

    def test_git_managed_dataset_files_are_documented_in_dataset_notices(self) -> None:
        self.assertTrue(DATASET_NOTICES_PATH.exists(), "DATASET_NOTICES.md should exist")
        contents = DATASET_NOTICES_PATH.read_text(encoding="utf-8")
        for path in GIT_MANAGED_DATASET_FILES:
            with self.subTest(path=path):
                self.assertIn(path, contents, f"{path} should be documented in DATASET_NOTICES.md")


if __name__ == "__main__":
    unittest.main()
