import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.livecode import LiveCodeBench
from data.download import convert_livecode_example


class LiveCodePublicExportTests(unittest.TestCase):
    def test_convert_livecode_example_retains_private_tests(self) -> None:
        row = convert_livecode_example(
            {
                "question_title": "demo",
                "question_content": "body",
                "question_id": "q1",
                "difficulty": "easy",
                "starter_code": "",
                "public_test_cases": [{"input": "1\n", "output": "2\n"}],
                "private_test_cases": [{"input": "secret\n", "output": "secret\n"}],
                "metadata": {},
            },
            "test.jsonl",
        )

        self.assertIn("private_test_cases", row["meta"])
        self.assertEqual(
            row["meta"]["private_test_cases"],
            [{"input": "secret\n", "output": "secret\n"}],
        )

    def test_livecode_benchmark_defaults_to_public_tests_for_export_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            dataset_path = Path(tempdir) / "livecode_public.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "problem": "demo",
                        "type": "LiveCodeBench-CodeGen",
                        "level": "easy",
                        "meta": {
                            "question_id": "q1",
                            "starter_code": "",
                            "public_test_cases": [{"input": "public\n", "output": "ok\n"}],
                            "private_test_cases": [{"input": "private\n", "output": "secret\n"}],
                            "metadata": {},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            benchmark = LiveCodeBench("LiveCode", str(dataset_path), tempdir)
            data = asyncio.run(benchmark.load_data())

            self.assertEqual(len(data), 1)
            io_obj = json.loads(data[0]["input_output"])
            self.assertEqual(io_obj["inputs"], ["public\n"])
            self.assertEqual(io_obj["outputs"], ["ok\n"])
            self.assertIn("private_test_cases", data[0]["metadata"]["original_data"]["meta"])

    def test_livecode_benchmark_can_use_private_tests_for_export_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            dataset_path = Path(tempdir) / "livecode_public.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "problem": "demo",
                        "type": "LiveCodeBench-CodeGen",
                        "level": "easy",
                        "meta": {
                            "question_id": "q1",
                            "starter_code": "",
                            "public_test_cases": [{"input": "public\n", "output": "ok\n"}],
                            "private_test_cases": [{"input": "private\n", "output": "secret\n"}],
                            "metadata": {},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            benchmark = LiveCodeBench("LiveCode", str(dataset_path), tempdir)
            with patch.dict(os.environ, {"AGENT_SLIMMING_LIVECODE_USE_PRIVATE_TESTS": "1"}, clear=False):
                data = asyncio.run(benchmark.load_data())

            self.assertEqual(len(data), 1)
            io_obj = json.loads(data[0]["input_output"])
            self.assertEqual(io_obj["inputs"], ["private\n"])
            self.assertEqual(io_obj["outputs"], ["secret\n"])

    def test_livecode_benchmark_keeps_private_tests_available_for_raw_internal_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            dataset_path = Path(tempdir) / "livecode_raw.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "question": "demo",
                        "question_id": "q1",
                        "difficulty": "easy",
                        "platform": "demo",
                        "starter_code": "",
                        "public_test_cases": [{"input": "public\n", "output": "ok\n"}],
                        "private_test_cases": [{"input": "private\n", "output": "secret\n"}],
                        "metadata": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            benchmark = LiveCodeBench("LiveCode", str(dataset_path), tempdir)
            with patch.dict(os.environ, {"AGENT_SLIMMING_LIVECODE_USE_PRIVATE_TESTS": "1"}, clear=False):
                data = asyncio.run(benchmark.load_data())

            self.assertEqual(len(data), 1)
            io_obj = json.loads(data[0]["input_output"])
            self.assertEqual(io_obj["inputs"], ["private\n"])
            self.assertEqual(io_obj["outputs"], ["secret\n"])


if __name__ == "__main__":
    unittest.main()
