import tempfile
import unittest

from benchmarks.aime import AIMEBenchmark


class AIMEBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.benchmark = AIMEBenchmark(
            name="AIME",
            file_path="unused.jsonl",
            log_path=self.tempdir.name,
        )

    def test_calculate_score_accepts_multiple_gold_answers(self) -> None:
        self.assertEqual(
            self.benchmark.calculate_score("080 or 081 (both were accepted)", r"\boxed{80}"),
            (1, "80"),
        )
        self.assertEqual(
            self.benchmark.calculate_score("080 or 081 (both were accepted)", r"\boxed{81}"),
            (1, "81"),
        )

    def test_calculate_score_rejects_non_accepted_answer(self) -> None:
        self.assertEqual(
            self.benchmark.calculate_score("080 or 081 (both were accepted)", r"\boxed{79}"),
            (0, "79"),
        )


if __name__ == "__main__":
    unittest.main()
