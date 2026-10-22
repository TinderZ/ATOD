"""Regression checks for the experiment's wall-time estimator."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from opd_eta import estimate, jsonl, remaining_periodic


def row(step, seconds, **overheads):
    return dict(step=step, wall_time=1000,
                metrics={"timing_s/step": seconds, **overheads})


class ETATest(unittest.TestCase):
    def test_periodic_counts_include_final_once(self):
        self.assertEqual(remaining_periodic(10, 150, 5), 28)
        self.assertEqual(remaining_periodic(10, 150, 10), 14)
        self.assertEqual(remaining_periodic(10, 151, 10), 15)
        self.assertEqual(remaining_periodic(10, 150, -1), 0)

    def test_no_iterations_does_not_invent_eta(self):
        self.assertIsNone(estimate([], now=1000)["eta_beijing"])

    def test_validation_and_checkpoint_not_double_counted(self):
        result = estimate([row(10, 125, **{"timing_s/testing": 20,
                          "timing_s/save_checkpoint": 5})], now=1010)
        self.assertEqual(result["mean_training_iteration_seconds"], 100)
        self.assertAlmostEqual(result["remaining_hours"]*3600,
                               140*100 + 28*20 + 14*5 - 10)

    def test_unknown_checkpoint_is_labeled(self):
        result = estimate([row(1, 100)], initial_validation_seconds=20, now=1000)
        self.assertEqual(result["unmeasured_overheads"], ["checkpoint_save"])

    def test_warmup_excluded_after_four_iterations(self):
        result = estimate([row(1, 1000), row(2, 100), row(3, 100), row(4, 100)],
                          initial_validation_seconds=20, now=1000)
        self.assertEqual(result["mean_training_iteration_seconds"], 100)

    def test_final_iteration_has_zero_remaining(self):
        result = estimate([row(150, 125)], now=1000)
        self.assertEqual(result["remaining_hours"], 0)
        self.assertEqual(result["state"], "complete")

    def test_incomplete_final_json_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.jsonl"
            path.write_text(json.dumps(row(1, 100)) + "\n" + '{"step": 2')
            self.assertEqual(len(jsonl(path)), 1)


if __name__ == "__main__":
    unittest.main()
