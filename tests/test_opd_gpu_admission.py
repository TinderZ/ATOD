"""Device query validation and races before OPD admission; no GPU allocation."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import opd_gpu_admission as admission


class DeviceQueryTests(unittest.TestCase):
    def query(self, before, gpu, after):
        replies = [subprocess.CompletedProcess([], 0, stdout=value) for value in (before, gpu, after)]
        with patch.object(admission.subprocess, "run", side_effect=replies):
            return admission.query_gpus()

    def test_device_becoming_occupied_during_query_cannot_be_admitted(self):
        rows = self.query("", "12, GPU-test, NVIDIA A100, 81920, 0, 0\n",
                          "GPU-test, 1234, 512\n")
        self.assertFalse(admission.idle(rows[12]))
        self.assertEqual(rows[12]["pids"], [1234])

    def test_empty_process_table_allows_idle_device(self):
        rows = self.query("", "0, GPU-test, NVIDIA A100, 81920, 50, 0\n", "")
        self.assertTrue(admission.idle(rows[0]))

    def test_malformed_process_data_fails_closed(self):
        with self.assertRaises(RuntimeError):
            self.query("not a process table", "0, GPU-test, A100, 81920, 0, 0\n", "")

    def test_duplicate_device_identity_fails_closed(self):
        with self.assertRaises(RuntimeError):
            self.query("", "0, GPU-test, A100, 81920, 0, 0\n1, GPU-test, A100, 81920, 0, 0\n", "")

    def test_cooperative_lock_is_exclusive_and_released_on_close(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, ATOD_GPU_LOCK_DIR=folder):
            first = admission.acquire_lock("GPU-test")
            self.assertIsNotNone(first)
            try:
                self.assertIsNone(admission.acquire_lock("GPU-test"))
            finally:
                first.close()
            again = admission.acquire_lock("GPU-test")
            self.assertIsNotNone(again)
            again.close()


if __name__ == "__main__":
    unittest.main()
