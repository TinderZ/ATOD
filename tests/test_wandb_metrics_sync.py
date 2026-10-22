"""Privacy and append/resume checks for the independent metric mirror."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sync_opd_wandb_metrics import complete_records, mirror_records, privacy_settings, progress_heartbeat, scalar_payload, supervisor_alive


class MetricsMirrorTest(unittest.TestCase):
    def test_only_finite_numeric_training_metrics_are_selected(self):
        payload = scalar_payload(dict(step=2, metrics={
            "actor/pg_loss": .5, "episode/success_rate": .25,
            "timing_s/step": 90, "val/success_rate": .1,
            "actor/example": "PRIVATE RESPONSE", "actor/table": [1, 2],
            "actor/nan": float("nan"), "actor/infinity": float("inf"),
            "actor/flag": True, "prompt": "PRIVATE PROMPT", "secret/count": 12,
        }))
        self.assertEqual(payload, {"actor/pg_loss": .5, "episode/success_rate": .25,
                                   "timing_s/step": 90, "val/success_rate": .1,
                                   "training/global_step": 2})

    def test_partial_final_record_is_read_after_writer_finishes_it(self):
        first = json.dumps(dict(step=0, metrics={})).encode() + b"\n"
        second = json.dumps(dict(step=1, metrics={"actor/pg_loss": .5})).encode()
        stream = io.BytesIO(first + second[:20])
        self.assertEqual([r["step"] for r in complete_records(stream)], [0])
        position = stream.tell()
        stream.seek(0, 2)
        stream.write(second[20:] + b"\n")
        stream.seek(position)
        self.assertEqual([r["step"] for r in complete_records(stream)], [1])

    def test_resume_uses_remote_step_and_never_relogs_older_updates(self):
        data = b"".join(json.dumps(dict(step=i, metrics={"actor/pg_loss": .5})).encode()
                         + b"\n" for i in range(4))
        remote = MagicMock()
        next_step, count = mirror_records(io.BytesIO(data), remote, next_step=3)
        self.assertEqual((next_step, count), (4, 1))
        self.assertEqual(remote.log.call_args.kwargs, {"commit": True})
        self.assertEqual(remote.log.call_args.args[0]["training/global_step"], 3)

    def test_progress_heartbeat_keeps_actual_training_step_without_loss(self):
        remote = MagicMock()
        progress_heartbeat(remote, 4)
        remote.log.assert_called_once_with({"training/global_step": 3}, commit=True)

    def test_privacy_disables_automatic_uploads(self):
        sdk = MagicMock()
        privacy_settings(sdk, Path("/private/local-only"), "online")
        settings = sdk.Settings.call_args.kwargs
        for key in ["disable_code", "disable_git", "disable_job_creation",
                    "x_disable_meta", "x_disable_stats", "x_disable_machine_info"]:
            self.assertIs(settings[key], True)
        self.assertIs(settings["x_save_requirements"], False)
        self.assertIs(settings["save_code"], False)
        self.assertEqual(settings["console"], "off")
        self.assertEqual(settings["ignore_globs"], ("*",))
        self.assertEqual(settings["base_url"], "https://api.wandb.ai")

    def test_invalid_step_does_not_get_uploaded(self):
        for step in [-1, True, "2", 2.5]:
            with self.assertRaises(ValueError):
                scalar_payload(dict(step=step, metrics={}))

    def test_missing_supervisor_is_detected_and_reused_pid_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            self.assertIsNone(supervisor_alive(run))
            (run / "supervisor.pid").write_text("999999999")
            self.assertFalse(supervisor_alive(run))
            (run / "supervisor.pid").write_text(str(os.getpid()))
            self.assertFalse(supervisor_alive(run))


if __name__ == "__main__":
    unittest.main()
