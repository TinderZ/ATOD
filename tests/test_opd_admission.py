"""Admission, migration and resume behavior for two continuously idle devices."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_alfworld_vanilla_opd as launcher

POOL = list(range(8))


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.clock = 1000.
        self.preflight = {"initial_devices": {str(i): {"uuid": f"GPU-test-{i}"} for i in POOL}}
        self.idle = {i: dict(index=i, uuid=f"GPU-test-{i}", pids=[],
                             used_mib=100, utilization=0) for i in POOL}
        self.busy = {i: dict(d, pids=[123], used_mib=20000) for i, d in self.idle.items()}

    def devices(self, *idle_indices):
        return {i: self.idle[i] if i in idle_indices else self.busy[i] for i in POOL}

    def sleep(self, seconds):
        self.clock += seconds

    def run_wait(self, query, acquire, wait_hours=.2, stop=False, pool=POOL, idle_seconds=300):
        with patch.object(launcher, "query_gpus", side_effect=query), \
             patch.object(launcher, "acquire_lock", side_effect=acquire), \
             patch.object(launcher, "write_json"), \
             patch.object(launcher, "STOP", stop), \
             patch.object(launcher.time, "time", side_effect=lambda: self.clock), \
             patch.object(launcher.time, "monotonic", side_effect=lambda: self.clock), \
             patch.object(launcher.time, "sleep", side_effect=self.sleep), \
             patch("builtins.print"):
            return launcher.wait_for_devices(Path("/unused"), self.preflight, wait_hours, pool, idle_seconds)

    def test_explicit_pair_starts_immediately_when_idle(self):
        locks = [MagicMock(), MagicMock()]
        selected, result = self.run_wait(lambda: self.devices(2, 3, 4, 5),
                                         MagicMock(side_effect=locks), pool=[4, 5], idle_seconds=0)
        self.assertEqual(selected, [4, 5])
        self.assertEqual(result, locks)
        self.assertEqual(self.clock, 1000.)

    def test_immediate_launch_refuses_busy_pair_without_waiting(self):
        acquire = MagicMock()
        with self.assertRaises(RuntimeError):
            self.run_wait(lambda: self.devices(2, 3, 4), acquire, pool=[4, 5], idle_seconds=0)
        acquire.assert_not_called()
        self.assertEqual(self.clock, 1000.)

    def test_immediate_launch_rechecks_pair_and_releases_locks(self):
        locks = [MagicMock(), MagicMock()]
        with self.assertRaises(RuntimeError):
            self.run_wait(iter([self.devices(4, 5), self.busy]).__next__,
                          MagicMock(side_effect=locks), pool=[4, 5], idle_seconds=0)
        for lock in locks:
            lock.close.assert_called_once()
        self.assertEqual(self.clock, 1000.)

    def test_busy_pool_is_never_admitted(self):
        acquire = MagicMock()
        with self.assertRaises(RuntimeError):
            self.run_wait(lambda: self.busy, acquire)
        acquire.assert_not_called()

    def test_one_idle_card_is_insufficient(self):
        acquire = MagicMock()
        with self.assertRaises(RuntimeError):
            self.run_wait(lambda: self.devices(3), acquire)
        acquire.assert_not_called()

    def test_any_pair_must_wait_full_five_minutes(self):
        locks = [MagicMock(), MagicMock()]
        selected, result = self.run_wait(lambda: self.devices(0, 7), MagicMock(side_effect=locks))
        self.assertEqual(selected, [0, 7])
        self.assertEqual(result, locks)
        self.assertEqual(self.clock, 1300.)

    def test_busy_interval_restarts_that_cards_timer(self):
        def query():
            return self.devices(0) if 1120 <= self.clock < 1150 else self.devices(0, 7)
        selected, _ = self.run_wait(query, MagicMock(side_effect=[MagicMock(), MagicMock()]))
        self.assertEqual(selected, [0, 7])
        self.assertEqual(self.clock, 1450.)

    def test_idle_cards_can_start_their_timers_at_different_times(self):
        def query():
            return self.devices(1) if self.clock < 1150 else self.devices(1, 6)
        selected, _ = self.run_wait(query, MagicMock(side_effect=[MagicMock(), MagicMock()]))
        self.assertEqual(selected, [1, 6])
        self.assertEqual(self.clock, 1450.)

    def test_rechecks_after_locking_and_releases_on_race(self):
        locks = [MagicMock(), MagicMock()]
        samples = iter([self.devices(0, 7)] * 11 + [self.busy])
        with self.assertRaises(RuntimeError):
            self.run_wait(lambda: next(samples), MagicMock(side_effect=locks), wait_hours=.09)
        for lock in locks:
            lock.close.assert_called_once()

    def test_locked_card_does_not_block_another_eligible_pair(self):
        handles = {f"GPU-test-{i}": MagicMock() for i in [2, 7]}
        selected, locks = self.run_wait(lambda: self.devices(0, 2, 7), lambda uuid: handles.get(uuid))
        self.assertEqual(selected, [2, 7])
        self.assertEqual(locks, [handles["GPU-test-2"], handles["GPU-test-7"]])

    def test_query_gap_restarts_continuous_idle_observation(self):
        def query():
            if self.clock == 1030:
                raise RuntimeError("temporary query failure")
            return self.devices(0, 7)
        self.run_wait(query, MagicMock(side_effect=[MagicMock(), MagicMock()]))
        self.assertEqual(self.clock, 1360.)

    def test_repeated_query_errors_never_allocate(self):
        acquire = MagicMock()
        with self.assertRaises(RuntimeError):
            self.run_wait(MagicMock(side_effect=RuntimeError("query unavailable")), acquire)
        acquire.assert_not_called()
        self.assertEqual(self.clock, 1060.)

    def test_changed_device_identity_fails_before_locking(self):
        acquire = MagicMock()
        wrong = self.devices(0, 7)
        wrong[0] = dict(wrong[0], uuid="GPU-replaced")
        with self.assertRaises(AssertionError):
            self.run_wait(lambda: wrong, acquire)
        acquire.assert_not_called()

    def test_child_sees_only_selected_pair(self):
        with patch.dict(launcher.os.environ, {"CUDA_VISIBLE_DEVICES": "",
                                             "ATOD_RAY_IPC_DIR": "/dev/shm/atod_opd_test",
                                             "TMPDIR": "/persistent/tmp/alfworld_vanilla_opd",
                                             "VLLM_RPC_BASE_PATH": "/incorrect/old/path"}):
            env = launcher.training_environment(Path("/unused"), [2, 7])
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,7")
            self.assertEqual(env["ATOD_BENCHMARK_NUM_GPUS"], "2")
            self.assertEqual(env["VLLM_RPC_BASE_PATH"], "/dev/shm/atod_opd_test")
            self.assertEqual(env["TMPDIR"], "/persistent/tmp/alfworld_vanilla_opd")
            self.assertEqual(launcher.os.environ["CUDA_VISIBLE_DEVICES"], "")

    def test_resume_allows_paths_and_resume_controls_but_rejects_budget_changes(self):
        from omegaconf import OmegaConf
        original = OmegaConf.create(dict(
            data=dict(train_files="old/train", val_files="old/val", max_response_length=512),
            trainer=dict(default_local_dir="old/ckpt", rollout_data_dir="old/rollout",
                         validation_data_dir="old/val", resume_mode="disable", resume_from_path=None,
                         total_training_steps=150), ray_init=dict(_temp_dir="old/ipc"),
            actor_rollout_ref=dict(model=dict(path="old/student"), ref=dict(model=dict(path="old/teacher")))))
        resumed = OmegaConf.create(OmegaConf.to_container(original))
        resumed.data.train_files = "new/train"
        resumed.trainer.resume_mode = "resume_path"
        resumed.trainer.resume_from_path = "old/ckpt/global_step_10"
        resumed.actor_rollout_ref.model.path = "new/student"
        resumed.actor_rollout_ref.ref.model.path = "new/teacher"
        launcher.validate_resume_config(resumed, original)
        resumed.data.max_response_length = 1024
        with self.assertRaises(AssertionError):
            launcher.validate_resume_config(resumed, original)

    def test_resume_overrides_are_passed_to_actual_training_child(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            self.assertFalse(any(arg.startswith("trainer.resume_") for arg in launcher.training_command(run)))
            checkpoint = "/old/checkpoints/global_step_10"
            (run / "resume_plan.json").write_text(json.dumps(dict(checkpoint=checkpoint)))
            command = launcher.training_command(run)
            self.assertIn("trainer.resume_mode=resume_path", command)
            self.assertIn(f"trainer.resume_from_path={checkpoint}", command)

    def test_gpu_indices_are_not_limited_to_eight_devices(self):
        with patch.dict(launcher.os.environ, {"ATOD_RAY_IPC_DIR": "/dev/shm/test"}):
            self.assertEqual(launcher.training_environment(Path("/unused"), [8, 12])["CUDA_VISIBLE_DEVICES"], "8,12")

    def test_source_snapshot_has_no_dependency_on_ignored_local_setup(self):
        import subprocess
        for name in launcher.SOURCE_FILES:
            self.assertTrue((launcher.ROOT / name).is_file(), name)
            ignored = subprocess.run(["git", "check-ignore", "--quiet", name], cwd=launcher.ROOT)
            self.assertNotEqual(ignored.returncode, 0, name)

    def test_full_hydra_config_can_move_models_and_derived_tokenizer_paths(self):
        with patch.dict(launcher.os.environ, ATOD_STUDENT_MODEL_PATH="/old/student",
                        ATOD_TEACHER_MODEL_PATH="/old/teacher"):
            old = launcher.config(Path("/old/run"))
            launcher.os.environ["ATOD_STUDENT_MODEL_PATH"] = "/new/student"
            launcher.os.environ["ATOD_TEACHER_MODEL_PATH"] = "/new/teacher"
            new = launcher.config(Path("/new/run"))
            launcher.validate_resume_config(new, old)
            self.assertEqual(new.critic.model.tokenizer_path, "/new/student")
            self.assertEqual(new.reward_model.model.input_tokenizer, "/new/student")
            new.trainer.total_training_steps = 100
            with self.assertRaises(AssertionError):
                launcher.validate_resume_config(new, old)

    def test_cancelled_wait_does_not_query_or_lock(self):
        acquire, query = MagicMock(), MagicMock()
        with self.assertRaises(RuntimeError):
            self.run_wait(query, acquire, stop=True)
        query.assert_not_called()
        acquire.assert_not_called()


if __name__ == "__main__":
    unittest.main()
