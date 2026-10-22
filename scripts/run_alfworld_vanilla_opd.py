"""Prepare and supervise the authorized two-GPU ALFWorld vanilla OPD run."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib.metadata
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

from opd_eta import update
from opd_gpu_admission import acquire_lock, idle, query_gpus

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "vanilla_opd_alfworld_1p7b"
STOP = False
SOURCE_FILES = [
    "verl/trainer/config/ppo_trainer.yaml",
    "configs/experiments/" + CONFIG + ".yaml",
    "scripts/launch_alfworld_vanilla_opd.sh", "scripts/run_alfworld_vanilla_opd.py",
    "scripts/opd_eta.py", "scripts/opd_timing_entry.py", "scripts/opd_ray_dashboard_agent.py",
    "scripts/verify_opd_checkpoint.py",
    "scripts/opd_gpu_admission.py",
    "verl/workers/actor/dp_actor.py", "verl/utils/loss_normalization.py",
    "verl/trainer/main_sod.py", "verl/trainer/ppo/sod_ray_trainer.py",
    "verl/trainer/ppo/sod_utils.py", "verl/trainer/ppo/ray_trainer.py",
    "verl/utils/tracking.py", "agent_system/multi_turn_rollout/rollout_loop.py",
    "agent_system/environments/prompts/alfworld.py",
    "verl/workers/sharding_manager/fsdp_vllm.py",
]


def beijing():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def config(run):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    os.environ["ATOD_PROJECT_ROOT"] = str(ROOT)
    os.environ["ATOD_BENCHMARK_DIR"] = str(run)
    os.environ.setdefault("ATOD_RAY_IPC_DIR", str(run / "runtime/ray_ipc_placeholder"))
    with initialize_config_dir(config_dir=str(ROOT / "configs/experiments"), version_base=None):
        cfg = compose(config_name=CONFIG)
        OmegaConf.resolve(cfg)
    resume_plan = run / "resume_plan.json"
    if resume_plan.exists():
        plan = json.loads(resume_plan.read_text())
        cfg.trainer.resume_mode = "resume_path"
        cfg.trainer.resume_from_path = plan["checkpoint"]
    assert cfg.data.max_response_length == 512
    assert cfg.data.apply_chat_template_kwargs.enable_thinking is False
    assert cfg.trainer.total_training_steps == cfg.trainer.total_epochs == 150
    assert cfg.data.train_batch_size == 16 and cfg.env.rollout.n == 8
    assert cfg.data.val_batch_size == 128 and cfg.env.max_steps == 50
    assert cfg.algorithm.sod.mode == "uniform" and cfg.algorithm.sod.opd_only
    assert cfg.actor_rollout_ref.actor.entropy_coeff == 0
    assert not cfg.actor_rollout_ref.actor.use_invalid_action_penalty
    assert cfg.trainer.n_gpus_per_node == cfg.ray_init.num_gpus == 2
    return cfg


def validate_resume_config(cfg, previous_cfg):
    """Permit only resume controls and paths belonging to the new attempt."""
    from omegaconf import OmegaConf
    expected = OmegaConf.create(OmegaConf.to_container(previous_cfg, resolve=True))
    for key in ("data.train_files", "data.val_files", "trainer.default_local_dir",
                "trainer.rollout_data_dir", "trainer.validation_data_dir", "ray_init._temp_dir",
                "trainer.resume_mode", "trainer.resume_from_path",
                "actor_rollout_ref.model.path", "actor_rollout_ref.ref.model.path",
                "critic.model.tokenizer_path", "reward_model.model.input_tokenizer"):
        current = OmegaConf.select(cfg, key)
        if current is not None or OmegaConf.select(expected, key) is not None:
            OmegaConf.update(expected, key, current)
    assert cfg == expected, "Resume changed training parameters outside resume controls/output paths"


def prepare(run, gpu_pool, idle_seconds, resume_from=None, checkpoint_report=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from transformers import AutoConfig, AutoTokenizer
    run.mkdir(parents=True, exist_ok=False)
    (run / "inputs").mkdir()
    if resume_from is not None:
        checkpoint = resume_from.resolve()
        previous_run = checkpoint.parent.parent
        assert checkpoint.name.startswith("global_step_") and checkpoint.parent.name == "checkpoints"
        step = int(checkpoint.name.removeprefix("global_step_"))
        assert 0 < step < 150
        assert checkpoint_report is not None, "Pass --checkpoint-report from verify_opd_checkpoint.py"
        report = checkpoint_report.resolve()
        validation = json.loads(report.read_text())
        assert validation["passed"] and validation["checkpoint"] == str(checkpoint)
        assert validation["world_size"] == 2 and validation["step"] == step
        previous_sources = json.loads((previous_run / "source_hashes.json").read_text())
        current_sources = source_hashes()
        # The supervisor adds resumption; all model, loss, rollout and trainer code is identical.
        for name, digest in previous_sources.items():
            if name.startswith(("verl/", "agent_system/")) and name != "verl/trainer/config/ppo_trainer.yaml":
                assert current_sources[name] == digest, f"Training source changed: {name}"
        write_json(run / "resume_plan.json", dict(
            checkpoint=str(checkpoint), previous_run=str(previous_run), resume_step=step,
            first_training_step=step+1, total_training_steps=150, cpu_validation=str(report),
            original_training_sources_unchanged=True,
            note="ALFWorld environment iterators are not serialized by the original trainer; a resume cannot reproduce an uninterrupted environment task sequence bit for bit."))
    cfg = config(run)
    if resume_from is not None:
        validate_resume_config(cfg, OmegaConf.load(previous_run / "resolved_config.yaml"))
        records = [json.loads(line) for line in (previous_run / "metrics.jsonl").read_text().splitlines()
                   if line.strip()]
        records = [row for row in records if row["step"] <= step]
        assert records and max(row["step"] for row in records) == step
        with (run / "metrics.jsonl").open("x") as stream:
            stream.write("".join(json.dumps(row) + "\n" for row in records))
            stream.flush()
            os.fsync(stream.fileno())
    for split, count in [("train", 16), ("val", 128)]:
        rows = [{"data_source": "text", "prompt": [{"role": "user", "content": ""}],
                 "ability": "agent", "extra_info": {"split": split, "index": i}}
                for i in range(count)]
        pq.write_table(pa.Table.from_pylist(rows), run / "inputs" / (split + ".parquet"))
    student_path = cfg.actor_rollout_ref.model.path
    teacher_path = cfg.actor_rollout_ref.ref.model.path
    models = {}
    tokenizers = []
    for role, path in [("student", student_path), ("teacher", teacher_path)]:
        folder = Path(path)
        info = AutoConfig.from_pretrained(folder, local_files_only=True)
        index = folder / "model.safetensors.index.json"
        weights = sorted(set(json.loads(index.read_text())["weight_map"].values())) \
            if index.exists() else ["model.safetensors"]
        assert all((folder / name).is_file() and (folder / name).stat().st_size > 0
                   for name in weights)
        tokenizers.append(AutoTokenizer.from_pretrained(folder, local_files_only=True))
        models[role] = dict(path=str(folder), model_type=info.model_type,
                            vocabulary_size=info.vocab_size, shards=weights)
    assert tokenizers[0].get_vocab() == tokenizers[1].get_vocab()
    chat = [{"role": "user", "content": "<think>reasoning</think><action>action</action>"}]
    rendered = [tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for tok in tokenizers]
    assert rendered[0] == rendered[1]
    assert rendered[0].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    data = Path(os.environ["ALFWORLD_DATA"])
    for name in ["logic/alfred.pddl", "json_2.1.1/train", "json_2.1.1/valid_seen"]:
        assert (data / name).exists(), name
    devices = query_gpus()
    assert all(i in devices for i in gpu_pool), "Missing requested devices"
    OmegaConf.save(cfg, run / "resolved_config.yaml")
    snapshot = run / "source_snapshot"
    for name in SOURCE_FILES:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    write_json(run / "source_hashes.json", source_hashes())
    for name, args in [("git_status.txt", ["status", "--short"]),
                       ("git_diff.patch", ["diff", "--"]),
                       ("git_commit.txt", ["rev-parse", "HEAD"]),
                       ("git_branch.txt", ["branch", "--show-current"])]:
        (run / name).write_text(subprocess.check_output(["git", *args], cwd=ROOT, text=True))
    packages = {name: importlib.metadata.version(name) for name in
                ["torch", "vllm", "transformers", "ray", "alfworld", "textworld", "pyarrow"]}
    (run / "pip_freeze.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))
    provenance = os.environ.get("ATOD_TEACHER_VERIFICATION")
    if provenance:
        shutil.copyfile(provenance, run / "teacher_verification.json")
    write_json(run / "cpu_preflight.json", dict(
        passed=True, prepared_beijing=beijing(), python=sys.executable, packages=packages,
        models=models, dataset=str(data), training_split="train",
        validation_split="valid_seen (eval_in_distribution)", seed=0,
        control_rows=dict(train=16, validation=128), trajectories_per_update=128,
        budget=dict(training_iterations=150, max_turns=50, max_generated_tokens_per_turn=512,
                    enable_thinking=False, validation_every=5),
        student_teacher_vocab_and_template_identical=True,
        full_action_and_think_prompt_protocol_retained=True,
        initial_devices={str(i): devices[i] for i in gpu_pool},
        admission=dict(gpu_pool=gpu_pool, required_idle_seconds=idle_seconds, training_gpu_count=2),
        resources=dict(gpu_candidates=gpu_pool, training_gpu_count=2,
                       ray_cpus=32, measured_ppu_host_memory_gib=196,
                       measured_ppu_peak_gpu_memory_mib=[97820, 96982],
                       checkpoint_retention=1, estimated_disk_gib=[25, 50]),
        differences_from_original_example=[
            "New teacher and two GPUs; microbatch 4 instead of 32, TP remains 2.",
            "Uniform OPD only; zero entropy and no invalid-action penalty.",
            "Loss uses corrected token/minibatch normalization.",
            "Eager mode and cached per-episode weights; weights remain fixed within rollout.",
            "Local logs and trajectories; checkpoint every 10 updates, keep latest one.",
            "Prevalidated empty control parquet rows; task prompts come from ALFWorld.",
        ]))
    print("CPU_PREFLIGHT_PASSED", run, flush=True)


def request_stop(_signal, _frame):
    global STOP
    STOP = True


def wait_for_devices(run, preflight, wait_hours, gpu_pool, idle_seconds):
    """Admit any two continuously idle candidates, without using occupied GPUs."""
    started, wall_started = time.monotonic(), time.time()
    idle_since = {i: None for i in gpu_pool}
    last_state, query_errors = None, 0
    while not STOP and time.monotonic() - started < wait_hours * 3600:
        try:
            devices = query_gpus()
        except Exception as error:
            # A gap in observations cannot count towards continuous idleness.
            idle_since = {i: None for i in gpu_pool}
            query_errors += 1
            write_json(run / "queue.json", dict(state="gpu_query_error", error=repr(error),
                       consecutive_errors=query_errors, updated_beijing=beijing()))
            print("GPU_QUERY_ERROR", repr(error), flush=True)
            if query_errors >= 3:
                raise RuntimeError("Three consecutive GPU queries failed; refusing to allocate") from error
            time.sleep(30)
            continue
        query_errors = 0
        now = time.monotonic()
        for i in gpu_pool:
            assert devices[i]["uuid"] == preflight["initial_devices"][str(i)]["uuid"]
            if idle(devices[i]):
                if idle_since[i] is None:
                    idle_since[i] = now
            else:
                idle_since[i] = None
        eligible = sorted((i for i in gpu_pool if idle_since[i] is not None
                           and now-idle_since[i] >= idle_seconds),
                          key=lambda i: (idle_since[i], i))
        state = dict(state="waiting_for_devices", updated_beijing=beijing(),
                     wait_started=wall_started, wait_deadline=wall_started + wait_hours * 3600,
                     gpu_pool=gpu_pool, required_idle_seconds=idle_seconds, eligible_devices=eligible,
                     idle_seconds_by_gpu={str(i): 0 if idle_since[i] is None else now-idle_since[i]
                                          for i in gpu_pool},
                     devices={str(i): devices[i] for i in gpu_pool})
        write_json(run / "queue.json", state)
        write_json(run / "eta.json", dict(state="waiting_for_devices", completed_updates=0,
                   total_updates=150, eta_beijing=None, updated_beijing=beijing()))
        signature = (tuple((i, tuple(devices[i]["pids"]), idle(devices[i])) for i in gpu_pool),
                     tuple(eligible))
        if signature != last_state:
            print("WAITING_FOR_GPUS", json.dumps(state), flush=True)
            last_state = signature
        for pair in combinations(eligible, 2):
            locks = []
            for i in pair:
                resource_lock = acquire_lock(devices[i]["uuid"])
                if resource_lock is None:
                    break
                locks.append(resource_lock)
            if len(locks) == 2:
                try:
                    fresh = query_gpus()
                    for i in gpu_pool:
                        assert fresh[i]["uuid"] == preflight["initial_devices"][str(i)]["uuid"]
                    if all(idle(fresh[i]) for i in pair) and not STOP:
                        write_json(run / "queue.json", dict(state="admitted", admitted_beijing=beijing(),
                                   wait_seconds=time.monotonic()-started, selected_gpus=list(pair),
                                   gpu_pool=gpu_pool, required_idle_seconds=idle_seconds,
                                   devices={str(i): fresh[i] for i in pair}))
                        return list(pair), locks
                except BaseException:
                    for resource_lock in locks:
                        resource_lock.close()
                    raise
                # A job appeared while taking locks. Restart observation before selecting again.
                for i in gpu_pool:
                    if not idle(fresh[i]):
                        idle_since[i] = None
                for resource_lock in locks:
                    resource_lock.close()
                break
            for resource_lock in locks:
                resource_lock.close()
        if idle_seconds == 0:
            raise RuntimeError("Requested GPUs are occupied or locked; immediate launch cancelled")
        time.sleep(30)
    raise RuntimeError("Stopped or device wait deadline reached; no occupied GPU was used")


def training_environment(run, selected):
    assert len(selected) == len(set(selected)) == 2
    assert all(i >= 0 for i in selected)
    return dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, selected)),
                VLLM_RPC_BASE_PATH=os.environ["ATOD_RAY_IPC_DIR"],
                ATOD_BENCHMARK_NUM_GPUS="2",
                ATOD_BENCHMARK_METRICS=str(run / "metrics.jsonl"),
                ATOD_BENCHMARK_EVENTS=str(run / "phase_events.jsonl"))


def training_command(run):
    command = [sys.executable, "-u", str(ROOT / "scripts/opd_timing_entry.py"),
               "--config-path", str(ROOT / "configs/experiments"), "--config-name", CONFIG]
    plan = run / "resume_plan.json"
    if plan.exists():
        checkpoint = json.loads(plan.read_text())["checkpoint"]
        command.extend(["trainer.resume_mode=resume_path", f"trainer.resume_from_path={checkpoint}"])
    return command


def supervise(run, wait_hours, gpu_pool, idle_seconds):
    from omegaconf import OmegaConf
    assert not os.environ.get("CUDA_VISIBLE_DEVICES"), "Supervisor must not allocate GPUs"
    preflight = json.loads((run / "cpu_preflight.json").read_text())
    assert preflight["passed"] and not (run / "launch.json").exists()
    assert preflight["admission"] == dict(gpu_pool=gpu_pool, required_idle_seconds=idle_seconds,
                                          training_gpu_count=2)
    lock = (run / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(run / "supervisor.pid", os.getpid())
    assert source_hashes() == json.loads((run / "source_hashes.json").read_text()), "Sources changed"
    locks = []
    child = None
    exit_code = 1
    started = time.time()
    try:
        selected, locks = wait_for_devices(run, preflight, wait_hours, gpu_pool, idle_seconds)
        assert source_hashes() == json.loads((run / "source_hashes.json").read_text()), "Sources changed while waiting"
        assert config(run) == OmegaConf.load(run / "resolved_config.yaml"), "Resolved configuration changed"
        driver_started = time.time()
        # Unix sockets cannot live on CPFS. Ray and vLLM IPC are ephemeral.
        ipc = tempfile.mkdtemp(prefix="atod_opd_alfworld_", dir=os.environ.get("ATOD_RAY_IPC_ROOT", "/dev/shm"))
        os.environ["ATOD_RAY_IPC_DIR"] = ipc
        cfg = config(run)
        OmegaConf.save(cfg, run / "runtime_config.yaml")
        env = training_environment(run, selected)
        command = training_command(run)
        write_json(run / "launch.json", dict(
            started=driver_started, started_beijing=beijing(), queued_at=started, supervisor_pid=os.getpid(),
            command=command, cwd=str(ROOT), cuda_visible_devices=env["CUDA_VISIBLE_DEVICES"],
            selected_physical_gpus=selected, monitored_gpu_pool=gpu_pool,
            conda_prefix=os.environ.get("CONDA_PREFIX"), ray_ipc_dir=ipc,
            vllm_rpc_base_path=env["VLLM_RPC_BASE_PATH"],
            proxy_variables_present={k: bool(env.get(k)) for k in
                ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]},
            environment={k: env.get(k) for k in
                ["HF_ENDPOINT", "HF_HOME", "TMPDIR", "ALFWORLD_DATA", "OMP_NUM_THREADS",
                 "HF_HUB_OFFLINE", "VLLM_ENABLE_V1_MULTIPROCESSING", "ATOD_BENCHMARK_METRICS",
                 "ATOD_BENCHMARK_EVENTS", "ATOD_RAY_IPC_DIR", "VLLM_RPC_BASE_PATH"]}))
        with (run / "train.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, pass_fds=tuple(f.fileno() for f in locks))
            write_json(run / "driver.pid", child.pid)
            print("TRAINING_DRIVER_STARTED", child.pid, run, flush=True)
            last_state = None
            while child.poll() is None:
                if STOP:
                    child.terminate()
                    exit_code = child.wait(timeout=120)
                    break
                try:
                    os.fsync(log.fileno())
                except OSError as error:
                    write_json(run / "storage_error.json", dict(
                        detected_beijing=beijing(), operation="train.log fsync",
                        errno=error.errno, error=str(error)))
                    raise
                try:
                    status = update(run)
                    phase = status.get("last_completed_phase", {})
                    state = (status["completed_updates"], phase.get("step"), phase.get("phase"))
                    if state != last_state:
                        with (run / "eta_history.jsonl").open("a") as stream:
                            stream.write(json.dumps(status) + "\n")
                        print("PROGRESS", json.dumps(status), flush=True)
                        last_state = state
                except Exception as error:
                    print("ETA_UPDATE_ERROR", repr(error), flush=True)
                time.sleep(15)
            else:
                exit_code = child.returncode
        update(run)
        return exit_code
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                print("ATTENTION: driver did not exit after SIGTERM; no global kill used", flush=True)
        write_json(run / "supervisor_status.json", dict(
            exit_code=exit_code, finished_beijing=beijing(), elapsed_seconds=time.time()-started))
        write_json(run / "exit_code.txt", exit_code)
        update(run)
        for resource_lock in locks:
            resource_lock.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--resume-from", type=Path,
                        help="During prepare, resume a validated global_step_N checkpoint")
    parser.add_argument("--checkpoint-report", type=Path,
                        help="CPU verification report for --resume-from")
    parser.add_argument("--wait-hours", type=float, default=24)
    parser.add_argument("--gpu-pool", type=int, nargs="+", required=True)
    parser.add_argument("--idle-seconds", type=int, default=300)
    parser.add_argument("--start-now", action="store_true",
                        help="Start on an explicit pair after idle checks, without a waiting period")
    args = parser.parse_args()
    assert 0 < args.wait_hours <= 24
    assert len(args.gpu_pool) >= 2 and len(set(args.gpu_pool)) == len(args.gpu_pool)
    assert all(i >= 0 for i in args.gpu_pool)
    if args.start_now:
        assert len(args.gpu_pool) == 2, "Immediate launch requires an explicit GPU pair"
        args.idle_seconds = 0
    else:
        assert args.idle_seconds >= 300
    run = args.run_dir.resolve()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if args.prepare:
        prepare(run, args.gpu_pool, args.idle_seconds, args.resume_from, args.checkpoint_report)
        return 0
    assert args.resume_from is None, "--resume-from is recorded during --prepare"
    assert args.checkpoint_report is None, "--checkpoint-report is recorded during --prepare"
    return supervise(run, args.wait_hours, args.gpu_pool, args.idle_seconds)


if __name__ == "__main__":
    sys.exit(main())
