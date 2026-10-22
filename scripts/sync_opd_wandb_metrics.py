"""Mirror scalar OPD metrics to W&B without attaching to the trainer."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
from zoneinfo import ZoneInfo

PREFIXES = (
    "actor/", "critic/", "val/", "episode/", "sod/", "sod_turn/",
    "training/", "timing_s/", "timing_per_token_ms/", "perf/",
    "response_length/", "prompt_length/", "benchmark/",
)
STOP = False


def scalar_payload(record):
    step = record["step"]
    if type(step) is not int or step < 0:
        raise ValueError("Metric step must be a nonnegative integer")
    payload = {
        key: value for key, value in record["metrics"].items()
        if isinstance(key, str) and key.startswith(PREFIXES)
        and type(value) in (int, float) and math.isfinite(value)
    }
    payload["training/global_step"] = step
    return payload


def complete_records(stream):
    """Leave an unfinished final line for the next poll."""
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line or not line.endswith(b"\n"):
            stream.seek(position)
            return
        if line.strip():
            yield json.loads(line)


def mirror_records(stream, remote_run, next_step):
    count = 0
    for record in complete_records(stream):
        payload = scalar_payload(record)
        step = record["step"]
        if step < next_step:
            continue
        # W&B's internal row counter also includes progress heartbeats.
        # Plot every metric against the actual training/global_step instead.
        remote_run.log(payload, commit=True)
        next_step = step + 1
        count += 1
    return next_step, count


def progress_heartbeat(remote_run, next_step):
    if next_step > 0:
        remote_run.log({"training/global_step": next_step - 1}, commit=True)


def privacy_settings(wandb, directory, mode):
    return wandb.Settings(
        mode=mode, base_url="https://api.wandb.ai", root_dir=str(directory), console="off",
        disable_code=True, disable_git=True, save_code=False,
        disable_job_creation=True, x_disable_meta=True,
        x_disable_stats=True, x_disable_machine_info=True,
        x_save_requirements=False, ignore_globs=("*",),
        program="metrics-only-sync", host="metrics-only",
        init_timeout=60, login_timeout=30, quiet=True,
    )


def write_state(path, value):
    temporary = path.with_name(f"{path.name}.{time.time_ns()}.partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def beijing():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def supervisor_alive(run):
    """Detect abrupt supervisor loss even when its final status could not be written."""
    pid_path = run / "supervisor.pid"
    if not pid_path.exists():
        return None
    pid = int(json.loads(pid_path.read_text()))
    process = Path("/proc") / str(pid)
    try:
        if process.stat().st_uid != os.getuid():
            return False
        command = (process / "cmdline").read_bytes().split(b"\0")
        return (str(run).encode() in command
                and any(item.endswith(b"run_alfworld_vanilla_opd.py") for item in command))
    except FileNotFoundError:
        return False


def stop(_signal, _frame):
    global STOP
    STOP = True


def preview(run):
    with (run / "metrics.jsonl").open("rb") as stream:
        rows = [dict(step=row["step"], metric_count=len(scalar_payload(row)))
                for row in complete_records(stream)]
    print(json.dumps(dict(network_used=False, records=rows,
                          allowed_prefixes=PREFIXES,
                          source="metrics.jsonl", file_uploads=False), indent=2))


def synchronize(args):
    import wandb
    from wandb.sdk.lib.service import ipc_support
    # This process's SDK service uses loopback TCP; EFC cannot bind Unix sockets.
    ipc_support.SUPPORTS_UNIX = False
    run = args.run_dir.resolve()
    directory = run / "wandb_metrics"
    directory.mkdir(exist_ok=True)
    lock = (directory / "sync.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = directory / "sync_state.json"
    run_id = args.run_id or "opd" + hashlib.sha256(str(run).encode()).hexdigest()[:17]
    state = dict(entity=args.entity, project=args.project, run_id=run_id,
                 source=str(run / "metrics.jsonl"), pid=os.getpid(),
                 mode=args.mode, started_beijing=beijing(), state="initializing",
                 upload_policy="Finite scalar training metrics only; no source files, logs or artifacts")
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if any(previous[key] != state[key] for key in ("entity", "project", "mode", "run_id")):
            raise ValueError("Existing mirror belongs to a different W&B destination or mode")
    write_state(state_path, state)
    write_state(directory / "sync.pid", os.getpid())
    # Sandbox SDK files away from source code, checkpoints and trajectories.
    os.chdir(directory)
    remote = None
    exit_code = 1
    try:
        remote = wandb.init(
            entity=args.entity, project=args.project,
            name=args.name or "vanilla-opd-" + run.name[:8],
            id=run_id, resume="allow" if args.mode == "online" else None,
            config={}, save_code=False, mode=args.mode,
            settings=privacy_settings(wandb, directory, args.mode),
        )
        remote.define_metric("training/global_step")
        remote.define_metric("*", step_metric="training/global_step")
        # Resume from the server's TRAINING step; SDK row numbers include heartbeats.
        next_step = int(remote.summary.get("training/global_step", -1)) + 1
        state.update(state="following", url=remote.url if args.mode == "online" else None,
                     next_step=next_step)
        write_state(state_path, state)
        print(json.dumps(dict(event="wandb_metrics_mirror_started", **state)), flush=True)
        last_progress = 0.0
        while not STOP:
            source = run / "metrics.jsonl"
            if not source.exists():
                time.sleep(args.poll_seconds)
                continue
            break
        if not STOP:
            with source.open("rb") as stream:
                while not STOP:
                    next_step, count = mirror_records(stream, remote, next_step)
                    state.update(next_step=next_step, last_queued_step=next_step-1,
                                 updated_beijing=beijing())
                    write_state(state_path, state)
                    if count:
                        print(json.dumps(dict(event="metrics_queued", records=count,
                                              last_step=next_step-1, beijing=beijing())), flush=True)
                    terminal = run / "supervisor_status.json"
                    if terminal.exists():
                        # One final read after observing the trainer's terminal marker.
                        mirror_records(stream, remote, next_step)
                        exit_code = int(json.loads(terminal.read_text())["exit_code"])
                        break
                    if not args.once and supervisor_alive(run) is False:
                        state["stop_reason"] = "Training supervisor disappeared without a terminal record"
                        print(json.dumps(dict(event="training_supervisor_lost", beijing=beijing())), flush=True)
                        exit_code = 1
                        break
                    if not args.once and time.monotonic() - last_progress >= 60:
                        # Keep long updates visible without copying or inventing losses.
                        progress_heartbeat(remote, next_step)
                        last_progress = time.monotonic()
                    if args.once:
                        exit_code = 0
                        break
                    time.sleep(args.poll_seconds)
        if STOP:
            exit_code = 130
        remote.finish(exit_code=exit_code)
        remote = None
        state.update(state="finished" if exit_code == 0 else "stopped",
                     exit_code=exit_code, finished_beijing=beijing())
        write_state(state_path, state)
        return exit_code
    except BaseException as error:
        state.update(state="failed", error_type=type(error).__name__,
                     finished_beijing=beijing())
        write_state(state_path, state)
        raise
    finally:
        if remote is not None:
            remote.finish(exit_code=exit_code)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--entity")
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--run-id", help="Continue an existing W&B run after resuming in a new local directory")
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--mode", choices=["online", "offline"], default="online")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.preview:
        preview(args.run_dir.resolve())
        return 0
    if not args.entity or not args.project:
        parser.error("--entity and --project are required for synchronization")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be at least 1")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return synchronize(args)


if __name__ == "__main__":
    raise SystemExit(main())
