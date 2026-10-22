"""Estimate remaining OPD time from measured iterations, validation and saves."""
import json
from statistics import mean, median, pstdev
import time
from datetime import datetime
from zoneinfo import ZoneInfo


def jsonl(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith("\n"):
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def remaining_periodic(completed, total, frequency):
    if frequency <= 0:
        return 0
    return len({s for s in range(completed + 1, total + 1)
                if s % frequency == 0 or s == total})


def estimate(records, total=150, test_freq=5, save_freq=10,
             initial_validation_seconds=None, now=None):
    now = time.time() if now is None else now
    rows = {int(r["step"]): r for r in records
            if int(r["step"]) > 0 and "timing_s/step" in r.get("metrics", {})}
    if not rows:
        return dict(state="collecting", completed_updates=0, total_updates=total,
                    eta_beijing=None, note="Waiting for the first completed training iteration.")
    ordered = [rows[s] for s in sorted(rows)]
    completed = int(ordered[-1]["step"])
    sample = ordered[1:] if len(ordered) >= 4 else ordered
    sample = sample[-10:]
    costs = [max(0., r["metrics"]["timing_s/step"]
                 - r["metrics"].get("timing_s/testing", 0.)
                 - r["metrics"].get("timing_s/save_checkpoint", 0.)) for r in sample]
    validations = [r["metrics"]["timing_s/testing"] for r in ordered
                   if "timing_s/testing" in r["metrics"]]
    checkpoints = [r["metrics"]["timing_s/save_checkpoint"] for r in ordered
                   if "timing_s/save_checkpoint" in r["metrics"]]
    val_cost = median(validations[-6:]) if validations else initial_validation_seconds
    save_cost = median(checkpoints[-6:]) if checkpoints else None
    tests = remaining_periodic(completed, total, test_freq)
    saves = remaining_periodic(completed, total, save_freq)
    unknown = []
    if tests and val_cost is None:
        unknown.append("periodic_validation")
    if saves and save_cost is None:
        unknown.append("checkpoint_save")
    remaining = max(0., (total - completed) * mean(costs)
                    + tests * (val_cost or 0.) + saves * (save_cost or 0.)
                    - max(0., now - ordered[-1]["wall_time"]))
    variation = pstdev(costs) / max(mean(costs), 1e-6)
    margin = .5 if len(ordered) < 4 else max(.15, min(.6, 2 * variation))
    phases = {}
    for key in {key for r in sample for key in r["metrics"] if key.startswith("timing_s/")}:
        phases[key.removeprefix("timing_s/")] = mean(
            r["metrics"][key] for r in sample if key in r["metrics"])
    latest = ordered[-1]["metrics"]
    return dict(state="complete" if completed >= total else "training",
                completed_updates=completed, total_updates=total, sample_updates=len(sample),
                mean_training_iteration_seconds=mean(costs),
                validation_seconds=val_cost, checkpoint_seconds=save_cost,
                remaining_validation_runs=tests, remaining_checkpoint_saves=saves,
                remaining_hours=remaining / 3600,
                remaining_hours_range=[remaining * (1-margin)/3600, remaining * (1+margin)/3600],
                eta_beijing=datetime.fromtimestamp(now+remaining, ZoneInfo("Asia/Shanghai")).isoformat(),
                confidence="preliminary" if len(ordered) < 4 or unknown else "measured_recent_window",
                range_note="Heuristic variability range, not a statistical confidence interval.",
                unmeasured_overheads=unknown, phase_seconds=phases,
                last_loss=latest.get("actor/pg_loss"),
                last_validation={k: v for k, v in latest.items() if k.startswith("val/")})


def update(run):
    records = jsonl(run / "metrics.jsonl")
    initialization = run / "initialization.json"
    initial = None
    if initialization.exists():
        init = json.loads(initialization.read_text())
        step_zero = next((r for r in records if r["step"] == 0), None)
        if step_zero:
            initial = max(0., step_zero["wall_time"] - init["worker_init_finished"])
    result = estimate(records, initial_validation_seconds=initial)
    events = jsonl(run / "phase_events.jsonl")
    if events:
        event = events[-1]
        result["last_completed_phase"] = {**event, "seconds": event["finished"]-event["started"]}
    peaks = {}
    for row in jsonl(run / "device_samples.jsonl"):
        for device in row.get("devices", []):
            key = str(device["physical_device"])
            peaks[key] = max(peaks.get(key, 0), device["memory_used_mib"])
    result["peak_gpu_memory_mib"] = peaks
    terminal = run / "supervisor_status.json"
    if terminal.exists():
        status = json.loads(terminal.read_text())
        result["exit_code"] = status["exit_code"]
        result["finished_beijing"] = status["finished_beijing"]
        if status["exit_code"] != 0 or result["completed_updates"] < result["total_updates"]:
            result["state"] = "failed" if status["exit_code"] != 0 else "stopped"
            result["eta_beijing"] = None
            result["note"] = "Training process exited; see error.txt, storage_error.json and train.log."
    result["updated_beijing"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    temporary = run / "eta.json.partial"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(run / "eta.json")
    return result
