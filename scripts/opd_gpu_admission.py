"""Read device occupancy and acquire cooperative locks without allocating GPUs."""
import csv
import fcntl
import io
import os
from pathlib import Path
import subprocess


def query_gpus():
    smi = os.environ.get("ATOD_ADMISSION_SMI", "nvidia-smi")

    def query(kind, fields):
        result = subprocess.run(
            [smi, f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15)
        return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))

    before = query("compute-apps", "gpu_uuid,pid,used_gpu_memory")
    rows = query("gpu", "index,uuid,name,memory.total,memory.used,utilization.gpu")
    after = query("compute-apps", "gpu_uuid,pid,used_gpu_memory")
    processes = {}
    for row in before + after:
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != 3 or not row[0].startswith("GPU-"):
            raise RuntimeError(f"Incomplete compute process status: {row}")
        pid, memory = int(row[1]), int(row[2])
        if pid <= 0 or memory < 0:
            raise RuntimeError(f"Invalid compute process status: {row}")
        processes.setdefault(row[0], set()).add(pid)
    devices = {}
    for row in rows:
        if len(row) != 6 or not row[1].startswith("GPU-"):
            raise RuntimeError(f"Incomplete device status: {row}")
        index, uuid, name, total, used, utilization = row
        device = dict(index=int(index), uuid=uuid, name=name, total_mib=int(total),
                      used_mib=int(used), utilization=int(utilization),
                      pids=sorted(processes.get(uuid, set())))
        if (device["index"] in devices or device["total_mib"] <= 0
                or not 0 <= device["used_mib"] <= device["total_mib"]
                or not 0 <= device["utilization"] <= 100):
            raise RuntimeError(f"Invalid device status: {row}")
        devices[device["index"]] = device
    if not devices or len({d["uuid"] for d in devices.values()}) != len(devices):
        raise RuntimeError("Missing or duplicate devices")
    if set(processes) - {d["uuid"] for d in devices.values()}:
        raise RuntimeError("Compute process refers to a missing device")
    return devices


def idle(device):
    return not device["pids"] and device["used_mib"] <= 512 and device["utilization"] == 0


def acquire_lock(uuid):
    default = Path(__file__).resolve().parents[1] / "outputs" / ".gpu_locks"
    directory = Path(os.environ.get("ATOD_GPU_LOCK_DIR", default))
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / (uuid + ".lock")).open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock
