"""Local OPD driver with a native PPU monitor and cleanup of its own Ray cluster."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = Path(os.environ['ATOD_BENCHMARK_DIR'])
SMI = '/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi'


def probe():
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    for device in range(2):
        value = torch.ones((2, 2), device=f'cuda:{device}', dtype=torch.bfloat16)
        assert (value @ value).float().cpu().tolist() == [[2., 2.], [2., 2.]]
    (RUN / 'gpu_preflight.json').write_text(json.dumps({'visible': os.environ['CUDA_VISIBLE_DEVICES'],
                                                      'devices': 2, 'bf16_arithmetic': 'passed'}))
    print('Both approved PPUs passed BF16 arithmetic.', flush=True)


def device_rows(native):
    lines = native.splitlines()
    rows = []
    for i, line in enumerate(lines):
        match = re.search(r'\|\s*(\d+)\s+PPU-', line)
        if not match or int(match[1]) not in [0, 1]:
            continue
        memory = re.search(r'(\d+)MiB\s*/\s*(\d+)MiB', lines[i + 1])
        util = re.search(r'\|\s*(\d+)%', lines[i + 1])
        assert memory and util, 'Incomplete PPU status'
        rows.append({'physical_device': int(match[1]), 'memory_used_mib': int(memory[1]),
                     'memory_total_mib': int(memory[2]), 'utilization_percent': int(util[1])})
    assert len(rows) == 2
    return rows


def monitor(stop):
    while not stop.is_set():
        try:
            native = subprocess.check_output([SMI], text=True, timeout=10)
            sample = {'wall_time': time.time(), 'devices': device_rows(native)}
        except Exception as error:
            sample = {'wall_time': time.time(), 'error': str(error)}
        with (RUN / 'ppu_samples.jsonl').open('a') as stream:
            stream.write(json.dumps(sample) + '\n')
        stop.wait(5)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1'
    assert not any(os.environ.get(k) for k in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'])
    native = subprocess.check_output([SMI], text=True, timeout=10)
    (RUN / 'ppu_before.log').write_text(native)
    for row in device_rows(native):
        assert row['memory_used_mib'] < 512 and row['utilization_percent'] == 0, 'Approved PPU is occupied'
    # The short child exits before loading models, releasing its CUDA contexts.
    subprocess.check_call([sys.executable, str(Path(__file__).resolve()), '--probe-only'])
    stop = threading.Event()
    thread = threading.Thread(target=monitor, args=(stop,), daemon=True)
    thread.start()
    started, exit_code = time.time(), 0
    def terminate(signum, frame):
        raise SystemExit(124)
    signal.signal(signal.SIGTERM, terminate)
    try:
        # CPFS cannot bind Unix sockets. Keep IPC on private tmpfs, while a
        # newly created session's logs point to the persistent experiment dir.
        import ray._private.node
        import ray._private.services
        import ray._private.utils
        # Only this private cluster uses minimal agents. Ray's full reporter
        # calls an NVML per-process API that the PPU HGGC library aborts on.
        ray._private.utils.get_dashboard_dependency_error = lambda: ImportError(
            'OPD benchmark selects minimal dashboard; native PPU monitor supplies GPU metrics'
        )
        # The agent is a separate interpreter. --minimal alone does not filter
        # modules in Ray 2.50 when full dashboard dependencies are installed.
        original_start_process = ray._private.services.start_ray_process
        def start_private_ray_process(command, process_type, **kwargs):
            if process_type == 'raylet':
                import ray.dashboard
                agent = str(Path(ray.dashboard.__file__).parent / 'agent.py')
                wrapper = str(ROOT / 'scripts/opd_ray_dashboard_agent.py')
                command = [part.replace(agent, wrapper) if part.startswith('--dashboard_agent_command=')
                           else part for part in command]
            return original_start_process(command, process_type, **kwargs)
        ray._private.services.start_ray_process = start_private_ray_process
        original_init_temp = ray._private.node.Node._init_temp
        def init_temp_with_persistent_logs(node):
            if node.head and node._ray_params.temp_dir:
                session = Path(node._ray_params.temp_dir) / node._session_name
                session.mkdir(parents=True, exist_ok=True)
                logs = RUN / 'runtime' / 'ray_logs' / node._session_name
                logs.mkdir(parents=True, exist_ok=True)
                (session / 'logs').symlink_to(logs, target_is_directory=True)
            return original_init_temp(node)
        ray._private.node.Node._init_temp = init_temp_with_persistent_logs
        from verl.trainer.main_sod import main as train
        train()
    except BaseException as error:
        exit_code = error.code if isinstance(error, SystemExit) and isinstance(error.code, int) else 1
        (RUN / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        try:
            import ray
            ray.shutdown()
        finally:
            stop.set()
            thread.join(timeout=15)
            (RUN / 'entry_status.json').write_text(json.dumps({'started': started, 'finished': time.time(),
                                                              'exit_code': exit_code}, indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['--probe-only']:
        probe()
    else:
        main()
