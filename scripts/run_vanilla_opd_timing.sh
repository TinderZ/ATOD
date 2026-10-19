#!/usr/bin/env bash
# Activate a compatible environment and approve resources before launching.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  cat <<'USAGE'
Usage: bash scripts/run_vanilla_opd_timing.sh <prepared run directory>

Required environment: CUDA_VISIBLE_DEVICES (approved numeric indices),
ALFWORLD_DATA, ATOD_STUDENT_MODEL_PATH, ATOD_TEACHER_MODEL_PATH.
First prepare inputs with ATOD_BENCHMARK_DIR=<run directory> python
scripts/prepare_opd_timing.py in the same active environment.

Optional: ATOD_DEVICE_MONITOR=nvidia|ppu (default nvidia), ATOD_DEVICE_SMI
(monitor executable on PATH or absolute path), ATOD_RAY_IPC_ROOT (default
/dev/shm for temporary sockets), ATOD_RUN_TIMEOUT (default 8h).
The PPU Ray dashboard workaround is enabled only when selecting ppu.
USAGE
  exit 0
fi
atod_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$atod_root"
run_dir="$(cd -- "${1:?Pass a new prepared experiment directory}" && pwd)"
[[ -f "$run_dir/cpu_preflight.json" && ! -f "$run_dir/launcher.pid" ]]
: "${CUDA_VISIBLE_DEVICES:?Set the approved device indices}"
: "${ALFWORLD_DATA:?Set the dataset root}"
: "${ATOD_STUDENT_MODEL_PATH:?Set the local student model directory}"
: "${ATOD_TEACHER_MODEL_PATH:?Set the local teacher model directory}"
export ATOD_PROJECT_ROOT="$atod_root"
ATOD_BENCHMARK_NUM_GPUS=$(python - <<'PY'
import os
items = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert all(item.isdecimal() for item in items), 'Use comma-separated numeric device indices'
assert len({int(item) for item in items}) == len(items), 'Duplicate device indices'
print(len(items))
PY
)
export ATOD_BENCHMARK_NUM_GPUS
export ATOD_DEVICE_MONITOR="${ATOD_DEVICE_MONITOR:-nvidia}"
case "$ATOD_DEVICE_MONITOR" in nvidia|ppu) ;; *) printf 'Expected nvidia or ppu monitor\n' >&2; exit 2 ;; esac
unset RAY_ADDRESS
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false RAY_USAGE_STATS_ENABLED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export ATOD_BENCHMARK_DIR="$run_dir"
export ATOD_BENCHMARK_METRICS="$run_dir/metrics.jsonl"
export ATOD_BENCHMARK_EVENTS="$run_dir/phase_events.jsonl"
ATOD_RAY_IPC_DIR=$(python - <<'PY'
import os, tempfile
print(tempfile.mkdtemp(prefix='atod_opd_', dir=os.environ.get('ATOD_RAY_IPC_ROOT', '/dev/shm')))
PY
)
export ATOD_RAY_IPC_DIR
printf '%s\n' "$$" > "$run_dir/launcher.pid"
printf 'Experiment directory: %s\n' "$run_dir"
python - "$run_dir" <<'PY'
from pathlib import Path
import importlib.metadata as md,json,os,shutil,subprocess,sys
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
run=Path(sys.argv[1]); root=Path(os.environ['ATOD_PROJECT_ROOT'])
files=['verl/utils/tracking.py','verl/trainer/main_sod.py','verl/trainer/ppo/sod_ray_trainer.py',
       'scripts/opd_timing_entry.py','scripts/opd_ray_dashboard_agent.py','scripts/run_vanilla_opd_timing.sh',
       'configs/experiments/vanilla_opd_timing.yaml']
for name in files:
    target=run/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/name,target)
(run/'packages.json').write_text(json.dumps({name:md.version(name) for name in
    ['torch','vllm','ray','transformers','flash-attn','tensordict','datasets','pyarrow','textworld','alfworld']},indent=2))
(run/'git_commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True))
(run/'git_diff.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=root))
(run/'git_status.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=root,text=True))
with initialize_config_dir(config_dir=str(root/'configs/experiments'),version_base=None):
    cfg=compose(config_name='vanilla_opd_timing');OmegaConf.resolve(cfg)
    OmegaConf.save(cfg,run/'runtime_config.yaml')
prepared=OmegaConf.load(run/'resolved_config.yaml')
for key in ['actor_rollout_ref.model.path', 'actor_rollout_ref.ref.model.path']:
    assert OmegaConf.select(prepared,key)==OmegaConf.select(cfg,key), 'Model path changed after preparation: '+key
PY
if timeout --signal=TERM "${ATOD_RUN_TIMEOUT:-8h}" python -u scripts/opd_timing_entry.py \
  --config-path "$atod_root/configs/experiments" \
  --config-name vanilla_opd_timing > "$run_dir/train.log" 2>&1; then
  experiment_status=0
else
  experiment_status=$?
fi
printf '%s\n' "$experiment_status" > "$run_dir/exit_code.txt"
exit "$experiment_status"
