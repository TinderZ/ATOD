#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/home/zzr/ATOD
run_dir="${1:?Pass a new prepared experiment directory}"
[[ -f "$run_dir/cpu_preflight.json" && ! -f "$run_dir/launcher.pid" ]]
source scripts/activate_atod.sh atod
unset RAY_ADDRESS
export CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false RAY_USAGE_STATS_ENABLED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export ATOD_BENCHMARK_DIR="$run_dir"
export ATOD_BENCHMARK_METRICS="$run_dir/metrics.jsonl"
export ATOD_BENCHMARK_EVENTS="$run_dir/phase_events.jsonl"
ATOD_RAY_IPC_DIR=$(python -c "import tempfile; print(tempfile.mkdtemp(prefix='zzr_opd_', dir='/dev/shm'))")
export ATOD_RAY_IPC_DIR
printf '%s\n' "$$" > "$run_dir/launcher.pid"
printf 'Experiment directory: %s\n' "$run_dir"
python - "$run_dir" <<'PY'
from pathlib import Path
import importlib.metadata as md,json,shutil,subprocess,sys
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
run=Path(sys.argv[1]); root=Path('/mnt/data/home/zzr/ATOD')
files=['verl/utils/tracking.py','verl/trainer/main_sod.py','verl/trainer/ppo/sod_ray_trainer.py',
       'scripts/opd_timing_entry.py','scripts/opd_ray_dashboard_agent.py','scripts/run_vanilla_opd_timing.sh',
       'configs/experiments/vanilla_opd_timing.yaml']
for name in files:
    target=run/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/name,target)
(run/'packages.json').write_text(json.dumps({name:md.version(name) for name in
    ['torch','vllm','ray','transformers','flash-attn','tensordict','datasets','pyarrow','textworld','alfworld']},indent=2))
(run/'git_diff.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=root))
(run/'git_status.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=root,text=True))
with initialize_config_dir(config_dir=str(root/'configs/experiments'),version_base=None):
    cfg=compose(config_name='vanilla_opd_timing');OmegaConf.resolve(cfg)
    OmegaConf.save(cfg,run/'runtime_config.yaml')
PY
if timeout --signal=TERM 8h python -u scripts/opd_timing_entry.py \
  --config-path /mnt/data/home/zzr/ATOD/configs/experiments \
  --config-name vanilla_opd_timing > "$run_dir/train.log" 2>&1; then
  experiment_status=0
else
  experiment_status=$?
fi
printf '%s\n' "$experiment_status" > "$run_dir/exit_code.txt"
/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi > "$run_dir/ppu_after.log"
exit "$experiment_status"
