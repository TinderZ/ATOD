#!/usr/bin/env bash
# One authorized episode after download and verification; no retries of inference.
set -euo pipefail
cd /mnt/data/home/zzr/ATOD
download_log_dir="${1:?Pass the active download log directory}"
mkdir "$download_log_dir/experiment_claim"
source scripts/activate_atod.sh atod
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%N)
workflow_dir="/mnt/data/home/zzr/ATOD/logs/${stamp}_qwen3_32b_completion"
mkdir -p "$workflow_dir"
printf '%s\n' "$$" > "$workflow_dir/launcher.pid"
printf 'Workflow logs: %s\n' "$workflow_dir"
while [[ ! -f "$download_log_dir/exit_code.txt" ]]; do
  sleep 30
done
if [[ "$(cat "$download_log_dir/exit_code.txt")" != 0 ]]; then
  printf 'Download failed; inference was not started.\n'
  exit 1
fi
python -u scripts/verify_qwen3_32b.py > "$workflow_dir/verification.log" 2>&1
python - "$workflow_dir" <<'PY'
import hashlib,json,re,subprocess,sys
from pathlib import Path
root=Path('/mnt/data/home/zzr/ATOD')
out=Path(sys.argv[1])
native=subprocess.check_output(['/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi'],text=True)
(out/'ppu_preflight.log').write_text(native)
lines=native.splitlines()
start=next(i for i,line in enumerate(lines) if re.search(r'\|\s*2\s+PPU-',line))
memory_line=lines[start+1]
memory=re.search(r'(\d+)MiB\s*/\s*(\d+)MiB',memory_line)
util=re.search(r'\|\s*(\d+)%',memory_line)
assert memory and util, 'Incomplete native PPU status'
assert int(memory[1])<256 and int(util[1])==0, 'PPU 2 is occupied; inference not started'
manifest={}
for name in ['agent_system/environments/prompts/alfworld.py','agent_system/environments/env_package/alfworld/projection.py','agent_system/environments/env_manager.py']:
    assert not subprocess.check_output(['git','diff','HEAD','--',name],cwd=root)
    manifest[name]={'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest(),'unchanged_from_commit':True}
(out/'protocol_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Checksums verified; native PPU 2 preflight passed; launching one approved episode.')
PY
if bash scripts/run_alfworld_teacher_smoke.sh true 1024 qwen3-32b > "$workflow_dir/experiment.launch.log" 2>&1; then
  experiment_status=0
else
  experiment_status=$?
fi
run_dir=$(python - "$workflow_dir/experiment.launch.log" <<'PY'
from pathlib import Path
import sys
for line in Path(sys.argv[1]).read_text().splitlines():
    if line.startswith('Output and log: '):
        print(line.removeprefix('Output and log: '))
        break
else:
    raise SystemExit('No experiment output directory recorded')
PY
)
printf '%s\n' "$run_dir" > "$workflow_dir/run_directory.txt"
printf '%s\n' "$experiment_status" > "$workflow_dir/experiment_exit_code.txt"
cp "$workflow_dir/protocol_manifest.json" "$run_dir/"
cp configs/setup/alfworld_qwen3_32b_plan.json "$run_dir/approved_plan.json"
cp /mnt/data/home/zzr/models/Qwen3-32B/local_verification.json "$run_dir/model_verification.json"
cp scripts/analyze_alfworld_teacher_comparison.py "$run_dir/source_snapshot/"
python scripts/analyze_alfworld_teacher_comparison.py "$run_dir" > "$workflow_dir/comparison.log" 2>&1
printf 'Experiment complete: %s (exit %s)\n' "$run_dir" "$experiment_status"
exit "$experiment_status"
