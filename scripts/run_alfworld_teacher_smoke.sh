#!/usr/bin/env bash
# Activate a compatible environment and approve resources before launching.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  cat <<'USAGE'
Usage: CUDA_VISIBLE_DEVICES=<one index> ALFWORLD_DATA=<dataset root> \
  bash scripts/run_alfworld_teacher_smoke.sh [true|false] [512|1024] \
  --model <local model dir> --game <training game.tw-pddl> [episode options]

Uses the active Python environment. Pass --model-repo, --model-revision and
--policy-label to record model provenance. ATOD_OUTPUT_ROOT controls outputs;
ATOD_RUN_TIMEOUT defaults to 20m. Optional ATOD_DEVICE_SMI names a monitor
executable on PATH (or its absolute path); no platform SDK path is assumed.
USAGE
  exit 0
fi
enable_thinking=false
if [[ "${1:-}" == true || "${1:-}" == false ]]; then enable_thinking="$1"; shift; fi
max_new_tokens=512
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then max_new_tokens="$1"; shift; fi
case "$enable_thinking" in true|false) ;; *) printf 'Expected true or false\n' >&2; exit 2 ;; esac
case "$max_new_tokens" in 512|1024) ;; *) printf 'Expected 512 or 1024 tokens\n' >&2; exit 2 ;; esac
: "${CUDA_VISIBLE_DEVICES:?Set one approved device index}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || { printf 'Expose exactly one device index\n' >&2; exit 2; }
: "${ALFWORLD_DATA:?Set the dataset root}"
atod_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$atod_root"
# Validate arguments without loading libraries or allocating a device.
python - "$@" <<'PY'
import sys
for option in ['--model', '--game']:
    if option not in [value.split('=')[0] for value in sys.argv[1:]]:
        raise SystemExit('Required episode argument: ' + option)
if any(value.split('=')[0] in {'--output-dir', '--enable-thinking', '--max-new-tokens', '--prepare-only'} for value in sys.argv[1:]):
    raise SystemExit('Use the launcher arguments for thinking/token settings; output directories are generated automatically')
PY
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%N)
run_dir="${ATOD_OUTPUT_ROOT:-$atod_root/outputs}/${stamp}_alfworld_teacher_seed0_thinking_${enable_thinking}_tokens${max_new_tokens}"
mkdir -p "$(dirname -- "$run_dir")"
mkdir "$run_dir"
mkdir "$run_dir/source_snapshot"
cp scripts/run_alfworld_teacher_episode.py scripts/run_alfworld_teacher_smoke.sh "$run_dir/source_snapshot/"
printf '%s\n' "$$" > "$run_dir/launcher.pid"
printf 'Output and log: %s\n' "$run_dir"
if [[ -n "${ATOD_DEVICE_SMI:-}" ]]; then "$ATOD_DEVICE_SMI" > "$run_dir/device_before.log"; fi
if timeout --signal=TERM "${ATOD_RUN_TIMEOUT:-20m}" python -u scripts/run_alfworld_teacher_episode.py \
  --seed 0 --max-steps 12 --max-new-tokens "$max_new_tokens" --max-input-tokens 2048 \
  --enable-thinking "$enable_thinking" "$@" --output-dir "$run_dir" > "$run_dir/run.log" 2>&1; then
  run_status=0
else
  run_status=$?
fi
printf '%s\n' "$run_status" > "$run_dir/exit_code.txt"
if [[ -n "${ATOD_DEVICE_SMI:-}" ]]; then "$ATOD_DEVICE_SMI" > "$run_dir/device_after.log"; fi
exit "$run_status"
