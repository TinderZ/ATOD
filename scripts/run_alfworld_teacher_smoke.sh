#!/usr/bin/env bash
# Single-GPU experiment: launch only after approving its resource plan.
set -euo pipefail
enable_thinking="${1:-false}"
case "$enable_thinking" in
  true|false) ;;
  *) printf 'Expected true or false for enable_thinking\n' >&2; exit 2 ;;
esac
model_preset="${3:-atod-4b}"
model_args=()
case "$model_preset" in
  atod-4b) model_slug=qwen3_4b ;;
  qwen3-32b)
    model_slug=qwen3_32b
    model_args=(--model /mnt/data/home/zzr/models/Qwen3-32B
      --model-repo Qwen/Qwen3-32B
      --model-revision 9216db5781bf21249d130ec9da846c4624c16137
      --policy-label qwen3_32b_official_teacher)
    ;;
  *) printf 'Expected atod-4b or qwen3-32b for model preset\n' >&2; exit 2 ;;
esac
max_new_tokens="${2:-512}"
case "$max_new_tokens" in
  512|1024) ;;
  *) printf 'Expected 512 or 1024 for max_new_tokens\n' >&2; exit 2 ;;
esac
cd /mnt/data/home/zzr/ATOD
source scripts/activate_atod.sh atod
export CUDA_VISIBLE_DEVICES=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%N)
run_dir="/mnt/data/home/zzr/ATOD/outputs/${stamp}_alfworld_teacher_${model_slug}_greedy_seed0_thinking_${enable_thinking}_tokens${max_new_tokens}"
mkdir -p "$run_dir"
mkdir -p "$run_dir/source_snapshot"
cp scripts/run_alfworld_teacher_episode.py scripts/run_alfworld_teacher_smoke.sh "$run_dir/source_snapshot/"
printf '%s\n' "$$" > "$run_dir/launcher.pid"
printf 'Output and log: %s\n' "$run_dir"
/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi > "$run_dir/ppu_before.log"
if timeout --signal=TERM 20m python -u scripts/run_alfworld_teacher_episode.py \
  --seed 0 --max-steps 12 --max-new-tokens "$max_new_tokens" --max-input-tokens 2048 \
  --enable-thinking "$enable_thinking" \
  "${model_args[@]}" \
  --output-dir "$run_dir" > "$run_dir/run.log" 2>&1; then
  run_status=0
else
  run_status=$?
fi
printf '%s\n' "$run_status" > "$run_dir/exit_code.txt"
/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi > "$run_dir/ppu_after.log"
exit "$run_status"
