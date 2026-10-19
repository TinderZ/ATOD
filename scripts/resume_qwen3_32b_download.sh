#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/home/zzr/ATOD
source scripts/activate_atod.sh atod
export CUDA_VISIBLE_DEVICES=''
stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%N)
log_dir="/mnt/data/home/zzr/ATOD/logs/${stamp}_qwen3_32b_modelscope_resume"
mkdir -p "$log_dir"
printf '%s\n' "$$" > "$log_dir/launcher.pid"
printf 'Resume logs: %s\n' "$log_dir"
if python -u scripts/resume_qwen3_32b_download.py --weight-source modelscope > "$log_dir/download.log" 2>&1; then
  download_status=0
else
  download_status=$?
fi
printf '%s\n' "$download_status" > "$log_dir/exit_code.txt"
exit "$download_status"
