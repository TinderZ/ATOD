#!/usr/bin/env bash
# Authorized public-model download; process-local mirror and proxy settings.
set -euo pipefail
cd /mnt/data/home/zzr/ATOD
source scripts/activate_atod.sh atod
export CUDA_VISIBLE_DEVICES=''
stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%N)
download_log_dir="/mnt/data/home/zzr/ATOD/logs/${stamp}_qwen3_32b_download"
mkdir -p "$download_log_dir"
printf '%s\n' "$$" > "$download_log_dir/launcher.pid"
printf 'Download logs: %s\n' "$download_log_dir"
df -h /mnt/data/home/zzr/models > "$download_log_dir/disk_before.log"
if hf download Qwen/Qwen3-32B \
  --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --local-dir /mnt/data/home/zzr/models/Qwen3-32B \
  --max-workers 2 > "$download_log_dir/download.log" 2>&1; then
  download_status=0
else
  download_status=$?
fi
printf '%s\n' "$download_status" > "$download_log_dir/exit_code.txt"
exit "$download_status"
