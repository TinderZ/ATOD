#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/home/zzr/ATOD
source scripts/activate_atod.sh atod-retriever
if [[ -n "${1:-}" ]]; then
  for ((i=0; i<720; i++)); do
    [[ -d "/proc/$1" ]] || break
    sleep 5
  done
  [[ ! -d "/proc/$1" ]] || { echo 'Retriever installation still running'; exit 1; }
fi
export CUDA_VISIBLE_DEVICES=""
python -m pip check
python -m pip freeze > logs/setup_20261016/retriever_freeze.txt
python scripts/check_atod_setup.py retriever
echo 'RETRIEVER SETUP VERIFIED'
