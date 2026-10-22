#!/usr/bin/env bash
set -euo pipefail
_opd_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$_opd_root"
if [[ "${1:-}" == --help ]]; then
  python scripts/run_alfworld_vanilla_opd.py --help
  exit 0
fi
: "${CONDA_PREFIX:?Activate your compatible conda environment first}"
: "${ALFWORLD_DATA:?Set the ALFWorld dataset directory}"
: "${ATOD_STUDENT_MODEL_PATH:?Set the local Qwen3-1.7B directory}"
: "${ATOD_TEACHER_MODEL_PATH:?Set the local teacher directory}"
# Use process-local network settings; do not change global pip/git configuration.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
# The CPU-only supervisor selects exactly two GPUs after admission.
export CUDA_VISIBLE_DEVICES='' CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export RAY_USAGE_STATS_ENABLED=0 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export ATOD_DEVICE_MONITOR="${ATOD_DEVICE_MONITOR:-nvidia}"
case "$ATOD_DEVICE_MONITOR" in nvidia|ppu) ;; *) echo 'Expected nvidia or ppu' >&2; exit 2 ;; esac
: "${TMPDIR:?Set TMPDIR to a writable persistent temporary directory}"
unset RAY_ADDRESS
mkdir -p "$TMPDIR"
exec python -u scripts/run_alfworld_vanilla_opd.py "$@"
