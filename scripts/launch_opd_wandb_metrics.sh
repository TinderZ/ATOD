#!/usr/bin/env bash
set -euo pipefail
_opd_metrics_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${1:-}" != --help ]]; then
  : "${CONDA_PREFIX:?Activate your compatible conda environment first}"
  : "${TMPDIR:?Set TMPDIR to a writable persistent temporary directory}"
  mkdir -p "$TMPDIR"
fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=online WANDB_CONSOLE=off WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true
export WANDB_ERROR_REPORTING=false
unset WANDB_ENTITY WANDB_PROJECT WANDB_RUN_ID WANDB_NAME WANDB_CONFIG_PATHS WANDB_SWEEP_ID WANDB_RESUME
cd "$_opd_metrics_root"
exec python -u scripts/sync_opd_wandb_metrics.py "$@"
