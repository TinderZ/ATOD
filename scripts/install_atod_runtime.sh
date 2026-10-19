#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/home/zzr/ATOD
source scripts/activate_atod.sh atod
# Wait for the already-running PyTorch bootstrap to finish before using pip.
if [[ -n "${1:-}" ]]; then
  for ((i=0; i<240; i++)); do
    [[ -d "/proc/$1" ]] || break
    sleep 5
  done
  [[ ! -d "/proc/$1" ]] || { echo 'PyTorch bootstrap still running'; exit 1; }
fi
export CUDA_VISIBLE_DEVICES=""
python -c 'import torch; print("PPU PyTorch:", torch.__version__)'
python scripts/prepare_cached_requirements.py atod
python -m pip install --no-index --find-links /mnt/data/home/zzr/cache/wheels --no-build-isolation --progress-bar off \
  -c logs/setup_20261016/atod_cached_constraints.txt \
  -r logs/setup_20261016/atod_cached_requirements.txt \
  /mnt/data/home/zzr/cache/wheels/vllm-0.11.0+ppu2.0.0-cp312-cp312-linux_x86_64.whl \
  /mnt/data/home/zzr/cache/wheels/flash_attn-2.7.4.post1+ppu2.0.0-cp312-cp312-linux_x86_64.whl \
  /mnt/data/home/zzr/cache/wheels/triton-3.4.0+ppu2.0.0-cp312-cp312-linux_x86_64.whl \
  /mnt/data/home/zzr/cache/wheels/torchvision-0.23.0+ppu1.7.0-cp312-cp312-linux_x86_64.whl \
  /mnt/data/home/zzr/cache/wheels/torchaudio-2.8.0+ppu1.7.0-cp312-cp312-linux_x86_64.whl \
  /mnt/data/home/zzr/cache/wheels/acext-1.0.0+ppu2.0.0-cp312-cp312-linux_x86_64.whl \
  --report logs/setup_20261016/atod_runtime_report.json
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install --no-deps --no-build-isolation -e agent_system/environments/env_package/search/third_party
python scripts/repair_setup_packages.py
python -m pip check
python -m pip freeze > logs/setup_20261016/atod_freeze.txt
echo 'ATOD RUNTIME INSTALL COMPLETE'
