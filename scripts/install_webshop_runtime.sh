#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/home/zzr/ATOD
source scripts/activate_atod.sh atod-webshop
if [[ -n "${1:-}" ]]; then
  for ((i=0; i<240; i++)); do
    [[ -d "/proc/$1" ]] || break
    sleep 5
  done
  [[ ! -d "/proc/$1" ]] || { echo 'PyTorch bootstrap still running'; exit 1; }
fi
export CUDA_VISIBLE_DEVICES=""
python -c 'import torch; print("PPU PyTorch:", torch.__version__)'
/usr/local/anaconda3/bin/python scripts/conda_mirror.py install \
  --prefix "$CONDA_PREFIX" openjdk=11 --freeze-installed -y
python scripts/prepare_cached_requirements.py webshop
python -m pip install --no-index --find-links /mnt/data/home/zzr/cache/wheels --no-build-isolation --progress-bar off \
  -c logs/setup_20261016/webshop_cached_constraints.txt \
  -r logs/setup_20261016/webshop_cached_requirements.txt \
  /mnt/data/home/zzr/cache/wheels/vllm-0.8.5+ppu1.7.0-cp310-cp310-linux_x86_64.whl \
  /mnt/data/home/zzr/cache/wheels/flash_attn-2.7.4.post1+ppu2.0.0-cp310-cp310-linux_x86_64.whl \
  /mnt/data/home/zzr/cache/wheels/triton-3.2.0+ppu1.7.0-cp310-cp310-linux_x86_64.whl \
  /mnt/data/home/zzr/cache/wheels/torchvision-0.21.0+ppu1.7.0-cp310-cp310-linux_x86_64.whl \
  /mnt/data/home/zzr/cache/wheels/torchaudio-2.6.0+ppu1.7.0-cp310-cp310-linux_x86_64.whl \
  /mnt/data/home/zzr/cache/wheels/xformers-0.0.29.post1+ppu1.7.0-cp310-cp310-linux_x86_64.whl \
  /mnt/data/home/zzr/cache/wheels/acext-1.0.0+ppu2.0.0-cp310-cp310-linux_x86_64.whl \
  --report logs/setup_20261016/webshop_runtime_report.json
python -m pip install --no-deps --no-build-isolation -e .
python scripts/download_webshop_nlp.py
python -m pip install /mnt/data/home/zzr/cache/wheels/en_core_web_sm-3.7.1-py3-none-any.whl
python scripts/repair_setup_packages.py
python -m pip check
python -m pip freeze > logs/setup_20261016/webshop_freeze.txt
echo 'WEBSHOP RUNTIME INSTALL COMPLETE'
