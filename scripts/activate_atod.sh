#!/usr/bin/env bash
# Usage: source /mnt/data/home/zzr/ATOD/scripts/activate_atod.sh [atod|atod-webshop|atod-retriever]
_atod_env="${1:-atod}"
case "$_atod_env" in
  atod|atod-webshop|atod-retriever) ;;
  *) printf 'Unknown ATOD environment: %s\n' "$_atod_env"; return 1 ;;
esac
source /mnt/data/home/zzr/ATOD/scripts/setup_no_proxy.sh
_atod_had_nounset=0
case "$-" in *u*) _atod_had_nounset=1; set +u ;; esac
source /usr/local/anaconda3/etc/profile.d/conda.sh
export JAVA_HOME="${JAVA_HOME:-}"
conda activate "/mnt/data/home/zzr/conda_envs/$_atod_env"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${LIBRARY_PATH:-}"
source /usr/local/PPU_SDK/envsetup.sh
if [[ "$_atod_had_nounset" == 1 ]]; then set -u; fi
export JAVA_HOME="$CONDA_PREFIX"
_atod_purelib="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export LD_LIBRARY_PATH="$_atod_purelib/lib:${LD_LIBRARY_PATH}"
export ACEXT_GEMM_CONFIG_DIR="$_atod_purelib/include/acext/lut/"
export ACEXT_OP_PATH="$_atod_purelib/lib/libacext_op.so"
unset _atod_purelib
if [[ -r /mnt/data/share/datasets/alfworld/logic/alfred.pddl ]]; then
  export ALFWORLD_DATA=/mnt/data/share/datasets/alfworld
else
  export ALFWORLD_DATA=/mnt/data/home/zzr/datasets/alfworld
fi
unset _atod_env _atod_had_nounset
