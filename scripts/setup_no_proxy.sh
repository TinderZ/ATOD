#!/usr/bin/env bash
# Source before ATOD installation/download commands; settings are process-local.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/data/home/zzr/cache/huggingface
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR=/mnt/data/home/zzr/cache/pip
export TMPDIR=/mnt/data/home/zzr/tmp/atod_setup
export PIP_INDEX_URL=https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple/
export PIP_FIND_LINKS=/mnt/data/home/zzr/cache/wheels
export PIP_PREFER_BINARY=1
# Public mirror supplies ordinary CPU wheels omitted by the PPU package index.
export PIP_EXTRA_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export MAX_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
export MAKEFLAGS=-j4
