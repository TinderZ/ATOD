# ALFWorld vanilla OPD: Qwen3-8B teacher / Qwen3-1.7B student

此配置保留本次实验的训练预算，使用外部 teacher、uniform OPD 和 vLLM 同步 rollout。
模型、数据、环境和输出路径通过环境变量/命令行指定，不依赖原 PPU 服务器目录。

| 项目 | 设置 |
| --- | --- |
| Teacher | `yyykeibin/Qwen3-8B-GRPO` |
| Teacher revision | `214e7d8d4ecfb5b0eef102e63d88eb14f44c4f78` |
| Student | `Qwen/Qwen3-1.7B` |
| 更新预算 | 总计 150 次更新，150 epochs，每 epoch 一批控制行 |
| 采样 | 16 个任务 × 8 条轨迹 = 128 条/更新，最多 50 turns |
| 每轮生成 | 512 tokens；`enable_thinking=False`；完整 think/action 提示协议 |
| 优化 | LR `1e-6`，global minibatch 256，microbatch 4/卡，PPO epochs 1 |
| OPD | `uniform`、`opd_only=True`、系数 1；entropy 系数 0；无额外 KL loss |
| 设备 | 两卡，FSDP，vLLM TP=2，BF16，eager，teacher 参数 CPU offload |
| 验证 | `valid_seen` 中 128 个任务；训练前验证，每 5 次更新验证 |
| 保存 | 每 10 次更新保存模型、优化器、调度器/RNG、dataloader；保留最新一个 |

actor 包含两处已验证的归一化修复：token-mean 对整个实际 minibatch 的有效 token
归一化；尾部不足一个 minibatch 时使用实际大小。相应 loss/梯度回归测试也已提交。

## 1. 准备环境和持久化目录

以下命令写入你选择的持久化位置。替换 `OPD_ROOT`，先激活自己的环境；脚本不会激活或修改公共环境。

```bash
export OPD_ROOT=/path/on/persistent-disk/opd
mkdir -p "$OPD_ROOT"/{models,datasets,cache,tmp,runs}
export HF_HOME="$OPD_ROOT/cache/huggingface"
export PIP_CACHE_DIR="$OPD_ROOT/cache/pip"
export TMPDIR="$OPD_ROOT/tmp"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT=https://hf-mirror.com
```

如果已有兼容环境可直接复用。原实验为 Python 3.12，torch `2.8.0+ppu2.0.0`、
vLLM `0.11.0+ppu2.0.0`、Transformers 4.57.3、Ray 2.50.0。
PPU 使用平台适配的 GPU 包；不要用 NVIDIA wheel 覆盖它们。
NVIDIA 新环境可按仓库 README 的 vLLM 0.11.0 安装方式准备：

```bash
conda create -p "$OPD_ROOT/env" python=3.12 -y
conda activate "$OPD_ROOT/env"
# 沿用新服务器已配置的镜像；仅在需要时为当前 shell 设置 PIP_INDEX_URL。
# export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
python -m pip install 'vllm==0.11.0'
python -m pip install 'flash-attn==2.7.4.post1' --no-build-isolation
python -m pip install -r requirements-vanilla-opd.txt
python -m pip install -e . --no-deps
python -m pip check
```

GPU wheel 必须与新服务器 CUDA/驱动兼容；NVIDIA 完整训练尚未在原 PPU 机器上验证。
依赖文件单独记录本次实验的运行库，不使用旧 `requirements.txt` 中的 Transformers 4.51.1。

## 2. 模型和 ALFWorld 数据

可复用服务器公共只读资源；下面是下载到个人持久化目录的命令。

```bash
export ATOD_STUDENT_MODEL_PATH="$OPD_ROOT/models/Qwen3-1.7B"
export ATOD_TEACHER_MODEL_PATH="$OPD_ROOT/models/Qwen3-8B-GRPO"
export ALFWORLD_DATA="$OPD_ROOT/datasets/alfworld"
hf download Qwen/Qwen3-1.7B --local-dir "$ATOD_STUDENT_MODEL_PATH"
hf download yyykeibin/Qwen3-8B-GRPO \
  --revision 214e7d8d4ecfb5b0eef102e63d88eb14f44c4f78 \
  --local-dir "$ATOD_TEACHER_MODEL_PATH"
alfworld-download
```

数据目录应包含 `logic/alfred.pddl`、`json_2.1.1/train`、`json_2.1.1/valid_seen`。
准备步骤会检查模型分片、词表、完整标签的 chat template 和数据目录。
实际任务观察由 ALFWorld 生成；parquet 中的空文本只是启动环境的控制行。

## 3. 启动同一配置

先检查 `nvidia-smi`，把 `0 1` 换为获准使用的两张物理卡。此脚本只会使用空闲卡。
CPU prepare 不分配 GPU，但会读取设备状态。

```bash
export ATOD_DEVICE_MONITOR=nvidia
# PPU 环境：source 平台自己的 SDK 环境，再设置 ATOD_DEVICE_MONITOR=ppu。
# 可选 ATOD_DEVICE_SMI=原生监控程序；ATOD_ADMISSION_SMI=支持 CSV 查询的 nvidia-smi 兼容程序。
export VLLM_CACHE_ROOT="$OPD_ROOT/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$OPD_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$OPD_ROOT/cache/triton"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export RUN="$OPD_ROOT/runs/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_vanilla_opd_t8b_s1p7b"

bash scripts/launch_alfworld_vanilla_opd.sh --prepare \
  --run-dir "$RUN" --gpu-pool 0 1 --start-now
nohup bash scripts/launch_alfworld_vanilla_opd.sh \
  --run-dir "$RUN" --gpu-pool 0 1 --start-now > "$RUN/launcher.log" 2>&1 &
echo $! > "$RUN/background.pid"
tail -f "$RUN/train.log"
```

`--start-now` 要求两卡空闲；否则退出。去掉此参数则等待候选卡连续空闲 300 秒，
最多等待 24 小时。`--gpu-pool` 可以列多个候选编号，最终仍只启动两卡。
训练前会再次核对 GPU UUID、占用、配置和代码 SHA256。
Ray/vLLM 的临时 IPC 默认在 `/dev/shm`；日志和训练产物在 `RUN` 中。
其他支持 Unix socket 的本地 IPC 位置可通过 `ATOD_RAY_IPC_ROOT` 指定。

输出包括 `metrics.jsonl`、`phase_events.jsonl`、`device_samples.jsonl`、`eta.json`、
`runtime_config.yaml`、`launch.json`、源文件快照、Git/依赖记录和 `checkpoints/`。
可用 `watch -n 5 cat "$RUN/eta.json"` 查看进度。
原 PPU 实测峰值接近 98 GB/卡，主存约 196 GiB；迁移后需确认设备容量。

## 4. W&B 仅上传数值指标

在服务器本地登录，不要把密钥写进脚本或提交到 Git。

```bash
unset WANDB_MODE
python -m wandb login --cloud --verify
export WANDB_ENTITY=your-entity
export WANDB_PROJECT=opd
nohup bash scripts/launch_opd_wandb_metrics.sh \
  --run-dir "$RUN" --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" \
  > "$RUN/wandb_metrics_launcher.log" 2>&1 &
```

同步进程不占 GPU，不修改训练进程，只读取有限数值指标；不上传轨迹、源码、配置、
console 或模型。训练已退出时同步进程也会结束。图表横轴为 `training/global_step`。

## 5. 从 checkpoint 恢复

换服务器时复制旧 run 的 `resolved_config.yaml`、`source_hashes.json`、`metrics.jsonl`
和 `checkpoints/global_step_N/`，保持相对目录结构。保留原模型/优化器两张分片，
恢复仍使用两卡。在新服务器重新做 CPU 校验，报告保存在新目录。

```bash
export OLD_RUN=/path/to/copied/old-run
export CHECKPOINT="$OLD_RUN/checkpoints/global_step_10"
export REPORT="$OPD_ROOT/runs/checkpoint_validation.json"
CUDA_VISIBLE_DEVICES='' python scripts/verify_opd_checkpoint.py \
  --checkpoint "$CHECKPOINT" --report "$REPORT"
export RUN="$OPD_ROOT/runs/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_resume_step10"
bash scripts/launch_alfworld_vanilla_opd.sh --prepare \
  --run-dir "$RUN" --gpu-pool 0 1 --start-now \
  --resume-from "$CHECKPOINT" --checkpoint-report "$REPORT"
# 再使用第 3 节的 nohup 启动命令，不重复传 --resume-from。
```

恢复参数会传给实际训练子进程：`trainer.resume_mode=resume_path` 和
`trainer.resume_from_path=.../global_step_N`。模型、优化器、调度器、RNG 和 dataloader
均通过原生 checkpoint manager 加载；总预算仍为 150，从 N+1 继续。
恢复时允许模型/输出目录迁移，但核对其他训练参数和训练核心源码。
ALFWorld 环境迭代器没有序列化，跨进程恢复不保证与不中断运行的任务序列逐位一致。

接回旧 W&B run 时，在第 4 节同步命令中额外传
`--run-id <旧run的ID>`；ID 可从旧目录的 `wandb_metrics/sync_state.json` 取得。
旧同步进程应已停止。程序从服务器已有的全局训练步数之后上传，不重传旧 loss。

## 6. CPU 回归检查

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_opd_admission.py tests/test_opd_gpu_admission.py \
  tests/test_opd_eta.py tests/test_wandb_metrics_sync.py \
  tests/workers/actor/test_loss_normalization.py
```

这些检查覆盖 microbatch/尾部 minibatch 的 loss 和梯度、两 rank CPU Gloo 归一化、
设备准入、实际启动命令中的恢复参数、ETA 和 W&B 数值过滤/续传。不会启动 GPU 训练。
