#!/bin/bash
set -x

export no_proxy=localhost,127.0.0.1,0.0.0.0

ENGINE=${1:-vllm}

# ============ 1. Start Retrieval Server ============
echo "Starting retrieval server..."
source /root/retriever-env/bin/activate

python examples/search/retriever/retrieval_server.py \
  --index_path ~/data/searchR1/e5_Flat.index \
  --corpus_path ~/data/searchR1/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --faiss_gpu \
  --port 8000 > logs/retrieval_server.log 2>&1 &
RETRIEVER_PID=$!
echo "Retrieval server PID: $RETRIEVER_PID"

# Wait for server to be ready
echo "Waiting for retrieval server to be ready..."
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s -o /dev/null -w "%{http_code}" -X POST http://0.0.0.0:8000/retrieve \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "topk": 1}' 2>/dev/null | grep -q "200"; then
        echo "Retrieval server is ready! (waited ${WAITED}s)"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  waiting... (${WAITED}s)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Retrieval server failed to start within ${MAX_WAIT}s"
    kill $RETRIEVER_PID 2>/dev/null
    exit 1
fi

# Cleanup retriever on script exit
trap "echo 'Stopping retrieval server...'; kill $RETRIEVER_PID 2>/dev/null; wait $RETRIEVER_PID 2>/dev/null" EXIT

# ============ 2. Start Training ============
echo "Starting SOD Search QA training..."
source /root/sdar-env/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key_here}"
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

# =====================================================
# SOD (Step-wise OPD) Configuration
# =====================================================
#   A_total = A_GRPO + opd_coef * w_k * (log_teacher - log_student)
#   w_k adapts per assistant turn based on student-teacher divergence
# =====================================================
use_external_teacher=true
sod_mode="stepwise"
opd_coef=1.0
epsilon=1e-6
delta=0.2
opd_only=false

# Model paths
# Student: vanilla Qwen3-4B
# Teacher: Qwen3-30B-A3B fine-tuned with GRPO on search (global_step_200 of grpo_qwen3_30ba3b run, merged to HF format)
student_model_path=Qwen3-4B
teacher_model_path=Qwen3-30B-A3B-GRPO

experiment_name="sod_search_qwen3_4b_teacher_grpo30ba3b_step150_${sod_mode}_coef${opd_coef}_delta${delta}"

train_data_size=128
val_data_size=512
group_size=8

TRAIN_DATA="$HOME/data/searchR1_processed_direct/train.parquet"
VAL_DATA="$HOME/data/searchR1_processed_direct/test_2048.parquet"

python3 -m verl.trainer.main_sod \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$student_model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sod.use_external_teacher=$use_external_teacher \
    +algorithm.sod.mode=$sod_mode \
    +algorithm.sod.opd_coef=$opd_coef \
    +algorithm.sod.epsilon=$epsilon \
    +algorithm.sod.delta=$delta \
    +algorithm.sod.opd_only=$opd_only \
    +algorithm.sod.skills_dir=skills/search \
    +algorithm.sod.skill_all=false \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url='http://0.0.0.0:8000/retrieve' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=15 \
    trainer.total_training_steps=150 \
    trainer.val_before_train=True ${@:2}
