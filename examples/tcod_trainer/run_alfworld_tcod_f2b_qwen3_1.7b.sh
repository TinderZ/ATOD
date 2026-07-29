set -x
source /root/sdar-env/bin/activate


ENGINE=${1:-vllm}

num_cpus_per_env_worker=0.1

# =====================================================
# TCOD-f2b (Forward-to-Backward) Configuration
# (with GRPO-trained Qwen3-4B as teacher)
# =====================================================
# Curriculum: student starts with 1 step, expands by 1 every checkpoint_steps.
#   distill_window = 1 + (global_step // checkpoint_steps)
#   effective_max_steps = min(distill_window, max_env_steps)
#
# Pure OPD distillation: A = opd_coef * (log_teacher - log_student)
# =====================================================

# TCOD parameters
tcod_strategy="f2b"
checkpoint_steps=3    # expand window by 1 step every 6 training steps
opd_coef=1.0
opd_only=true         # pure OPD, no GRPO

# Model paths
# Student: vanilla Qwen3-1.7B
# Teacher: Qwen3-4B fine-tuned with GRPO on alfworld (global_step_150 of grpo_qwen3_4b run, merged to HF format)
student_model_path=Qwen3-1.7B
teacher_model_path=Qwen3-4B-GRPO

# =====================================================
# Training Configuration
# =====================================================
train_data_size=16
val_data_size=128
group_size=8
experiment_name="tcod_f2b_alfworld_qwen3_1.7b_teacher_grpo4b_step300_ckpt${checkpoint_steps}"
export ALFWORLD_DATA=$HOME/data/alfworld

export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key_here}"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_tcod \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$student_model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sod.use_external_teacher=true \
    +algorithm.tcod.strategy=$tcod_strategy \
    +algorithm.tcod.checkpoint_steps=$checkpoint_steps \
    +algorithm.tcod.opd_coef=$opd_coef \
    +algorithm.tcod.opd_only=$opd_only \
    +algorithm.tcod.skills_dir=skills/alfworld \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
