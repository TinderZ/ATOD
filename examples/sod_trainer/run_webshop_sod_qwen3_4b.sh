set -x
source /root/verl-webshop-env/bin/activate


ENGINE=${1:-vllm}

num_cpus_per_env_worker=0.1

# =====================================================
# SOD (Step-wise OPD) Configuration with GRPO-30B-A3B teacher
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
# Teacher: Qwen3-30B-A3B fine-tuned with GRPO on webshop (global_step_200 of grpo_qwen3_30ba3b run)
student_model_path=Qwen3-4B
teacher_model_path=Qwen3-30B-A3B-GRPO

train_data_size=16
val_data_size=128
group_size=8
experiment_name="sod_webshop_qwen3_4b_teacher_grpo30ba3b_step150_coef${opd_coef}_delta${delta}"
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key_here}"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_sod \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$student_model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sod.use_external_teacher=$use_external_teacher \
    +algorithm.sod.mode=$sod_mode \
    +algorithm.sod.opd_coef=$opd_coef \
    +algorithm.sod.epsilon=$epsilon \
    +algorithm.sod.delta=$delta \
    +algorithm.sod.opd_only=$opd_only \
    +algorithm.sod.skills_dir=skills/webshop \
    +algorithm.sod.skill_all=false \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshopv1' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
