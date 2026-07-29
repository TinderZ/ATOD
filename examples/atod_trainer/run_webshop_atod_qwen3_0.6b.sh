set -x
source /root/verl-webshop-env/bin/activate


ENGINE=${1:-vllm}

num_cpus_per_env_worker=0.1

# =====================================================
# ATOD OPD+GRPO+TIP (NO Soft Curriculum)
# WITH Linear Coefficient Annealing
# (kl_coef ↓, rl_coef ↑) over the first `coef_anneal_steps` steps
# Use this to isolate the effect of coef-annealing from soft curriculum.
# =====================================================
# Base coefficients (= initial values when anneal is enabled)
kl_coef=1.0          # kl_init: start strong distillation in early training
rl_coef=1.0          # rl_init: GRPO weight at start (unscaled)

# Linear annealing schedule
enable_coef_anneal=true
coef_anneal_steps=80 # T: number of steps to linearly transition coefs
kl_coef_min=0.1      # kl floor: keep a soft distillation prior throughout
rl_coef_max=2.0      # rl ceiling: GRPO becomes dominant in late training

# TIP settings (kept identical to baseline)
enable_tip=true
tip_rho=1.0
tip_min_turns=3
tip_min_divergence=0.01
tip_smoothing=0.0

# Soft Curriculum: DISABLED in this run
enable_soft_curriculum=false
# The values below are not used when enable_soft_curriculum=false,
# but kept here for completeness so the config schema stays consistent.
curriculum_checkpoint_steps=3
curriculum_softness=1.0
curriculum_bias=0.5
curriculum_min_weight=0.05
curriculum_max_weight=1.0

# Model paths
# Student: vanilla Qwen3-0.6B
# Teacher: Qwen3-4B fine-tuned with GRPO on webshop (global_step_150 of grpo_qwen3_4b run)
student_model_path=Qwen3-0.6B
teacher_model_path=Qwen3-4B-GRPO

train_data_size=16
val_data_size=128
group_size=8
experiment_name="atod_anneal_nocurr_webshop_qwen3_0.6b_teacher_grpo4b_step300_klInit${kl_coef}to${kl_coef_min}_rlInit${rl_coef}to${rl_coef_max}_T${coef_anneal_steps}_tip"
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key_here}"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_atod \
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
    +algorithm.atod.kl_coef=$kl_coef \
    +algorithm.atod.rl_coef=$rl_coef \
    +algorithm.atod.enable_coef_anneal=$enable_coef_anneal \
    +algorithm.atod.coef_anneal_steps=$coef_anneal_steps \
    +algorithm.atod.kl_coef_min=$kl_coef_min \
    +algorithm.atod.rl_coef_max=$rl_coef_max \
    +algorithm.atod.enable_tip=$enable_tip \
    +algorithm.atod.tip_rho=$tip_rho \
    +algorithm.atod.tip_min_turns=$tip_min_turns \
    +algorithm.atod.tip_min_divergence=$tip_min_divergence \
    +algorithm.atod.tip_smoothing=$tip_smoothing \
    +algorithm.atod.enable_soft_curriculum=$enable_soft_curriculum \
    +algorithm.atod.curriculum_checkpoint_steps=$curriculum_checkpoint_steps \
    +algorithm.atod.curriculum_softness=$curriculum_softness \
    +algorithm.atod.curriculum_bias=$curriculum_bias \
    +algorithm.atod.curriculum_min_weight=$curriculum_min_weight \
    +algorithm.atod.curriculum_max_weight=$curriculum_max_weight \
    +algorithm.atod.skills_dir=skills/webshop \
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
    trainer.resume_mode=disable \
    trainer.val_before_train=True $@
