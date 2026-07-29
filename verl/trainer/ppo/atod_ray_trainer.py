"""
ATOD OPD+GRPO Trainer with TIP weighting and Soft Curriculum.

Improved formula (soft curriculum only on OPD term):
  A_final = [soft_w(k,s) × kl_coef × (log_teacher - log_student) × TIP(k)
             + rl_coef × A_GRPO] × response_mask

Extends SODRayTrainer with:
- TIP: Turn Importance Profiling from student-teacher divergence + entropy
- Soft Curriculum: f2b-style sigmoid weight that only modulates OPD term
- GRPO RL: Group-normalized environment reward (always full strength)
"""

from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.rlsd_utils import SkillProvider
from verl.trainer.ppo.sod_ray_trainer import SODRayTrainer
from verl.trainer.ppo.atod_utils import compute_atod_advantage
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)

from agent_system.multi_turn_rollout import adjust_batch


class ATODRayTrainer(SODRayTrainer):
    """
    ATOD OPD+GRPO trainer with TIP weighting and soft curriculum.

    Uses external teacher model for OPD distillation + GRPO environment reward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atod_cfg = self.config.algorithm.get("atod", {})
        # OPD + RL coefficients
        self.atod_kl_coef = atod_cfg.get("kl_coef", 1.0)
        self.atod_rl_coef = atod_cfg.get("rl_coef", 0.3)
        # TIP parameters
        self.enable_tip = atod_cfg.get("enable_tip", True)
        self.tip_rho = atod_cfg.get("tip_rho", 1.0)
        self.tip_min_turns = atod_cfg.get("tip_min_turns", 3)
        self.tip_min_divergence = atod_cfg.get("tip_min_divergence", 0.01)
        self.tip_smoothing = atod_cfg.get("tip_smoothing", 0.0)
        # T-DUR budget-preserving reparam (mean-preserving turn reweight).
        # Default OFF so existing runs reproduce the original z_k in [0,1] behavior.
        self.tip_mean_preserve = atod_cfg.get("tip_mean_preserve", False)
        self.tip_w_min = atod_cfg.get("tip_w_min", 0.5)
        self.tip_w_max = atod_cfg.get("tip_w_max", 2.0)
        self.tip_eps = atod_cfg.get("tip_eps", 1e-8)
        # TIP granularity dispatcher: "turn" (default, original) or "token" (B1 ablation)
        self.tip_granularity = atod_cfg.get("tip_granularity", "turn")
        # token-level only: clip top-quantile of entropy outliers (paper default 0.98)
        self.tip_entropy_clip_quantile = atod_cfg.get("tip_entropy_clip_quantile", 0.98)
        # Ablation: per-turn importance scoring signal.
        # "softor" (default, uses both entropy & divergence) | "entropy_only" | "divergence_only".
        # Only affects the turn-level path; ignored for token-level TIP.
        self.tip_score_mode = atod_cfg.get("tip_score_mode", "softor")
        # Soft curriculum parameters
        self.enable_soft_curriculum = atod_cfg.get("enable_soft_curriculum", True)
        self.curriculum_checkpoint_steps = atod_cfg.get("curriculum_checkpoint_steps", 6.0)
        self.curriculum_softness = atod_cfg.get("curriculum_softness", 1.0)
        self.curriculum_bias = atod_cfg.get("curriculum_bias", 0.5)
        self.curriculum_min_weight = atod_cfg.get("curriculum_min_weight", 0.05)
        self.curriculum_max_weight = atod_cfg.get("curriculum_max_weight", 1.0)
        self.curriculum_warmup_steps = atod_cfg.get("curriculum_warmup_steps", 0)

        # ---- Linear coefficient annealing (NEW) ----
        # When enabled, kl_coef linearly decays from kl_coef -> kl_coef_min and
        # rl_coef linearly grows from rl_coef -> rl_coef_max over the first
        # `coef_anneal_steps` training steps. After that, both are clamped.
        self.enable_coef_anneal = atod_cfg.get("enable_coef_anneal", False)
        self.coef_anneal_steps = atod_cfg.get("coef_anneal_steps", 80)
        self.kl_coef_min = atod_cfg.get("kl_coef_min", 0.1)
        self.rl_coef_max = atod_cfg.get("rl_coef_max", 2.0)

        # ---- Within-turn RTG (Plan B: turn-aware sequence-level KL) ----
        # When enabled, the OPD per-token signal is replaced by a within-turn
        # discounted return-to-go of the student-teacher logp diff. Default OFF
        # so the legacy token-level behavior is preserved bit-for-bit.
        self.enable_within_turn_rtg = atod_cfg.get("enable_within_turn_rtg", False)
        self.opd_gamma = atod_cfg.get("opd_gamma", 0.99)
        self.opd_length_normalization = atod_cfg.get("opd_length_normalization", True)

        # Force external teacher
        self.use_external_teacher = True

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="ATOD OPD+GRPO Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    del batch
                    batch = gen_batch_output

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Compute student log probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # Teacher forward pass (external teacher)
                    with _timer("teacher_forward", timing_raw):
                        teacher_log_probs = self._compute_teacher_log_probs(batch)
                        batch.batch["teacher_log_probs"] = teacher_log_probs

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute GRPO advantages (used as RL term)
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                        # ---- ATOD: Replace advantage with atod OPD+GRPO ----
                        response_mask = batch.batch["response_mask"]
                        student_log_probs = batch.batch["old_log_probs"]
                        teacher_lp = batch.batch["teacher_log_probs"]
                        grpo_adv = batch.batch["advantages"]

                        # Extract trajectory grouping info (SDAR stores each turn as one sample)
                        traj_uids = batch.non_tensor_batch.get("traj_uid", None)
                        turn_steps = batch.non_tensor_batch.get("turn_step", None)
                        if traj_uids is not None:
                            traj_uids = list(traj_uids)
                        if turn_steps is not None:
                            turn_steps = list(turn_steps)

                        atod_adv, atod_metrics = compute_atod_advantage(
                            grpo_advantages=grpo_adv,
                            student_log_probs=student_log_probs,
                            teacher_log_probs=teacher_lp,
                            response_mask=response_mask,
                            global_step=self.global_steps,
                            traj_uids=traj_uids,
                            turn_steps=turn_steps,
                            kl_coef=self.atod_kl_coef,
                            rl_coef=self.atod_rl_coef,
                            enable_tip=self.enable_tip,
                            tip_rho=self.tip_rho,
                            tip_min_turns=self.tip_min_turns,
                            tip_min_divergence=self.tip_min_divergence,
                            tip_smoothing=self.tip_smoothing,
                            tip_mean_preserve=self.tip_mean_preserve,
                            tip_w_min=self.tip_w_min,
                            tip_w_max=self.tip_w_max,
                            tip_eps=self.tip_eps,
                            tip_granularity=self.tip_granularity,
                            tip_entropy_clip_quantile=self.tip_entropy_clip_quantile,
                            tip_score_mode=self.tip_score_mode,
                            enable_soft_curriculum=self.enable_soft_curriculum,
                            curriculum_checkpoint_steps=self.curriculum_checkpoint_steps,
                            curriculum_softness=self.curriculum_softness,
                            curriculum_bias=self.curriculum_bias,
                            curriculum_min_weight=self.curriculum_min_weight,
                            curriculum_max_weight=self.curriculum_max_weight,
                            curriculum_warmup_steps=self.curriculum_warmup_steps,
                            enable_coef_anneal=self.enable_coef_anneal,
                            coef_anneal_steps=self.coef_anneal_steps,
                            kl_coef_min=self.kl_coef_min,
                            rl_coef_max=self.rl_coef_max,
                            enable_within_turn_rtg=self.enable_within_turn_rtg,
                            opd_gamma=self.opd_gamma,
                            opd_length_normalization=self.opd_length_normalization,
                        )
                        batch.batch["advantages"] = atod_adv
                        metrics.update(atod_metrics)

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
