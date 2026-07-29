"""
SOD (Step-wise On-policy Distillation) Trainer.

Extends SkillSDRayTrainer to support:
1. External teacher model via RefPolicy worker (instead of self-distillation).
2. SOD/OPD advantage modification (instead of auxiliary loss).

When use_external_teacher=True:
  - Teacher log-probs come from an independent RefPolicy worker loaded from
    a separate model path (config: actor_rollout_ref.ref.model.path).
  - Distillation is done via token-level advantage modification, NOT auxiliary loss.

When use_external_teacher=False:
  - Falls back to original SDAR/SkillSD behavior (self-distillation with
    privileged skill information).
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
from verl.trainer.ppo.rlsd_ray_trainer import RLSDRayTrainer, build_teacher_batch
from verl.trainer.ppo.sod_utils import compute_opd_advantage, compute_per_turn_disagreement_entropy
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)

from agent_system.multi_turn_rollout import adjust_batch


class SODRayTrainer(RLSDRayTrainer):
    """
    SOD trainer that supports both:
    - External teacher model distillation (SOD/OPD mode)
    - Self-distillation with privileged info (original SDAR/RLSD mode, fallback)

    When use_external_teacher=True:
      - Teacher log-probs are obtained from the RefPolicy worker
      - Advantages are modified with OPD/SOD signal (no auxiliary loss)
    When use_external_teacher=False:
      - Falls back to original SDAR self-distillation behavior
    """

    def __init__(self, *args, skill_provider: SkillProvider = None, **kwargs):
        super().__init__(*args, skill_provider=skill_provider, **kwargs)
        # SOD config
        sod_cfg = self.config.algorithm.get("sod", {})
        self.use_external_teacher = sod_cfg.get("use_external_teacher", False)
        self.sod_mode = sod_cfg.get("mode", "stepwise")  # "stepwise" or "gated"
        # Step-wise OPD parameters
        self.sod_opd_coef = sod_cfg.get("opd_coef", 1.0)
        self.sod_epsilon = sod_cfg.get("epsilon", 1e-6)
        self.sod_delta = sod_cfg.get("delta", 0.5)
        self.sod_opd_only = sod_cfg.get("opd_only", False)
        # Gated OPD parameters
        self.sod_gamma = sod_cfg.get("gamma", 1.0)
        self.sod_beta_min = sod_cfg.get("beta_min", 0.0)
        self.sod_beta_max = sod_cfg.get("beta_max", 0.3)

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        """
        Compute teacher log probs.

        If use_external_teacher=True: use RefPolicy worker (independent teacher model).
        If use_external_teacher=False: use same actor with privileged skill info (original SDAR).
        """
        if self.use_external_teacher:
            # Use the independent reference policy (teacher) model
            assert self.use_reference_policy, (
                "[SOD] use_external_teacher=True requires RefPolicy worker. "
                "Make sure actor_rollout_ref.ref.model.path is set and "
                "algorithm.sod.use_external_teacher=True triggers RefPolicy creation."
            )
            if not self.ref_in_actor:
                ref_output = self.ref_policy_wg.compute_ref_log_prob(batch)
            else:
                ref_output = self.actor_rollout_wg.compute_ref_log_prob(batch)
            teacher_log_probs = ref_output.batch["ref_log_prob"]
            return teacher_log_probs
        else:
            # Original SDAR self-distillation path
            return super()._compute_teacher_log_probs(batch)

    def fit(self):
        """
        The training loop of SOD.

        Key differences from SkillSD/SDAR:
        - When use_external_teacher=True:
          * Teacher log-probs come from RefPolicy (not same actor with skills)
          * Distillation is done by modifying advantages (not auxiliary loss)
        - When use_external_teacher=False:
          * Falls back to original SkillSD/SDAR behavior
        """
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

        desc = "SOD Training" if self.use_external_teacher else "SDAR Training (fallback)"
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc=desc)
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

                    # Compute student log probs (old_log_probs)
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

                    # ---- SOD: Teacher forward pass ----
                    with _timer("teacher_forward", timing_raw):
                        teacher_log_probs = self._compute_teacher_log_probs(batch)
                        batch.batch["teacher_log_probs"] = teacher_log_probs

                    # Standard KL ref policy (if enabled separately for KL penalty/loss)
                    if self.use_reference_policy and not self.use_external_teacher:
                        # Only compute ref separately if NOT already used as teacher
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

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

                        # Compute standard GRPO advantages
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

                        # ---- SOD: Modify advantages with OPD signal ----
                        if self.use_external_teacher:
                            response_mask = batch.batch["response_mask"]
                            student_log_probs = batch.batch["old_log_probs"]
                            teacher_lp = batch.batch["teacher_log_probs"]
                            grpo_adv = batch.batch["advantages"]

                            # Extract trajectory grouping info (used by stepwise mode)
                            traj_uids = batch.non_tensor_batch.get("traj_uid", None)
                            turn_steps = batch.non_tensor_batch.get("turn_step", None)
                            if traj_uids is not None:
                                traj_uids = list(traj_uids)
                            if turn_steps is not None:
                                turn_steps = list(turn_steps)

                            modified_adv, sod_metrics = compute_opd_advantage(
                                grpo_advantages=grpo_adv,
                                student_log_probs=student_log_probs,
                                teacher_log_probs=teacher_lp,
                                response_mask=response_mask,
                                mode=self.sod_mode,
                                gamma=self.sod_gamma,
                                beta_min=self.sod_beta_min,
                                beta_max=self.sod_beta_max,
                                opd_coef=self.sod_opd_coef,
                                epsilon=self.sod_epsilon,
                                delta=self.sod_delta,
                                opd_only=self.sod_opd_only,
                                traj_uids=traj_uids,
                                turn_steps=turn_steps,
                            )
                            batch.batch["advantages"] = modified_adv
                            metrics.update(sod_metrics)

                            # Log teacher-student gap
                            delta_t = (teacher_lp - student_log_probs) * response_mask
                            metrics["sod/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                            metrics["sod/teacher_student_gap_std"] = masked_mean(delta_t ** 2, response_mask).sqrt().item()

                            # ---- Per-turn disagreement proxy d_k & entropy proxies h_k ----
                            metrics.update(compute_per_turn_disagreement_entropy(
                                student_log_probs=student_log_probs,
                                teacher_log_probs=teacher_lp,
                                response_mask=response_mask,
                                turn_steps=turn_steps,
                                prefix="sod_turn",
                            ))
                        else:
                            # Fallback: SDAR self-distillation (keep advantages unchanged,
                            # distillation via auxiliary loss in update_policy)
                            response_mask = batch.batch["response_mask"]
                            student_log_probs = batch.batch["old_log_probs"]
                            teacher_lp = batch.batch["teacher_log_probs"]
                            delta_t = (teacher_lp - student_log_probs) * response_mask
                            metrics["sdar/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                            metrics["sdar/teacher_student_gap_std"] = masked_mean(delta_t ** 2, response_mask).sqrt().item()

                            # ---- Per-turn disagreement proxy d_k & entropy proxies h_k ----
                            fb_turn_steps = batch.non_tensor_batch.get("turn_step", None)
                            if fb_turn_steps is not None:
                                fb_turn_steps = list(fb_turn_steps)
                            metrics.update(compute_per_turn_disagreement_entropy(
                                student_log_probs=student_log_probs,
                                teacher_log_probs=teacher_lp,
                                response_mask=response_mask,
                                turn_steps=fb_turn_steps,
                                prefix="sdar_turn",
                            ))

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
