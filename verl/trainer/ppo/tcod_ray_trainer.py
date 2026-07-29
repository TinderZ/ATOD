"""
TCOD (Temporal Curriculum for On-Policy Distillation) Trainer.

Extends SODRayTrainer with curriculum scheduling:
  - f2b: Gradually increase the number of steps the student executes.
  - b2f: Gradually decrease the expert prefix length.

Both strategies use pure OPD distillation (no GRPO signal).

The curriculum is applied at the rollout level by dynamically controlling
`config.env.max_steps` (f2b) or injecting expert prefix actions (b2f).
"""

from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import open_dict
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
from verl.trainer.ppo.sod_utils import compute_opd_advantage
from verl.trainer.ppo.tcod_utils import compute_f2b_max_steps, compute_b2f_expert_prefix_len
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)

from agent_system.multi_turn_rollout import adjust_batch


class TCODRayTrainer(SODRayTrainer):
    """
    TCOD trainer with temporal curriculum scheduling.

    Supports:
      - strategy="f2b": Forward-to-Backward, gradually increase student steps.
      - strategy="b2f": Backward-to-Forward, gradually decrease expert prefix.

    Both use pure OPD distillation from external teacher (opd_only=True by default).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TCOD config
        tcod_cfg = self.config.algorithm.get("tcod", {})
        self.tcod_strategy = tcod_cfg.get("strategy", "f2b")  # "f2b" or "b2f"
        self.tcod_checkpoint_steps = tcod_cfg.get("checkpoint_steps", 6)
        self.tcod_max_env_steps = self.config.env.max_steps  # Save original max_steps

        # Force pure OPD mode for TCOD
        self.use_external_teacher = True
        self.sod_mode = tcod_cfg.get("opd_mode", "uniform")
        self.sod_opd_coef = tcod_cfg.get("opd_coef", 1.0)
        self.sod_opd_only = tcod_cfg.get("opd_only", True)  # Default: pure OPD, no GRPO

    def _get_effective_max_steps(self, global_step: int) -> int:
        """Compute effective max steps based on TCOD strategy."""
        if self.tcod_strategy == "f2b":
            return compute_f2b_max_steps(
                global_step=global_step,
                checkpoint_steps=self.tcod_checkpoint_steps,
                max_env_steps=self.tcod_max_env_steps,
            )
        else:
            # b2f: student always runs up to max_env_steps from the checkpoint
            return self.tcod_max_env_steps

    def _get_expert_prefix_len(self, global_step: int, total_expert_actions: int) -> int:
        """Compute expert prefix length for b2f strategy."""
        if self.tcod_strategy != "b2f":
            return 0
        return compute_b2f_expert_prefix_len(
            global_step=global_step,
            checkpoint_steps=self.tcod_checkpoint_steps,
            total_expert_actions=total_expert_actions,
        )

    def fit(self):
        """
        The training loop of TCOD.

        Key features:
        - f2b: Dynamically reduces env.max_steps to limit student trajectory length.
        - b2f: Injects expert_prefix_steps into env_kwargs so environment executes
          expert actions before student takes over.
        - Uses pure OPD distillation (teacher logp - student logp) as advantage.
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

        desc = f"TCOD-{self.tcod_strategy} Training"
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

                # ---- TCOD: Apply curriculum scheduling ----
                effective_max_steps = self._get_effective_max_steps(self.global_steps)

                # For b2f: compute expert prefix and inject into env_kwargs
                if self.tcod_strategy == "b2f":
                    # Get expert actions count from dataset (if available)
                    expert_actions_key = "expert_actions"
                    if expert_actions_key in gen_batch.non_tensor_batch:
                        # Per-sample expert actions (list of action strings)
                        # Use the average length for scheduling
                        expert_actions_list = gen_batch.non_tensor_batch[expert_actions_key]
                        avg_expert_len = int(np.mean([len(a) if isinstance(a, (list, np.ndarray)) else 0 for a in expert_actions_list]))
                    else:
                        avg_expert_len = self.tcod_max_env_steps
                    expert_prefix_len = self._get_expert_prefix_len(self.global_steps, avg_expert_len)
                    metrics["tcod/expert_prefix_len"] = expert_prefix_len

                    # Inject expert_prefix_steps into env_kwargs for each sample
                    batch_size = len(gen_batch.batch['input_ids'])
                    if "env_kwargs" not in gen_batch.non_tensor_batch:
                        gen_batch.non_tensor_batch["env_kwargs"] = np.array([{} for _ in range(batch_size)], dtype=object)
                    for i in range(batch_size):
                        if gen_batch.non_tensor_batch["env_kwargs"][i] is None:
                            gen_batch.non_tensor_batch["env_kwargs"][i] = {}
                        gen_batch.non_tensor_batch["env_kwargs"][i]["expert_prefix_steps"] = expert_prefix_len
                        # Pass expert actions if available
                        if expert_actions_key in gen_batch.non_tensor_batch:
                            gen_batch.non_tensor_batch["env_kwargs"][i]["expert_actions"] = gen_batch.non_tensor_batch[expert_actions_key][i]

                # For f2b: temporarily override max_steps
                with open_dict(self.config):
                    original_max_steps = self.config.env.max_steps
                    self.config.env.max_steps = effective_max_steps

                metrics["tcod/effective_max_steps"] = effective_max_steps
                metrics["tcod/strategy"] = 0 if self.tcod_strategy == "f2b" else 1

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    # Restore original max_steps
                    with open_dict(self.config):
                        self.config.env.max_steps = original_max_steps

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

                    # ---- TCOD: Teacher forward pass (external teacher) ----
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

                        # Compute GRPO advantages (needed as base even if opd_only)
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

                        # ---- TCOD: Replace advantage with OPD signal ----
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

                        modified_adv, opd_metrics = compute_opd_advantage(
                            grpo_advantages=grpo_adv,
                            student_log_probs=student_log_probs,
                            teacher_log_probs=teacher_lp,
                            response_mask=response_mask,
                            mode=self.sod_mode,
                            opd_coef=self.sod_opd_coef,
                            opd_only=self.sod_opd_only,
                            traj_uids=traj_uids,
                            turn_steps=turn_steps,
                        )
                        batch.batch["advantages"] = modified_adv
                        metrics.update(opd_metrics)

                        # Teacher-student gap metrics
                        delta_t = (teacher_lp - student_log_probs) * response_mask
                        metrics["tcod/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                        metrics["tcod/teacher_student_gap_std"] = masked_mean(delta_t ** 2, response_mask).sqrt().item()

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
