"""
SOD (Step-wise On-policy Distillation) utilities.

Implements:
1. Step boundary extraction from response_mask (multi-turn agent turns).
2. Step-wise adaptive OPD weight computation (w_k).
3. OPD advantage fusion (gated β or step-wise weighted).

Reference: SOD paper (arXiv:2605.07725)
"""

import torch
from typing import Optional

from verl.utils.torch_functional import masked_mean


def _masked_signed_stats(
    tensor: torch.Tensor,
    response_mask: torch.Tensor,
    prefix: str,
) -> dict:
    """Compute signed mean/std/min/max plus absolute-mean over masked tokens.

    Only positions where response_mask > 0 contribute. If no valid tokens exist,
    all stats fall back to 0.0 to avoid NaN polluting wandb plots.

    Args:
        tensor: (bsz, seq_len) tensor of per-token values (e.g. opd_term, rl_term).
        response_mask: (bsz, seq_len) binary mask, 1 for valid response tokens.
        prefix: metric key prefix, e.g. "hybrid/opd" -> emits
            "hybrid/opd_mean", "hybrid/opd_std", "hybrid/opd_min",
            "hybrid/opd_max", "hybrid/opd_abs_mean".

    Returns:
        dict of float metrics.
    """
    out: dict = {}
    with torch.no_grad():
        mask_bool = response_mask.bool()
        valid = tensor[mask_bool]
        n_valid = valid.numel()
        if n_valid == 0:
            out[f"{prefix}_mean"] = 0.0
            out[f"{prefix}_std"] = 0.0
            out[f"{prefix}_min"] = 0.0
            out[f"{prefix}_max"] = 0.0
            out[f"{prefix}_abs_mean"] = 0.0
            return out
        valid_f = valid.float()
        out[f"{prefix}_mean"] = valid_f.mean().item()
        # unbiased=False to be consistent across small batches
        out[f"{prefix}_std"] = (
            valid_f.std(unbiased=False).item() if n_valid > 1 else 0.0
        )
        out[f"{prefix}_min"] = valid_f.min().item()
        out[f"{prefix}_max"] = valid_f.max().item()
        out[f"{prefix}_abs_mean"] = valid_f.abs().mean().item()
    return out


def compute_per_turn_disagreement_entropy(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    turn_steps: Optional[list] = None,
    prefix: str = "sod_turn",
    max_turn_log: int = 50,
) -> dict:
    """Per-turn-index disagreement proxy d_k and entropy proxies h_k.

    SDAR-aware: each sample corresponds to exactly one agent turn, and its
    turn index within the trajectory is given by ``turn_steps`` (0-indexed).

    Per sample i, over valid response tokens (response_mask > 0):

        d_i      = (1/N_i) * sum_t | log pi_student(a_t|s_t) - log pi_teacher(a_t|s_t) |
                   -> disagreement proxy (sampled-token |Δ log p|, a KL proxy)
        h_stu_i  = (1/N_i) * sum_t ( - log pi_student(a_t|s_t) )
                   -> student uncertainty proxy (sampled-token neg-log-prob entropy)
        h_tea_i  = (1/N_i) * sum_t ( - log pi_teacher(a_t|s_t) )
                   -> teacher uncertainty proxy (symmetric counterpart of h_stu_i)

    Samples sharing the same turn index are then aggregated (token-weighted mean)
    so each metric line can be tracked as a function of turn index across training.

    Emits keys:
        {prefix}/disagreement/turn_{k}
        {prefix}/student_entropy/turn_{k}
        {prefix}/teacher_entropy/turn_{k}
        {prefix}/count/turn_{k}          (number of samples at turn k)
    plus batch-level (token-weighted) means:
        {prefix}/disagreement_mean
        {prefix}/student_entropy_mean
        {prefix}/teacher_entropy_mean
        {prefix}/max_turn                (largest observed turn index in batch)

    Turn indices >= ``max_turn_log`` are folded into a single tail bucket
    ``turn_{max_turn_log}p`` to bound the number of emitted keys.

    Args:
        student_log_probs: (bsz, seq_len) student log-probs.
        teacher_log_probs: (bsz, seq_len) teacher log-probs.
        response_mask: (bsz, seq_len) binary mask, 1 for valid response tokens.
        turn_steps: (bsz,) per-sample turn index (0-indexed). If None, every
            sample is treated as turn 0 (only batch-level means are meaningful).
        prefix: metric key prefix.
        max_turn_log: turn indices >= this value share one tail bucket.

    Returns:
        dict of float metrics.
    """
    from collections import defaultdict

    metrics: dict = {}
    bsz = response_mask.shape[0]
    with torch.no_grad():
        mask = response_mask.float()
        n_tok = mask.sum(dim=1)  # (bsz,)

        abs_diff_sum = ((student_log_probs - teacher_log_probs).abs() * mask).sum(dim=1)  # (bsz,)
        neg_stu_sum = ((-student_log_probs) * mask).sum(dim=1)  # (bsz,)
        neg_tea_sum = ((-teacher_log_probs) * mask).sum(dim=1)  # (bsz,)

        # ---- Batch-level token-weighted means ----
        total_tok = n_tok.sum().clamp(min=1)
        metrics[f"{prefix}/disagreement_mean"] = (abs_diff_sum.sum() / total_tok).item()
        metrics[f"{prefix}/student_entropy_mean"] = (neg_stu_sum.sum() / total_tok).item()
        metrics[f"{prefix}/teacher_entropy_mean"] = (neg_tea_sum.sum() / total_tok).item()

        # ---- Group samples by turn index ----
        groups: dict = defaultdict(list)
        max_turn = 0
        for i in range(bsz):
            if n_tok[i].item() <= 0:
                continue
            if turn_steps is not None and turn_steps[i] is not None:
                k = int(turn_steps[i])
            else:
                k = 0
            max_turn = max(max_turn, k)
            bucket = k if k < max_turn_log else max_turn_log
            groups[bucket].append(i)

        metrics[f"{prefix}/max_turn"] = float(max_turn)

        for bucket, idxs in groups.items():
            idx_t = torch.as_tensor(idxs, device=n_tok.device)
            tok_k = n_tok[idx_t].sum().clamp(min=1)
            suffix = f"turn_{bucket}" if bucket < max_turn_log else f"turn_{max_turn_log}p"
            metrics[f"{prefix}/disagreement/{suffix}"] = (abs_diff_sum[idx_t].sum() / tok_k).item()
            metrics[f"{prefix}/student_entropy/{suffix}"] = (neg_stu_sum[idx_t].sum() / tok_k).item()
            metrics[f"{prefix}/teacher_entropy/{suffix}"] = (neg_tea_sum[idx_t].sum() / tok_k).item()
            metrics[f"{prefix}/count/{suffix}"] = float(len(idxs))

    return metrics


def extract_step_boundaries(response_mask: torch.Tensor) -> list[list[tuple[int, int]]]:
    """Extract per-sample step (assistant turn) boundaries from response_mask.

    In multi-turn agent loops, response_mask is 1 for assistant tokens and 0 for
    tool-response / padding tokens. Each contiguous run of 1s is one "step".

    Args:
        response_mask: (batch_size, seq_len), binary mask.

    Returns:
        List (length = batch_size) of lists of (start, end) index pairs.
        Each (start, end) pair satisfies: response_mask[i, start:end] are all 1.
    """
    batch_boundaries = []
    bsz, seq_len = response_mask.shape
    for i in range(bsz):
        mask_i = response_mask[i]  # (seq_len,)
        boundaries = []
        in_segment = False
        seg_start = 0
        for t in range(seq_len):
            if mask_i[t].item() == 1 and not in_segment:
                seg_start = t
                in_segment = True
            elif mask_i[t].item() == 0 and in_segment:
                boundaries.append((seg_start, t))
                in_segment = False
        if in_segment:
            boundaries.append((seg_start, seq_len))
        batch_boundaries.append(boundaries)
    return batch_boundaries


def compute_stepwise_opd_weights(
    old_log_probs: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    traj_uids: Optional[list] = None,
    turn_steps: Optional[list] = None,
    epsilon: float = 1e-6,
    delta: float = 0.5,
) -> tuple[torch.Tensor, list[dict]]:
    """Compute per-token step-wise OPD weights w_k for each trajectory.

    SDAR-aware: each sample = one turn. Trajectories are grouped by `traj_uids`.

    For each trajectory τ:
      1. Compute d_k = mean(|log π_θ - log π_teacher|) per turn k.
      2. w_1 = 1.0
         w_k = min(∏_{u=1}^{k-1} (d_u+ε)/(d_{u+1}+ε),  1+δ)
      3. Broadcast w_k to all response tokens of sample at turn k.

    Args:
        old_log_probs: (bsz, seq_len) student log-probs.
        ref_log_prob:  (bsz, seq_len) teacher log-probs.
        response_mask: (bsz, seq_len) binary mask.
        traj_uids: (bsz,) trajectory uids. Same uid → same trajectory.
            If None, falls back to per-sample (each sample = one trajectory),
            which makes step-wise weights all 1.0 (degenerate to uniform).
        turn_steps: (bsz,) per-sample turn index (0-indexed). Used for ordering.
        epsilon, delta: as in formula.

    Returns:
        weight_mask: (bsz, seq_len) per-token w_k weights.
        log_info: list of dicts (one per trajectory) with d_values, w_values, etc.
    """
    bsz, seq_len = response_mask.shape
    weight_mask = torch.zeros_like(response_mask, dtype=torch.float32)

    # Per-sample (per-turn) divergence d_k = mean(|student - teacher|) over response tokens
    abs_diff = (old_log_probs - ref_log_prob).abs()  # (bsz, seq_len)
    per_sample_d = torch.zeros(bsz)
    per_sample_n_tokens = torch.zeros(bsz, dtype=torch.long)
    for i in range(bsz):
        mask_i = response_mask[i].float()
        n_tok = mask_i.sum().item()
        per_sample_n_tokens[i] = int(n_tok)
        if n_tok > 0:
            per_sample_d[i] = (abs_diff[i] * mask_i).sum().item() / n_tok

    # Group samples by trajectory
    if traj_uids is None:
        # Fallback: each sample is its own trajectory → w=1.0 for all
        traj_groups = {f"sample_{i}": [i] for i in range(bsz)}
    else:
        traj_groups = {}
        for i in range(bsz):
            uid = str(traj_uids[i])
            traj_groups.setdefault(uid, []).append(i)
        if turn_steps is not None:
            for uid in traj_groups:
                traj_groups[uid].sort(key=lambda i: int(turn_steps[i]) if turn_steps[i] is not None else 0)

    log_info = []

    # Compute w_k per trajectory
    for uid, indices in traj_groups.items():
        K = len(indices)
        d_values = [per_sample_d[i].item() for i in indices]
        n_tokens_per_step = [int(per_sample_n_tokens[i].item()) for i in indices]

        # w_1 = 1.0; w_k = min(cum_prod, 1+delta)
        w_values = [1.0]
        if K > 1:
            cum_prod = 1.0
            for u in range(K - 1):
                ratio = (d_values[u] + epsilon) / (d_values[u + 1] + epsilon)
                cum_prod *= ratio
                w_k = min(cum_prod, 1.0 + delta)
                w_values.append(w_k)

        # Broadcast w_k to each sample's response tokens
        for k, sample_idx in enumerate(indices):
            weight_mask[sample_idx] = response_mask[sample_idx].float() * w_values[k]

        log_info.append({
            "traj_uid": uid,
            "n_steps": K,
            "d_values": d_values,
            "w_values": w_values,
            "n_tokens_per_step": n_tokens_per_step,
        })

    return weight_mask, log_info


def compute_opd_advantage(
    grpo_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    mode: str = "stepwise",
    # Gated OPD (mode="gated") parameters
    gamma: float = 1.0,
    beta_min: float = 0.0,
    beta_max: float = 0.3,
    # Step-wise OPD (mode="stepwise") and uniform (mode="uniform") parameters
    opd_coef: float = 1.0,
    epsilon: float = 1e-6,
    delta: float = 0.5,
    opd_only: bool = False,
    # SDAR-specific: per-sample trajectory grouping (used by stepwise mode)
    traj_uids: Optional[list] = None,
    turn_steps: Optional[list] = None,
) -> tuple[torch.Tensor, dict]:
    """Compute SOD/OPD token-level advantage by fusing GRPO advantage with distillation signal.

    Three modes:
      - "uniform": Plain OPD with a fixed coefficient for all tokens.
          A_total = A_grpo + opd_coef * (log_teacher - log_student)
      - "gated": OPD with sigmoid-gated per-sample β weight.
          A_total = A_grpo + β(A_grpo) * (log_teacher - log_student)
      - "stepwise": SOD with step-wise adaptive weights.
          A_total = A_grpo + opd_coef * w_k * (log_teacher - log_student)

    Args:
        grpo_advantages: (bsz, seq_len) or (bsz,) GRPO advantages.
        student_log_probs: (bsz, seq_len) student log-probs.
        teacher_log_probs: (bsz, seq_len) teacher log-probs.
        response_mask: (bsz, seq_len) binary mask.
        mode: "uniform", "gated", or "stepwise".
        gamma, beta_min, beta_max: gated OPD parameters.
        opd_coef, epsilon, delta: step-wise/uniform OPD parameters.
        opd_only: if True, use only OPD signal (no GRPO advantage).

    Returns:
        A_total: (bsz, seq_len) fused token-level advantages.
        metrics: dict with SOD/OPD statistics.
    """
    # Raw KL distillation signal: (teacher - student), masked
    raw_local_adv = (teacher_log_probs - student_log_probs) * response_mask
    metrics = {}

    if mode == "stepwise":
        # Step-wise weighted OPD (SOD)
        stepwise_weights, stepwise_log_info = compute_stepwise_opd_weights(
            old_log_probs=student_log_probs,
            ref_log_prob=teacher_log_probs,
            response_mask=response_mask,
            traj_uids=traj_uids,
            turn_steps=turn_steps,
            epsilon=epsilon,
            delta=delta,
        )
        stepwise_weights = stepwise_weights.to(raw_local_adv.device)

        # Weighted OPD signal: w_k * opd_coef * (log_teacher - log_student) per token
        weighted_opd = opd_coef * stepwise_weights * raw_local_adv

        # GRPO term used in the final fusion (broadcast to (bsz, seq_len) and masked)
        if grpo_advantages.dim() == 1:
            grpo_term = grpo_advantages.unsqueeze(1).expand_as(response_mask) * response_mask
        else:
            grpo_term = grpo_advantages * response_mask

        if opd_only:
            A_total = weighted_opd
        else:
            A_total = grpo_term + weighted_opd

        # Metrics
        with torch.no_grad():
            mask_sum = response_mask.sum().clamp(min=1)
            metrics["sod/raw_kl_mean"] = (raw_local_adv.abs().sum() / mask_sum).item()
            metrics["sod/weighted_opd_mean"] = (weighted_opd.abs().sum() / mask_sum).item()
            n_steps_list = [info["n_steps"] for info in stepwise_log_info]
            metrics["sod/avg_n_steps"] = sum(n_steps_list) / max(len(n_steps_list), 1)
            w_all = [w for info in stepwise_log_info for w in info["w_values"]]
            if w_all:
                metrics["sod/avg_weight"] = sum(w_all) / len(w_all)
                metrics["sod/min_weight"] = min(w_all)
                metrics["sod/max_weight"] = max(w_all)

            # ---- Per-component signed statistics (for analyzing GRPO vs OPD scale) ----
            metrics.update(_masked_signed_stats(grpo_term, response_mask, "sod/grpo"))
            metrics.update(_masked_signed_stats(weighted_opd, response_mask, "sod/opd"))
            opd_abs = weighted_opd.abs().sum().item()
            grpo_abs = grpo_term.abs().sum().item()
            metrics["sod/opd_grpo_ratio"] = opd_abs / max(grpo_abs, 1e-8)

    elif mode == "gated":
        # Gated OPD (original OPD)
        base_adv = masked_mean(grpo_advantages, response_mask, axis=1)  # (bsz,)
        gate = torch.sigmoid(-gamma * base_adv)
        beta_jk = beta_min + (beta_max - beta_min) * gate  # (bsz,)
        beta_token = beta_jk.unsqueeze(1)  # (bsz, 1)

        # Per-token OPD term and GRPO term used in the final fusion
        opd_term = beta_token * raw_local_adv  # already masked via raw_local_adv
        if grpo_advantages.dim() == 1:
            grpo_term = grpo_advantages.unsqueeze(1).expand_as(response_mask) * response_mask
        else:
            grpo_term = grpo_advantages * response_mask

        A_total = grpo_term + opd_term

        # Metrics
        with torch.no_grad():
            metrics["opd/beta_mean"] = beta_jk.mean().item()
            metrics["opd/beta_min"] = beta_jk.min().item()
            metrics["opd/beta_max"] = beta_jk.max().item()
            mask_sum = response_mask.sum().clamp(min=1)
            metrics["opd/raw_kl_mean"] = (raw_local_adv.abs().sum() / mask_sum).item()

            # ---- Per-component signed statistics ----
            metrics.update(_masked_signed_stats(grpo_term, response_mask, "opd/grpo"))
            metrics.update(_masked_signed_stats(opd_term, response_mask, "opd/opd"))
            opd_abs = opd_term.abs().sum().item()
            grpo_abs = grpo_term.abs().sum().item()
            metrics["opd/opd_grpo_ratio"] = opd_abs / max(grpo_abs, 1e-8)

    elif mode == "uniform":
        # Plain OPD: fixed coefficient, all tokens weighted equally
        uniform_opd = opd_coef * raw_local_adv

        if grpo_advantages.dim() == 1:
            grpo_term = grpo_advantages.unsqueeze(1).expand_as(response_mask) * response_mask
        else:
            grpo_term = grpo_advantages * response_mask

        if opd_only:
            A_total = uniform_opd
        else:
            A_total = grpo_term + uniform_opd

        # Metrics
        with torch.no_grad():
            mask_sum = response_mask.sum().clamp(min=1)
            metrics["opd/coef"] = opd_coef
            metrics["opd/raw_kl_mean"] = (raw_local_adv.abs().sum() / mask_sum).item()
            metrics["opd/uniform_opd_mean"] = (uniform_opd.abs().sum() / mask_sum).item()

            # ---- Per-component signed statistics ----
            metrics.update(_masked_signed_stats(grpo_term, response_mask, "opd/grpo"))
            metrics.update(_masked_signed_stats(uniform_opd, response_mask, "opd/opd"))
            opd_abs = uniform_opd.abs().sum().item()
            grpo_abs = grpo_term.abs().sum().item()
            metrics["opd/opd_grpo_ratio"] = opd_abs / max(grpo_abs, 1e-8)

    else:
        raise ValueError(f"Unknown OPD mode: {mode}. Expected 'uniform', 'gated', or 'stepwise'.")

    return A_total, metrics
