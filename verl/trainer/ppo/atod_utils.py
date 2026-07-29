"""
ATOD OPD+GRPO with TIP weighting and Soft Curriculum utilities.

Implements the improved atod advantage:
  A_final = [soft_w(k,s) × kl_coef(s) × (log_teacher - log_student) × TIP(k)
             + rl_coef(s) × GRPO_normed_reward] × response_mask

Key improvement over TCOD original: soft curriculum only modulates the OPD term,
not the RL term, because RL reward comes from the environment (always reliable),
while OPD signal degrades when student diverges from teacher in later turns.

Components:
1. TIP (Turn Importance Profiling): per-turn weight from Soft-OR(divergence, entropy)
2. Soft Curriculum: sigmoid-based f2b weight that softly downweights far turns for OPD
3. GRPO RL signal: K-rollout group-normalized environment reward
4. Linear coefficient annealing (optional): linearly decay kl_coef from kl_init to
   kl_min and grow rl_coef from rl_init to rl_max over the first T training steps.
   The intuition is that early in training the student is far from the teacher and
   benefits from a strong distillation signal to quickly approach teacher-level
   performance, while later in training the GRPO/RL signal should dominate so the
   student can break through the teacher upper bound.
"""

import torch
import math
from typing import Optional

from verl.trainer.ppo.sod_utils import extract_step_boundaries, _masked_signed_stats


# ---------------------------------------------------------------------------
# Within-turn return-to-go helper (mirrors verl-0.6.1-OPD's seq-level KL impl)
# ---------------------------------------------------------------------------

def _discounted_return_to_go(
    rewards: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Discounted return-to-go (right cumulative sum with discount).

    For each (b, t):
        returns[b, t] = sum_{i=t}^{T-1} gamma^(i-t) * rewards[b, i]

    `mask` is used to reset the running return at invalid positions so that
    padding / post-EOS tokens do not leak credit backwards. In the SDAR atod
    setup each sample is one turn and `mask` is the per-sample response_mask,
    so this naturally computes a within-turn RTG.

    Args:
        rewards: (bsz, seq_len) per-token rewards.
        mask: (bsz, seq_len) response mask (1 valid, 0 invalid).
        gamma: discount factor.

    Returns:
        returns: (bsz, seq_len) discounted return-to-go.
    """
    if gamma == 1.0:
        # Vectorized fast path for the undiscounted case.
        return (rewards * mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

    T = rewards.shape[1]
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        running = rewards[:, t] + gamma * running
        returns[:, t] = running
        # Reset running at invalid (mask=0) positions so credit does not flow
        # across padding / turn boundaries.
        running = running * mask[:, t]
    return returns


def compute_tip_weights(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    traj_uids: Optional[list] = None,
    turn_steps: Optional[list] = None,
    tip_rho: float = 1.0,
    tip_min_turns: int = 3,
    tip_min_divergence: float = 0.01,
    tip_smoothing: float = 0.0,
    # ---- T-DUR budget-preserving reparameterization (mean-preserving reweight) ----
    tip_mean_preserve: bool = False,
    tip_w_min: float = 0.5,
    tip_w_max: float = 2.0,
    tip_eps: float = 1e-8,
    max_turn_positions: int = 50,
    # ---- Ablation: which signal(s) to use for per-turn importance scoring ----
    # "softor"          -> 1 - (1-nd)*(1-ne)   (original Soft-OR, uses BOTH)
    # "entropy_only"    -> ne                  (ablation: entropy only)
    # "divergence_only" -> nd                  (ablation: teacher-student divergence only)
    tip_score_mode: str = "softor",
) -> tuple[torch.Tensor, dict]:
    """Compute TIP (Turn Importance Profiling) per-turn weights.

    SDAR-aware: each sample in the batch is ONE turn. Multiple turns of the same
    trajectory are grouped by `traj_uids`, and TIP statistics are computed within
    each trajectory group.

    For each turn k in trajectory τ:
      divergence_k = mean(|student_logp - teacher_logp|)  over response tokens
      entropy_proxy_k = mean(-student_logp)               over response tokens

      After min-max normalization within trajectory τ:
      importance_k = 1 - (1 - norm_ent_k) * (1 - norm_div_k)   # Soft-OR

    Args:
        student_log_probs: (bsz, seq_len) student log-probs.
        teacher_log_probs: (bsz, seq_len) teacher log-probs.
        response_mask: (bsz, seq_len) binary mask (response tokens).
        traj_uids: (bsz,) trajectory unique ids. Samples sharing the same uid
            belong to the same trajectory and are treated as multiple turns.
            If None, falls back to per-sample (each sample = one trajectory),
            which makes TIP a no-op.
        turn_steps: (bsz,) per-sample turn index within trajectory (0-indexed).
            Used to order turns within each trajectory.
        tip_rho: Top-ρ fraction (1.0 = continuous weights).
        tip_min_turns: Skip TIP for trajectories with fewer turns.
        tip_min_divergence: Skip TIP if max divergence below threshold.
        tip_smoothing: Floor weight via smoothing.
        tip_mean_preserve: If True, apply a token-weighted mean-preserving
            reparameterization so the per-token OPD budget mean stays ~1 within
            each trajectory (up-weight > 1, down-weight < 1), instead of using the
            raw Soft-OR priority z_k ∈ [0, 1] which can only down-weight and
            systematically shrinks the overall OPD strength. Only applied in the
            continuous-weight regime (tip_rho >= 1.0); it is a no-op when Top-ρ
            hard selection is active, to avoid resurrecting hard-dropped turns.
            Procedure per trajectory τ with turns k and token counts N_k:
              z̄_τ = Σ_j N_j z_j / Σ_j N_j
              r_k  = z_k / (z̄_τ + eps)
              r̂_k  = clip(r_k, w_min, w_max)
              w_k  = r̂_k / ( (Σ_j N_j r̂_j / Σ_j N_j) + eps )
        tip_w_min: Lower clip bound for the reparameterized weight (default 0.5).
        tip_w_max: Upper clip bound for the reparameterized weight (default 2.0).
        tip_eps: Numerical epsilon used in the (re-)normalization (default 1e-8).

    Returns:
        tip_weight_mask: (bsz, seq_len) per-token TIP weights.
        metrics: dict with TIP statistics.
    """
    bsz, seq_len = response_mask.shape
    tip_weight_mask = torch.ones_like(response_mask, dtype=torch.float32)
    metrics = {}

    # Step 0: compute per-sample (per-turn) divergence and entropy
    per_sample_div = torch.zeros(bsz)
    per_sample_ent = torch.zeros(bsz)
    for i in range(bsz):
        mask_i = response_mask[i].float()
        n_tokens = mask_i.sum().item()
        if n_tokens == 0:
            continue
        per_sample_div[i] = ((student_log_probs[i] - teacher_log_probs[i]).abs() * mask_i).sum().item() / n_tokens
        per_sample_ent[i] = ((-student_log_probs[i]) * mask_i).sum().item() / n_tokens

    # Step 1: group samples by trajectory uid
    if traj_uids is None:
        # Fallback: each sample is its own trajectory → TIP is a no-op
        traj_groups = {f"sample_{i}": [i] for i in range(bsz)}
    else:
        traj_groups = {}
        for i in range(bsz):
            uid = str(traj_uids[i])
            traj_groups.setdefault(uid, []).append(i)
        # Within each group, sort by turn_step if available
        if turn_steps is not None:
            for uid in traj_groups:
                traj_groups[uid].sort(key=lambda i: int(turn_steps[i]) if turn_steps[i] is not None else 0)

    # Step 2: per-trajectory TIP computation
    n_skipped_few_turns = 0
    n_skipped_low_div = 0
    n_computed = 0
    n_traj_total = len(traj_groups)
    all_importance = []
    all_max_div = []
    # Collect (turn_position, importance) for per-turn-position analysis
    per_turn_position_weights = []  # list of (turn_position, importance)

    for uid, indices in traj_groups.items():
        n_turns = len(indices)

        # Guard: too few turns
        if n_turns < tip_min_turns:
            n_skipped_few_turns += 1
            continue

        # Per-turn signals (within this trajectory)
        divergences = [per_sample_div[i].item() for i in indices]
        entropies = [per_sample_ent[i].item() for i in indices]

        max_div = max(divergences)
        all_max_div.append(max_div)

        # Guard: divergence too small
        if max_div < tip_min_divergence:
            n_skipped_low_div += 1
            continue

        # Min-Max normalization within trajectory
        min_div, max_div_v = min(divergences), max(divergences)
        min_ent, max_ent = min(entropies), max(entropies)

        norm_divs = [
            (d - min_div) / (max_div_v - min_div) if max_div_v - min_div > 1e-8 else 0.5
            for d in divergences
        ]
        norm_ents = [
            (e - min_ent) / (max_ent - min_ent) if max_ent - min_ent > 1e-8 else 0.5
            for e in entropies
        ]

        # Soft-OR (or its ablations, controlled by `tip_score_mode`).
        #   softor:          1 - (1-nd)*(1-ne)  -- original, uses BOTH signals
        #   entropy_only:    ne                 -- ablation: entropy only
        #   divergence_only: nd                 -- ablation: teacher-student divergence only
        if tip_score_mode == "entropy_only":
            importances = list(norm_ents)
        elif tip_score_mode == "divergence_only":
            importances = list(norm_divs)
        else:  # "softor" (default, original behavior)
            importances = [1.0 - (1.0 - nd) * (1.0 - ne) for nd, ne in zip(norm_divs, norm_ents)]

        # Optional Top-ρ
        if tip_rho < 1.0:
            k_keep = max(1, int(math.ceil(n_turns * tip_rho)))
            threshold = sorted(importances, reverse=True)[k_keep - 1]
            importances = [imp if imp >= threshold else 0.0 for imp in importances]

        # Optional smoothing
        if tip_smoothing > 0:
            importances = [(1 - tip_smoothing) * imp + tip_smoothing for imp in importances]

        # ---- T-DUR mean-preserving reparameterization (budget-preserving) ----
        # Only valid in the continuous-weight regime (tip_rho >= 1.0). When Top-ρ
        # hard selection is active we skip it, otherwise the clip lower bound would
        # revive turns whose importance was intentionally zeroed out.
        if tip_mean_preserve and tip_rho >= 1.0:
            # token counts N_k per turn, in the SAME order as `indices`/`importances`
            n_tokens_per_turn = [float(response_mask[idx].sum().item()) for idx in indices]
            total_tokens = sum(n_tokens_per_turn)
            if total_tokens > 0:
                # token-weighted trajectory mean of raw priorities z_k
                z_bar = sum(n * z for n, z in zip(n_tokens_per_turn, importances)) / total_tokens
                # mean-preserving reparam -> up/down-weight around 1.0
                r = [z / (z_bar + tip_eps) for z in importances]
                # clip to [w_min, w_max]
                r = [min(tip_w_max, max(tip_w_min, rk)) for rk in r]
                # token-weighted re-normalization so the token-weighted mean stays ~1
                r_bar = sum(n * rk for n, rk in zip(n_tokens_per_turn, r)) / total_tokens
                importances = [rk / (r_bar + tip_eps) for rk in r]

        all_importance.extend(importances)
        n_computed += 1

        # Broadcast importance back to each sample's response_mask region
        for k, sample_idx in enumerate(indices):
            # Determine actual turn position (0-indexed)
            if turn_steps is not None and sample_idx < len(turn_steps) and turn_steps[sample_idx] is not None:
                turn_pos = int(turn_steps[sample_idx])
            else:
                turn_pos = k  # fallback to sorted position within trajectory
            per_turn_position_weights.append((turn_pos, importances[k]))

            tip_weight_mask[sample_idx] = response_mask[sample_idx].float() * importances[k] + \
                                           (1 - response_mask[sample_idx].float())  # non-response stays 1

    # Metrics
    with torch.no_grad():
        metrics["tip/n_traj_total"] = float(n_traj_total)
        metrics["tip/n_computed"] = float(n_computed)
        metrics["tip/n_skipped_few_turns"] = float(n_skipped_few_turns)
        metrics["tip/n_skipped_low_div"] = float(n_skipped_low_div)
        metrics["tip/active_ratio"] = n_computed / max(n_traj_total, 1)
        metrics["tip/avg_n_turns_per_traj"] = bsz / max(n_traj_total, 1)
        if all_max_div:
            metrics["tip/avg_max_div"] = sum(all_max_div) / len(all_max_div)
        if all_importance:
            metrics["tip/mean_importance"] = sum(all_importance) / len(all_importance)
            metrics["tip/min_importance"] = min(all_importance)
            metrics["tip/max_importance"] = max(all_importance)
        else:
            metrics["tip/mean_importance"] = 1.0
            metrics["tip/min_importance"] = 1.0
            metrics["tip/max_importance"] = 1.0

        # ---- Per-turn-position weight analysis (for tracking weight evolution
        #      across training stages). Each metric is keyed by turn position
        #      0..max_turn_positions-1, aggregated across all trajectories in
        #      the batch that have a turn at that position. ----
        turn_pos_importances: dict[int, list[float]] = {}
        for pos, imp in per_turn_position_weights:
            if 0 <= pos < max_turn_positions:
                turn_pos_importances.setdefault(pos, []).append(imp)

        for pos in sorted(turn_pos_importances.keys()):
            weights_at_pos = turn_pos_importances[pos]
            n_at_pos = len(weights_at_pos)
            mean_w = sum(weights_at_pos) / n_at_pos
            metrics[f"tip/turn_{pos}_mean_weight"] = mean_w
            metrics[f"tip/turn_{pos}_count"] = float(n_at_pos)
            if n_at_pos > 1:
                var_w = sum((w - mean_w) ** 2 for w in weights_at_pos) / n_at_pos
                metrics[f"tip/turn_{pos}_std_weight"] = var_w ** 0.5

    return tip_weight_mask, metrics


def compute_token_tip_weights(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    tip_min_divergence: float = 0.01,
    tip_smoothing: float = 0.0,
    entropy_clip_quantile: float = 0.98,
) -> tuple[torch.Tensor, dict]:
    """Token-level TIP weights (B1: isolates granularity vs. compute_tip_weights).

    Uses the SAME single-sample log-prob proxies as the turn-level version, so the
    only thing that changes vs. `compute_tip_weights` is the granularity at which
    Soft-OR is applied:
      - turn-level: average proxies within a turn -> one scalar per turn, broadcast
      - token-level: keep per-token proxies, normalize per-batch, Soft-OR per token

    Per-token proxies (response tokens only):
      h_t   = -student_log_probs[t]                 # NLL of sampled token (entropy proxy)
      d_t   = |student_log_probs[t] - teacher_log_probs[t]|  # |delta logp| (divergence proxy)

    Per-batch normalization (matches paper Sec. 6 closer than per-trajectory):
      h_t <- min(h_t, quantile(h_t over response tokens, entropy_clip_quantile))
      h_norm = (h_t - h_min) / (h_max - h_min)      # over response tokens only
      d_norm = (d_t - d_min) / (d_max - d_min)

    Soft-OR (parameter-free, same as paper Eq. 7):
      s_t = 1 - (1 - h_norm) * (1 - d_norm)         in [0, 1]

    Returned weights are SOFT (continuous), no Top-K hard selection. This is the
    minimal change vs. the turn-level baseline so any delta in training behavior
    can be attributed to granularity alone.

    Args:
        student_log_probs: (bsz, seq_len) student log-probs of sampled tokens.
        teacher_log_probs: (bsz, seq_len) teacher log-probs of sampled tokens.
        response_mask: (bsz, seq_len) binary mask, 1 for response tokens.
        tip_min_divergence: If max d over the batch is below this, return ones
            (TIP becomes a no-op for this batch). Mirrors the turn-level guard.
        tip_smoothing: Floor in [0, 1]. Final w_t = (1 - smoothing) * s_t + smoothing.
            Set 0.0 for pure Soft-OR. A small value (e.g. 0.1) prevents total
            signal collapse when most tokens have near-zero importance.
        entropy_clip_quantile: Top-quantile clipping for entropy outliers, e.g.
            0.98 = clip top 2% (paper default). Set 1.0 to disable.

    Returns:
        tip_weight_mask: (bsz, seq_len) per-token TIP weights, in [0, 1] outside
            response (set to 1.0 there to be neutral; downstream multiplies by
            response_mask anyway).
        metrics: dict with TIP statistics.
    """
    bsz, seq_len = response_mask.shape
    device = student_log_probs.device
    metrics: dict = {}

    # Default: identity weights (TIP no-op fallback).
    tip_weight_mask = torch.ones_like(response_mask, dtype=torch.float32, device=device)

    mask_bool = response_mask.bool()
    n_response = int(mask_bool.sum().item())
    metrics["tip_token/n_response_tokens"] = float(n_response)

    if n_response == 0:
        metrics["tip_token/active"] = 0.0
        metrics["tip_token/skipped_empty"] = 1.0
        metrics["tip_token/mean_importance"] = 1.0
        metrics["tip_token/min_importance"] = 1.0
        metrics["tip_token/max_importance"] = 1.0
        return tip_weight_mask, metrics

    # ---- per-token proxies on response tokens only ----
    # Use float32 for numerical stability of normalization.
    student_lp = student_log_probs.float()
    teacher_lp = teacher_log_probs.float()

    h_full = (-student_lp)                         # entropy proxy
    d_full = (student_lp - teacher_lp).abs()       # divergence proxy

    h_resp = h_full[mask_bool]                     # (n_response,)
    d_resp = d_full[mask_bool]

    # Guard: divergence too small over the whole batch -> no-op
    max_d = float(d_resp.max().item()) if d_resp.numel() > 0 else 0.0
    metrics["tip_token/max_div"] = max_d
    if max_d < tip_min_divergence:
        metrics["tip_token/active"] = 0.0
        metrics["tip_token/skipped_low_div"] = 1.0
        metrics["tip_token/mean_importance"] = 1.0
        metrics["tip_token/min_importance"] = 1.0
        metrics["tip_token/max_importance"] = 1.0
        return tip_weight_mask, metrics

    # ---- per-batch min-max with optional top-quantile clip on entropy ----
    if 0.0 < entropy_clip_quantile < 1.0 and h_resp.numel() > 1:
        # torch.quantile is exact; cheap on response tokens (typically <= a few k).
        h_clip = torch.quantile(h_resp, entropy_clip_quantile)
        h_resp_eff = h_resp.clamp(max=h_clip)
        h_full_eff = h_full.clamp(max=h_clip)
    else:
        h_resp_eff = h_resp
        h_full_eff = h_full

    h_min = h_resp_eff.min()
    h_max = h_resp_eff.max()
    d_min = d_resp.min()
    d_max = d_resp.max()

    h_range = (h_max - h_min).clamp(min=1e-8)
    d_range = (d_max - d_min).clamp(min=1e-8)

    h_norm = ((h_full_eff - h_min) / h_range).clamp(0.0, 1.0)
    d_norm = ((d_full - d_min) / d_range).clamp(0.0, 1.0)

    # Soft-OR
    s = 1.0 - (1.0 - h_norm) * (1.0 - d_norm)      # (bsz, seq_len), in [0, 1]

    # Optional smoothing floor
    if tip_smoothing > 0.0:
        s = (1.0 - tip_smoothing) * s + tip_smoothing

    # Apply only on response positions; keep 1.0 elsewhere (neutral).
    response_mask_f = response_mask.float()
    tip_weight_mask = s * response_mask_f + (1.0 - response_mask_f)

    # ---- metrics over response tokens ----
    with torch.no_grad():
        s_resp = s[mask_bool]
        h_norm_resp = h_norm[mask_bool]
        d_norm_resp = d_norm[mask_bool]

        metrics["tip_token/active"] = 1.0
        metrics["tip_token/mean_importance"] = float(s_resp.mean().item())
        metrics["tip_token/min_importance"] = float(s_resp.min().item())
        metrics["tip_token/max_importance"] = float(s_resp.max().item())
        metrics["tip_token/std_importance"] = float(s_resp.std(unbiased=False).item())

        metrics["tip_token/h_mean_raw"] = float(h_resp.mean().item())
        metrics["tip_token/h_max_raw"] = float(h_resp.max().item())
        metrics["tip_token/d_mean_raw"] = float(d_resp.mean().item())
        metrics["tip_token/d_max_raw"] = max_d

        # Region diagnostics (paper's two-axis taxonomy)
        # A: high entropy positions  (h_norm >= 0.5)
        # Q3: low entropy + high divergence  (h_norm < 0.5 and d_norm >= 0.5)
        region_a = (h_norm_resp >= 0.5).float()
        region_q3 = ((h_norm_resp < 0.5) & (d_norm_resp >= 0.5)).float()
        metrics["tip_token/region_a_ratio"] = float(region_a.mean().item())
        metrics["tip_token/region_q3_ratio"] = float(region_q3.mean().item())

        # Effective signal scaling: average weight * fraction kept (always 1.0 here
        # since soft, but exposed for parity with a future Top-K variant).
        metrics["tip_token/effective_scale"] = float(s_resp.mean().item())

    return tip_weight_mask, metrics


def compute_soft_curriculum_weights(
    response_mask: torch.Tensor,
    global_step: int,
    turn_steps: Optional[list] = None,
    checkpoint_steps: float = 6.0,
    softness: float = 1.0,
    bias: float = 0.5,
    min_weight: float = 0.05,
    max_weight: float = 1.0,
    warmup_steps: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Compute f2b-style soft curriculum weights per turn.

    SDAR-aware: uses `turn_steps` (per-sample turn index) directly.

    W(s) = 1 + s / checkpoint_steps
    soft_w(k) = clamp(sigmoid((W - k + bias) / softness), min_weight, max_weight)

    Args:
        response_mask: (bsz, seq_len) binary mask.
        global_step: Current training step.
        turn_steps: (bsz,) per-sample turn index (0-indexed). If None, treats
            each sample as turn 0.
        checkpoint_steps: How fast the window expands.
        softness: Sigmoid temperature.
        bias: Shift so frontier turn gets weight > 0.5.
        min_weight: Min weight for far turns.
        max_weight: Max weight.
        warmup_steps: Steps before curriculum activates.

    Returns:
        curriculum_mask: (bsz, seq_len) per-token curriculum weights.
        metrics: dict with curriculum statistics.
    """
    bsz, seq_len = response_mask.shape
    curriculum_mask = torch.ones_like(response_mask, dtype=torch.float32) * max_weight
    metrics = {}

    # Warmup: uniform max_weight
    if global_step < warmup_steps:
        metrics["curriculum/window"] = 0.0
        metrics["curriculum/in_warmup"] = 1.0
        return curriculum_mask, metrics

    W = 1.0 + global_step / checkpoint_steps
    metrics["curriculum/window"] = W

    all_weights = []
    for i in range(bsz):
        k = int(turn_steps[i]) if turn_steps is not None and turn_steps[i] is not None else 0
        z = (W - k + bias) / softness
        w = torch.sigmoid(torch.tensor(z)).item()
        w = max(min_weight, min(max_weight, w))
        # Apply only to response tokens; non-response stays as max_weight (irrelevant after mask)
        curriculum_mask[i] = response_mask[i].float() * w + (1 - response_mask[i].float()) * max_weight
        all_weights.append(w)

    with torch.no_grad():
        if all_weights:
            metrics["curriculum/mean_weight"] = sum(all_weights) / len(all_weights)
            metrics["curriculum/min_weight"] = min(all_weights)
            metrics["curriculum/max_weight"] = max(all_weights)

    return curriculum_mask, metrics


def _compute_linear_anneal_coefs(
    global_step: int,
    kl_init: float,
    kl_min: float,
    rl_init: float,
    rl_max: float,
    anneal_steps: int,
) -> tuple[float, float, float]:
    """Linearly anneal kl_coef from kl_init -> kl_min and rl_coef from rl_init -> rl_max.

    progress = min(global_step / anneal_steps, 1.0)
    kl_coef(s) = max(kl_min, kl_init - (kl_init - kl_min) * progress)
    rl_coef(s) = rl_init + (rl_max - rl_init) * progress

    After s >= anneal_steps the coefs are clamped to (kl_min, rl_max).
    Setting anneal_steps <= 0 disables annealing and returns (kl_init, rl_init).
    """
    if anneal_steps <= 0:
        return kl_init, rl_init, 0.0
    progress = min(max(global_step / float(anneal_steps), 0.0), 1.0)
    kl_eff = kl_init - (kl_init - kl_min) * progress
    kl_eff = max(kl_min, kl_eff)
    rl_eff = rl_init + (rl_max - rl_init) * progress
    return kl_eff, rl_eff, progress


def compute_atod_advantage(
    grpo_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    global_step: int,
    # SDAR-specific: per-sample trajectory grouping
    traj_uids: Optional[list] = None,
    turn_steps: Optional[list] = None,
    # OPD parameters
    kl_coef: float = 1.0,
    # RL parameters
    rl_coef: float = 0.3,
    # TIP parameters
    enable_tip: bool = True,
    tip_rho: float = 1.0,
    tip_min_turns: int = 3,
    tip_min_divergence: float = 0.01,
    tip_smoothing: float = 0.0,
    # T-DUR budget-preserving reparam (turn-level only; no-op when disabled)
    tip_mean_preserve: bool = False,
    tip_w_min: float = 0.5,
    tip_w_max: float = 2.0,
    tip_eps: float = 1e-8,
    # Granularity dispatcher: "turn" (default, original behavior) or "token" (B1)
    tip_granularity: str = "turn",
    # Token-level only: top-quantile clip on entropy outliers (paper default 0.98)
    tip_entropy_clip_quantile: float = 0.98,
    # Ablation: per-turn importance scoring signal.
    # "softor" (default) | "entropy_only" | "divergence_only". Only affects the
    # turn-level path (tip_granularity="turn"); ignored for token-level TIP.
    tip_score_mode: str = "softor",
    # Soft curriculum parameters
    enable_soft_curriculum: bool = True,
    curriculum_checkpoint_steps: float = 6.0,
    curriculum_softness: float = 1.0,
    curriculum_bias: float = 0.5,
    curriculum_min_weight: float = 0.05,
    curriculum_max_weight: float = 1.0,
    curriculum_warmup_steps: int = 0,
    # ---- Linear coefficient annealing (NEW) ----
    enable_coef_anneal: bool = False,
    coef_anneal_steps: int = 80,
    kl_coef_min: float = 0.1,
    rl_coef_max: float = 2.0,
    # ---- Within-turn RTG (Plan B: turn-aware sequence-level KL) ----
    # When enabled, the OPD per-token signal becomes
    #   raw_kl_t = sum_{i=t}^{T_k} gamma^(i-t) * (log p_tea_i - log p_stu_i)
    # i.e. "current token logp diff + discounted future tokens' logp diff",
    # computed within each sample's response_mask boundary (= one turn in SDAR).
    # Default OFF → falls back to the original token-level behavior.
    enable_within_turn_rtg: bool = False,
    opd_gamma: float = 0.99,
    opd_length_normalization: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Compute atod advantage: OPD distillation + GRPO RL with TIP and soft curriculum.

    Improved formula (soft curriculum only on OPD term):
      A_final = [soft_w(k,s) × kl_coef × (log_teacher - log_student) × TIP(k)
                 + rl_coef × A_GRPO] × response_mask

    SDAR-aware: each sample = one turn. Trajectories are grouped by traj_uids
    for TIP statistics; turn_steps provides per-sample turn index for curriculum.

    Args:
        grpo_advantages: (bsz, seq_len) GRPO advantages.
        student_log_probs: (bsz, seq_len) student log-probs.
        teacher_log_probs: (bsz, seq_len) teacher log-probs.
        response_mask: (bsz, seq_len) binary mask.
        global_step: Current training step.
        traj_uids: (bsz,) trajectory uids, used by TIP for trajectory grouping.
        turn_steps: (bsz,) per-sample turn index (0-indexed), used by both TIP
            (for ordering) and soft curriculum (as turn position).
        (other params): See individual functions above.

    Returns:
        A_total: (bsz, seq_len) final atod advantage.
        metrics: dict.
    """
    metrics = {}

    # ---- 0. Linear coefficient annealing (kl_coef ↓, rl_coef ↑ over time) ----
    if enable_coef_anneal:
        eff_kl_coef, eff_rl_coef, anneal_progress = _compute_linear_anneal_coefs(
            global_step=global_step,
            kl_init=kl_coef,
            kl_min=kl_coef_min,
            rl_init=rl_coef,
            rl_max=rl_coef_max,
            anneal_steps=coef_anneal_steps,
        )
    else:
        eff_kl_coef, eff_rl_coef, anneal_progress = kl_coef, rl_coef, 0.0

    # ---- 1. OPD distillation term ----
    # Per-token logp difference (k1-style estimator).
    delta_logp = (teacher_log_probs - student_log_probs)

    if enable_within_turn_rtg:
        # Within-turn discounted return-to-go (Plan B).
        # Rewards are detached because they will play the role of an advantage
        # term plugged into the PPO objective; gradient should only flow through
        # log p_student via the policy ratio, not through the reward signal.
        per_token_reward = (delta_logp * response_mask).detach()
        rtg = _discounted_return_to_go(per_token_reward, response_mask, float(opd_gamma))
        if opd_length_normalization:
            # Discounted "effective length" L_t = sum_{i=t}^{T_k} gamma^(i-t) * mask_i
            lengths = _discounted_return_to_go(
                response_mask.float(), response_mask, float(opd_gamma)
            )
            rtg = rtg / lengths.clamp(min=1.0)
        raw_kl = rtg * response_mask
        metrics["atod/opd_signal"] = 1.0  # 1 = within-turn RTG
        metrics["atod/opd_gamma"] = float(opd_gamma)
        metrics["atod/opd_length_norm"] = 1.0 if opd_length_normalization else 0.0
    else:
        # Original token-level behavior: raw_kl_t = (log p_tea_t - log p_stu_t).
        raw_kl = delta_logp * response_mask
        metrics["atod/opd_signal"] = 0.0  # 0 = token-level (legacy)

    opd_term = eff_kl_coef * raw_kl

    # ---- 2. TIP weighting (on OPD term only) ----
    if enable_tip:
        if tip_granularity == "token":
            tip_weights, tip_metrics = compute_token_tip_weights(
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                response_mask=response_mask,
                tip_min_divergence=tip_min_divergence,
                tip_smoothing=tip_smoothing,
                entropy_clip_quantile=tip_entropy_clip_quantile,
            )
        elif tip_granularity == "turn":
            tip_weights, tip_metrics = compute_tip_weights(
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                response_mask=response_mask,
                traj_uids=traj_uids,
                turn_steps=turn_steps,
                tip_rho=tip_rho,
                tip_min_turns=tip_min_turns,
                tip_min_divergence=tip_min_divergence,
                tip_smoothing=tip_smoothing,
                tip_mean_preserve=tip_mean_preserve,
                tip_w_min=tip_w_min,
                tip_w_max=tip_w_max,
                tip_eps=tip_eps,
                tip_score_mode=tip_score_mode,
            )
        else:
            raise ValueError(
                f"Unknown tip_granularity={tip_granularity!r}; expected 'turn' or 'token'."
            )
        tip_weights = tip_weights.to(opd_term.device)
        opd_term = opd_term * tip_weights
        metrics["atod/tip_granularity"] = 1.0 if tip_granularity == "token" else 0.0
        metrics.update(tip_metrics)

    # ---- 3. Soft curriculum (on OPD term only, NOT on RL term) ----
    if enable_soft_curriculum:
        curriculum_weights, curriculum_metrics = compute_soft_curriculum_weights(
            response_mask=response_mask,
            global_step=global_step,
            turn_steps=turn_steps,
            checkpoint_steps=curriculum_checkpoint_steps,
            softness=curriculum_softness,
            bias=curriculum_bias,
            min_weight=curriculum_min_weight,
            max_weight=curriculum_max_weight,
            warmup_steps=curriculum_warmup_steps,
        )
        curriculum_weights = curriculum_weights.to(opd_term.device)
        opd_term = opd_term * curriculum_weights  # soft curriculum only on OPD
        metrics.update(curriculum_metrics)

    # ---- 4. GRPO RL term (unmodulated by curriculum) ----
    if grpo_advantages.dim() == 1:
        rl_term = eff_rl_coef * grpo_advantages.unsqueeze(1) * response_mask
    else:
        rl_term = eff_rl_coef * grpo_advantages * response_mask

    # ---- 5. Combine ----
    A_total = opd_term + rl_term

    # ---- Metrics ----
    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        metrics["atod/opd_mean"] = (opd_term.abs().sum() / mask_sum).item()
        metrics["atod/rl_mean"] = (rl_term.abs().sum() / mask_sum).item()
        metrics["atod/raw_kl_mean"] = (raw_kl.abs().sum() / mask_sum).item()
        # Effective (possibly annealed) coefficients
        metrics["atod/kl_coef"] = eff_kl_coef
        metrics["atod/rl_coef"] = eff_rl_coef
        # Base (config) coefficients and anneal status
        metrics["atod/kl_coef_base"] = kl_coef
        metrics["atod/rl_coef_base"] = rl_coef
        metrics["atod/anneal_enabled"] = 1.0 if enable_coef_anneal else 0.0
        metrics["atod/anneal_progress"] = anneal_progress
        opd_abs = opd_term.abs().sum().item()
        rl_abs = rl_term.abs().sum().item()
        metrics["atod/opd_rl_ratio"] = opd_abs / max(rl_abs, 1e-8)

        # ---- Per-component signed statistics (for analyzing GRPO vs OPD scale) ----
        # opd_term corresponds to the OPD/distillation signal (post TIP & curriculum).
        # rl_term corresponds to the GRPO/RL signal (post rl_coef).
        metrics.update(_masked_signed_stats(opd_term, response_mask, "atod/opd"))
        metrics.update(_masked_signed_stats(rl_term, response_mask, "atod/rl"))

    return A_total, metrics
