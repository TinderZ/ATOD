"""Preserve a minibatch loss reduction when accumulating microbatch gradients."""

import torch
import torch.distributed as dist


class MiniBatchLossNormalizer:
    """Normalize against actual minibatch counts across gradient-averaging ranks.

    FSDP/DDP average parameter gradients, so each rank's contribution must be
    multiplied by world_size / global_denominator. Ulysses replicates minibatch
    masks within each SP group; those copies appear in both the count and world
    size and cancel. Construct once per optimizer step, before microbatching.
    """

    def __init__(self, response_mask):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.response_length = response_mask.size(-1)
        counts = torch.stack([
            response_mask.detach().sum(dtype=torch.float32),
            response_mask.new_tensor(response_mask.size(0), dtype=torch.float32),
        ])
        if self.world_size > 1:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        self.tokens, self.sequences = counts.unbind()

    def __call__(self, loss, response_mask, loss_agg_mode):
        """Scale an already-reduced microbatch loss, retaining its autograd graph."""
        if loss_agg_mode == "token-mean":
            # Undo masked_mean's local denominator (including its epsilon).
            tokens = response_mask.detach().sum(dtype=torch.float32)
            scale = self.world_size * (tokens + 1e-8) / (self.tokens + 1e-8)
        elif loss_agg_mode in ("seq-mean-token-sum", "seq-mean-token-mean"):
            scale = self.world_size * response_mask.size(0) / self.sequences.clamp(min=1)
        elif loss_agg_mode == "seq-mean-token-sum-norm":
            # agg_loss divides the token SUM only by the response width.
            scale = self.world_size * response_mask.size(-1) / self.response_length
        else:
            raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
        return loss * scale
