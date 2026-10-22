"""CPU gradient regressions for actor microbatching and partial minibatches.

The production update_policy function body is loaded without its GPU memory
logging decorator. Only model forward, optimizer/device routing are CPU toys;
the real batching, policy loss and normalization code are executed unchanged.
"""

import ast
from datetime import timedelta
import json
import multiprocessing as mp
from pathlib import Path
import socket
import tempfile
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo
from verl.utils.loss_normalization import MiniBatchLossNormalizer
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches

ROOT = Path(__file__).resolve().parents[3]
MODES = ("token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm")


def production_update():
    path = ROOT / "verl/workers/actor/dp_actor.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DataParallelPPOActor")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "update_policy")
    method.decorator_list = []
    namespace = dict(torch=torch, DataProto=DataProto, compute_policy_loss=compute_policy_loss,
        compute_policy_loss_gspo=compute_policy_loss_gspo, agg_loss=agg_loss,
        MiniBatchLossNormalizer=MiniBatchLossNormalizer, append_to_dict=append_to_dict,
        rearrange_micro_batches=rearrange_micro_batches,
        get_torch_device=lambda: SimpleNamespace(current_device=lambda: torch.device("cpu")))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["update_policy"]


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self):
        return self.weight * 1.0


def run_update(lengths, advantages, mini_size, micro_size, mode="token-mean", dynamic=False,
               multi_turn=False, multimodal=False, policy_mode="vanilla", entropy_coeff=0.):
    model = ScalarModel()
    config = OmegaConf.create(dict(ppo_mini_batch_size=mini_size, ppo_micro_batch_size_per_gpu=micro_size,
        ppo_epochs=1, use_dynamic_bsz=dynamic, ppo_max_token_len_per_gpu=max(lengths)+2,
        use_kl_loss=False, clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.2, clip_ratio_c=3.,
        entropy_coeff=entropy_coeff, loss_agg_mode=mode, policy_loss={"loss_mode": policy_mode}))
    actor = SimpleNamespace(config=config, actor_module=model, ulysses_sequence_parallel_size=1,
                            actor_optimizer=torch.optim.SGD(model.parameters(), lr=0.))
    gradients = []
    def forward(self, micro_batch, temperature, calculate_entropy=False):
        lp = model().expand_as(micro_batch["responses"])
        return (lp + .5 if calculate_entropy else None), lp
    def step(self):
        gradients.append(float(model.weight.grad))
        norm = model.weight.grad.detach().abs().clone()
        self.actor_optimizer.step()
        return norm
    actor._forward_micro_batch = MethodType(forward, actor)
    actor._optimizer_step = MethodType(step, actor)
    width = max(lengths)
    mask = torch.tensor([[int(j < n) for j in range(width)] for n in lengths])
    attention = torch.ones_like(mask) if multi_turn else mask
    tensors = dict(responses=torch.ones(len(lengths), width, dtype=torch.long),
        input_ids=torch.ones(len(lengths), width+1, dtype=torch.long),
        attention_mask=torch.cat([torch.ones(len(lengths), 1, dtype=torch.long), attention], dim=1),
        position_ids=torch.arange(width+1).repeat(len(lengths), 1),
        old_log_probs=torch.zeros(len(lengths), width),
        advantages=torch.tensor(advantages).float()[:, None].expand(-1, width))
    if multi_turn:
        tensors["loss_mask"] = torch.cat([torch.zeros(len(lengths), 1), mask], dim=1)
    non_tensor = {"multi_modal_inputs": np.array([{} for _ in lengths], dtype=object)} if multimodal else {}
    data = DataProto.from_dict(tensors=tensors, non_tensors=non_tensor,
                               meta_info={"temperature": 1., "multi_turn": multi_turn})
    return gradients, production_update()(actor, data)


def distributed_worker(rank, endpoint, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=endpoint,
                            timeout=timedelta(seconds=30))
    try:
        lengths = [1, 1] if rank == 0 else [2, 4]
        advantages = [1., 2.] if rank == 0 else [3., 4.]
        mask = torch.tensor([[int(j < n) for j in range(4)] for n in lengths])
        normalizer = MiniBatchLossNormalizer(mask)
        model = DistributedDataParallel(ScalarModel())
        total = torch.zeros(())
        for i in range(2):
            lp = model().expand(1, 4)
            loss, *_ = compute_policy_loss(torch.zeros_like(lp), lp,
                torch.full_like(lp, advantages[i]), mask[i:i+1], cliprange=.2)
            contribution = normalizer(loss, mask[i:i+1], "token-mean")
            total += contribution.detach()
            contribution.backward()
        dist.all_reduce(total)
        total /= dist.get_world_size()
        (Path(folder)/f"rank{rank}.json").write_text(json.dumps({
            "gradient": float(model.module.weight.grad), "loss": float(total)}))
    finally:
        dist.destroy_process_group()


class NormalizerTests(unittest.TestCase):
    def test_matches_full_minibatch_loss_and_gradient_for_all_reducers(self):
        mask = torch.tensor([[1,0,0,0], [1,1,1,0], [1,1,1,1]])
        values = torch.arange(12).float().reshape(3,4)
        for mode in MODES:
            for sizes in ([1,1,1], [2,1], [3]):
                with self.subTest(mode=mode, sizes=sizes):
                    weight = torch.tensor(1., requires_grad=True)
                    expected = agg_loss(weight*values, mask, mode)
                    grad = torch.autograd.grad(expected, weight)[0]
                    norm = MiniBatchLossNormalizer(mask)
                    loss = sum(norm(agg_loss(weight*v, m, mode), m, mode)
                        for v,m in zip(values.split(sizes), mask.split(sizes)))
                    actual_grad = torch.autograd.grad(loss, weight)[0]
                    torch.testing.assert_close(loss, expected)
                    torch.testing.assert_close(actual_grad, grad)

    def test_zero_token_microbatch_has_zero_contribution(self):
        mask = torch.tensor([[0,0], [1,1]])
        weight = torch.tensor(1., requires_grad=True)
        norm = MiniBatchLossNormalizer(mask)
        first = norm(agg_loss(weight*torch.ones(1,2), mask[:1], "token-mean"), mask[:1], "token-mean")
        self.assertEqual(float(first), 0.)
        self.assertEqual(float(torch.autograd.grad(first, weight)[0]), 0.)


class ActorIntegrationTests(unittest.TestCase):
    def test_variable_lengths_are_invariant_to_microbatch_partition(self):
        for micro in (1, 2):
            grads, metrics = run_update([1,3], [1,3], 2, micro)
            self.assertAlmostEqual(grads[0], -2.5, places=6)
            self.assertAlmostEqual(float(np.mean(metrics["actor/pg_loss"])), -2.5, places=6)

    def test_partial_minibatch_uses_actual_size(self):
        for rows in (33, 40):
            for micro in (1, 3):
                with self.subTest(rows=rows, micro=micro):
                    grads, _ = run_update([1]*rows, [1]*rows, 16, micro)
                    self.assertEqual(len(grads), 3)
                    for gradient in grads:
                        self.assertAlmostEqual(gradient, -1., places=6)

    def test_multiturn_uses_loss_mask_for_normalization(self):
        grads, _ = run_update([1,3], [1,3], 2, 1, multi_turn=True)
        self.assertAlmostEqual(grads[0], -2.5, places=6)

    def test_dynamic_and_fixed_microbatching_match(self):
        reference, _ = run_update([1,3,4,2,1], [1,3,2,4,5], 5, 5)
        actual, _ = run_update([1,3,4,2,1], [1,3,2,4,5], 5, 1, dynamic=True)
        self.assertAlmostEqual(actual[0], reference[0], places=6)

    def test_multimodal_partial_batches_are_retained(self):
        grads, _ = run_update([1]*5, [1]*5, 3, 2, multimodal=True)
        self.assertEqual(len(grads), 2)
        for gradient in grads:
            self.assertAlmostEqual(gradient, -1., places=6)

    def test_sequence_mean_and_gspo_keep_their_reduction(self):
        for mode, policy in (("seq-mean-token-mean", "vanilla"), ("token-mean", "gspo")):
            grads, _ = run_update([1,3], [1,3], 2, 1, mode=mode, policy_mode=policy)
            self.assertAlmostEqual(grads[0], -2., places=6)

    def test_entropy_keeps_relative_weight(self):
        for micro in (1,2):
            grads, _ = run_update([1,3], [1,3], 2, micro, entropy_coeff=.1)
            self.assertAlmostEqual(grads[0], -2.6, places=6)


class DistributedTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires CPU Gloo")
    def test_two_ranks_with_unequal_token_counts_match_global_mean(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1",0))
            port = sock.getsockname()[1]
        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/"outputs") as folder:
            ctx = mp.get_context("spawn")
            workers = [ctx.Process(target=distributed_worker, args=(rank, f"tcp://127.0.0.1:{port}", folder)) for rank in range(2)]
            try:
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(timeout=45)
                    self.assertEqual(worker.exitcode, 0)
                for rank in range(2):
                    result = json.loads((Path(folder)/f"rank{rank}.json").read_text())
                    self.assertAlmostEqual(result["gradient"], -25/8, places=6)
                    self.assertAlmostEqual(result["loss"], -25/8, places=6)
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=10)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
