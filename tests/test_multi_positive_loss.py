import json
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from modules.until_module import (
    MultiPositiveCrossEn,
    allgather_no_grad,
    allgather_with_grad,
)

_DDP_WORLD_SIZE = 2
_DDP_LINEAR_WEIGHT = torch.tensor([[0.4, -0.2], [0.3, 0.7]])
_DDP_LOCAL_INPUTS = (
    torch.tensor([[1.0, 2.0], [-1.0, 0.5]]),
    torch.tensor([[3.0, -2.0], [0.25, 4.0]]),
)


class _GatherLinear(nn.Module):
    def __init__(self, rank, world_size):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(_DDP_LINEAR_WEIGHT)

    def forward(self, local_input):
        local_features = self.linear(local_input)
        global_features = allgather_with_grad(
            local_features,
            SimpleNamespace(world_size=self.world_size, rank=self.rank),
        )
        return global_features.square().mean()


def _ddp_allgather_worker(rank, world_size, init_method, output_dir):
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=15),
    )
    try:
        model = DistributedDataParallel(_GatherLinear(rank, world_size))
        loss = model(_DDP_LOCAL_INPUTS[rank])
        loss.backward()
        (Path(output_dir) / f"rank-{rank}-grad.json").write_text(
            json.dumps(model.module.linear.weight.grad.tolist()),
            encoding="utf-8",
        )
    finally:
        torch.distributed.destroy_process_group()


def _single_process_global_gradient():
    model = _GatherLinear(rank=0, world_size=1)
    global_input = torch.cat(_DDP_LOCAL_INPUTS, dim=0)
    loss = model(global_input)
    loss.backward()
    return model.linear.weight.grad


def test_unique_groups_equal_diagonal_cross_entropy():
    logits = torch.tensor(
        [[4.0, 1.0, -1.0], [0.5, 3.0, 0.0], [-2.0, 1.0, 2.5]],
        requires_grad=True,
    )
    groups = torch.tensor([10, 11, 12])

    actual = MultiPositiveCrossEn()(logits, groups)
    expected = F.cross_entropy(logits, torch.arange(3))

    torch.testing.assert_close(actual, expected)


def test_same_video_columns_are_all_positives_for_rectangular_logits():
    logits = torch.tensor([[2.0, 1.0, 0.0], [1.5, 2.5, -1.0]])
    query_groups = torch.tensor([7, 7])
    candidate_groups = torch.tensor([7, 7, 9])

    actual = MultiPositiveCrossEn()(logits, query_groups, candidate_groups)
    expected = -(
        torch.logsumexp(logits[:, :2], dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()

    torch.testing.assert_close(actual, expected)


def test_duplicate_groups_are_positives_in_both_retrieval_directions():
    logits = torch.tensor(
        [[4.0, 1.0, 0.0], [3.0, 2.0, -1.0], [0.0, 1.0, 5.0]]
    )
    groups = torch.tensor([7, 7, 9])
    mask = groups[:, None].eq(groups[None, :])
    loss_fct = MultiPositiveCrossEn()

    actual = (
        loss_fct(logits, groups, groups)
        + loss_fct(logits.T, groups, groups)
    ) / 2
    expected_t2v = -(
        torch.logsumexp(logits.masked_fill(~mask, float("-inf")), dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()
    expected_v2t = -(
        torch.logsumexp(logits.T.masked_fill(~mask.T, float("-inf")), dim=1)
        - torch.logsumexp(logits.T, dim=1)
    ).mean()

    torch.testing.assert_close(actual, (expected_t2v + expected_v2t) / 2)


def test_missing_positive_fails_with_query_index_and_group_id():
    logits = torch.zeros(2, 2)

    with pytest.raises(ValueError, match=r"query indices=\[1\].*group IDs=\[2\]"):
        MultiPositiveCrossEn()(
            logits,
            torch.tensor([1, 2]),
            torch.tensor([1, 3]),
        )


def test_extreme_logits_have_finite_loss_and_gradients():
    logits = torch.tensor(
        [[1000.0, -1000.0], [-1000.0, 1000.0]], requires_grad=True
    )

    loss = MultiPositiveCrossEn()(logits, torch.tensor([1, 2]))
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    ("logits", "query_groups", "candidate_groups", "message"),
    [
        (torch.zeros(2), torch.tensor([1, 2]), None, "logits must be 2D"),
        (
            torch.zeros(2, 3),
            torch.tensor([1, 2]),
            torch.tensor([1, 2]),
            "logit/group shape mismatch",
        ),
        (
            torch.zeros(2, 2),
            torch.tensor([1.0, 2.0]),
            None,
            "query_group_ids must use an integer dtype",
        ),
        (
            torch.zeros(2, 2),
            torch.tensor([1, 2]),
            torch.tensor([1.0, 2.0]),
            "candidate_group_ids must use an integer dtype",
        ),
    ],
)
def test_invalid_shapes_and_group_dtypes_fail(
    logits, query_groups, candidate_groups, message
):
    with pytest.raises(ValueError, match=message):
        MultiPositiveCrossEn()(logits, query_groups, candidate_groups)


def test_world_size_one_gathers_work_without_distributed_attributes():
    no_grad_tensor = torch.tensor([4, 5])
    grad_tensor = torch.tensor([4.0, 5.0], requires_grad=True)

    gathered_labels = allgather_no_grad(no_grad_tensor, SimpleNamespace())
    gathered_features = allgather_with_grad(grad_tensor, SimpleNamespace())
    gathered_features.sum().backward()

    assert torch.equal(gathered_labels, no_grad_tensor)
    assert gathered_labels.requires_grad is False
    assert torch.equal(grad_tensor.grad, torch.ones_like(grad_tensor))


def test_ddp_allgather_gradient_matches_single_process_global_batch(tmp_path):
    init_file = tmp_path / "gloo-rendezvous"
    process_context = mp.spawn(
        _ddp_allgather_worker,
        args=(
            _DDP_WORLD_SIZE,
            f"file://{init_file}",
            str(tmp_path),
        ),
        nprocs=_DDP_WORLD_SIZE,
        join=False,
    )
    try:
        deadline = time.monotonic() + 30
        while not process_context.join(
            timeout=max(0, deadline - time.monotonic())
        ):
            if time.monotonic() >= deadline:
                pytest.fail(
                    "two-process gloo/DDP regression timed out after 30 seconds"
                )
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
        for process in process_context.processes:
            process.join(timeout=5)

    expected = _single_process_global_gradient()
    for rank in range(_DDP_WORLD_SIZE):
        actual = torch.tensor(
            json.loads(
                (tmp_path / f"rank-{rank}-grad.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("gather", [allgather_no_grad, allgather_with_grad])
def test_multi_rank_gather_requires_initialized_process_group(gather, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="world_size=2.*not initialized"):
        gather(torch.tensor([1]), SimpleNamespace(world_size=2, rank=0))


def test_no_grad_gather_preserves_integer_labels_without_autograd(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    def fake_all_gather(output, tensor):
        output[0].copy_(tensor)
        output[1].copy_(tensor + 10)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    labels = torch.tensor([4, 5], dtype=torch.long)

    gathered = allgather_no_grad(
        labels, SimpleNamespace(world_size=2, rank=0)
    )

    assert gathered.dtype == torch.long
    assert gathered.requires_grad is False
    assert gathered.tolist() == [4, 5, 14, 15]


def test_cross_rank_feature_and_group_gathers_keep_rank_order_aligned(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    def fake_all_gather(output, tensor):
        output[0].copy_(tensor)
        output[1].copy_(tensor + 100)

    def fake_all_reduce(tensor, op):
        assert tensor.is_contiguous()
        assert op == torch.distributed.ReduceOp.SUM
        tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    task_config = SimpleNamespace(world_size=2, rank=0)
    features = torch.tensor([[1.0], [2.0]], requires_grad=True)
    group_ids = torch.tensor([10, 20])

    global_features = allgather_with_grad(features, task_config)
    global_group_ids = allgather_no_grad(group_ids, task_config)
    global_features.sum().backward()

    assert global_features[:, 0].tolist() == [1.0, 2.0, 101.0, 102.0]
    assert global_group_ids.tolist() == [10, 20, 110, 120]
    assert torch.equal(features.grad, torch.full_like(features, 2.0))


def test_grad_gather_backward_fails_if_process_group_is_no_longer_initialized(
    monkeypatch,
):
    distributed_state = {"initialized": True}
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "is_initialized",
        lambda: distributed_state["initialized"],
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda output, tensor: [item.copy_(tensor) for item in output],
    )
    features = torch.tensor([[1.0]], requires_grad=True)
    gathered = allgather_with_grad(
        features, SimpleNamespace(world_size=2, rank=0)
    )

    distributed_state["initialized"] = False
    with pytest.raises(RuntimeError, match="world_size=2.*not initialized"):
        gathered.sum().backward()


def test_grad_gather_backward_revalidates_task_config(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda output, tensor: [item.copy_(tensor) for item in output],
    )
    task_config = SimpleNamespace(world_size=2, rank=0)
    features = torch.tensor([[1.0]], requires_grad=True)
    gathered = allgather_with_grad(features, task_config)

    task_config.world_size = 3
    with pytest.raises(RuntimeError, match="configured world_size=3.*actual world_size=2"):
        gathered.sum().backward()
