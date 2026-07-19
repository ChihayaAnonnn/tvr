import logging
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

import main_task_retrieval
import modules.modeling as modeling
from main_task_retrieval import _unpack_train_batch, train_epoch
from modules.modeling import UATVR
from modules.until_module import CrossEn, MultiPositiveCrossEn


@pytest.mark.parametrize(
    ("batch_size", "group_index"),
    ((5, None), (6, 5), (8, None), (9, 8)),
)
def test_unpack_train_batch_preserves_optional_group_ids(batch_size, group_index):
    batch = tuple(torch.tensor([index]) for index in range(batch_size))

    unpacked = _unpack_train_batch(batch)

    assert len(unpacked) == 6
    if group_index is None:
        assert unpacked[-1] is None
    else:
        torch.testing.assert_close(unpacked[-1], batch[group_index])


class _RecordingTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.clip = nn.Module()
        self.clip.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, *inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return self.weight.square()


@pytest.mark.parametrize(
    ("warmup_epochs", "expected_scales"),
    ((2.0, (0.5, 0.75)), (None, (1.0, 1.0))),
)
def test_train_epoch_forwards_group_ids_and_per_step_warmup_scales(
    monkeypatch, warmup_epochs, expected_scales
):
    monkeypatch.setattr(
        main_task_retrieval, "logger", logging.getLogger(__name__), raising=False
    )
    model = _RecordingTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    group_ids = torch.tensor([7, 7])
    batch = tuple(torch.zeros(2, dtype=torch.long) for _ in range(5)) + (
        group_ids,
    )
    args = SimpleNamespace(
        n_display=100,
        gradient_accumulation_steps=1,
        epochs=3,
    )
    if warmup_epochs is not None:
        args.rspr_warmup_epochs = warmup_epochs

    train_epoch(
        1,
        args,
        model,
        [batch, batch],
        torch.device("cpu"),
        1,
        optimizer,
        None,
        0,
    )

    assert len(model.calls) == 2
    for (_, kwargs), expected_scale in zip(model.calls, expected_scales):
        torch.testing.assert_close(kwargs["group_ids"], group_ids)
        assert kwargs["rspr_rank_scale"] == expected_scale
        assert kwargs["rspr_anchor_scale"] == expected_scale


def test_gather_group_ids_preserves_rank_major_order_dtype_and_device(monkeypatch):
    task_config = SimpleNamespace(world_size=2)
    calls = []

    def fake_allgather(local_ids, received_config):
        calls.append((local_ids, received_config))
        return torch.cat((local_ids, torch.tensor([21, 22], device=local_ids.device)))

    monkeypatch.setattr(modeling, "allgather", fake_allgather)
    local_ids = torch.tensor([[11], [12]], dtype=torch.int32)

    gathered = modeling.gather_group_ids(local_ids, task_config)

    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], torch.tensor([11, 12]))
    assert calls[0][1] is task_config
    assert gathered.dtype == torch.long
    assert gathered.device == local_ids.device
    torch.testing.assert_close(gathered, torch.tensor([11, 12, 21, 22]))


@pytest.mark.parametrize("invalid_ids", (None, torch.tensor([1.0, 2.0])))
def test_gather_group_ids_requires_explicit_integer_ids(invalid_ids):
    with pytest.raises(ValueError, match="explicit group_ids|integer dtype"):
        modeling.gather_group_ids(invalid_ids, SimpleNamespace())


def _uatvr_forward_harness(logits, mode):
    model = UATVR.__new__(UATVR)
    nn.Module.__init__(model)
    model.task_config = SimpleNamespace()
    if mode is not None:
        model.task_config.rspr_mode = mode
    model.loose_type = True
    model.loss_fct = CrossEn()
    model.multi_positive_loss = MultiPositiveCrossEn()
    similarity_calls = []

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped):
        del self, token_type_ids, attention_mask, shaped
        batch_size = input_ids.size(0)
        return torch.ones(batch_size, 1, 2), torch.ones(batch_size, 2, 2)

    def get_visual_output(self, video, video_mask, shaped, video_frame):
        del self, video_mask, shaped, video_frame
        return torch.ones(video.size(0), 2, 2)

    def get_similarity_logits(self, *args, **kwargs):
        del self, args, kwargs
        similarity_calls.append(True)
        zero = logits.new_zeros(())
        return logits, zero, zero

    model.get_sequence_output = MethodType(get_sequence_output, model)
    model.get_visual_output = MethodType(get_visual_output, model)
    model.get_similarity_logits = MethodType(get_similarity_logits, model)
    model.train()
    return model, similarity_calls


def _forward_inputs(batch_size=3):
    return (
        torch.zeros(batch_size, 1, 2, dtype=torch.long),
        torch.zeros(batch_size, 1, 2, dtype=torch.long),
        torch.ones(batch_size, 1, 2, dtype=torch.long),
        torch.zeros(batch_size, 1, 1, 1, 1, 1, 1),
        torch.ones(batch_size, 1, 1, dtype=torch.long),
    )


@pytest.mark.parametrize("mode", ("off", "mean", "stochastic"))
def test_explicit_rspr_modes_reject_missing_ids_before_similarity(mode):
    model, similarity_calls = _uatvr_forward_harness(torch.eye(3), mode)

    with pytest.raises(ValueError, match="explicit group_ids"):
        model(*_forward_inputs(), group_ids=None)

    assert similarity_calls == []


def test_missing_mode_defaults_to_legacy_diagonal_cross_entropy(monkeypatch):
    logits = torch.tensor([[3.0, 0.0], [0.0, 2.0]], requires_grad=True)
    model, similarity_calls = _uatvr_forward_harness(logits, None)
    gather_calls = []
    monkeypatch.setattr(
        modeling, "allgather", lambda *args: gather_calls.append(args)
    )

    loss = model(*_forward_inputs(batch_size=2))

    expected = 0.5 * (CrossEn()(logits) + CrossEn()(logits.T))
    torch.testing.assert_close(loss, expected)
    assert similarity_calls == [True]
    assert gather_calls == []


def test_evaluation_does_not_require_or_gather_group_ids(monkeypatch):
    model, similarity_calls = _uatvr_forward_harness(torch.eye(2), "stochastic")
    model.eval()
    gather_calls = []
    monkeypatch.setattr(
        modeling, "allgather", lambda *args: gather_calls.append(args)
    )

    result = model(*_forward_inputs(batch_size=2))

    assert result is None
    assert gather_calls == []
    assert similarity_calls == []


def test_forward_gathers_ids_once_and_uses_same_group_positives(monkeypatch):
    logits = torch.tensor(
        [[0.0, 4.0, -1.0], [3.0, 0.0, -1.0], [-2.0, -2.0, 2.0]],
        requires_grad=True,
    )
    group_ids = torch.tensor([1, 1, 2])
    model, similarity_calls = _uatvr_forward_harness(logits, "legacy")
    gather_calls = []

    def identity_allgather(ids, task_config):
        gather_calls.append((ids, task_config))
        return ids

    monkeypatch.setattr(modeling, "allgather", identity_allgather)

    loss = model(*_forward_inputs(), group_ids=group_ids)

    expected, positive_mask = MultiPositiveCrossEn().bidirectional(
        logits, group_ids
    )
    torch.testing.assert_close(loss, expected)
    assert positive_mask[:2, :2].all()
    assert len(gather_calls) == 1
    assert similarity_calls == [True]


def _masked_cross_entropy(logits, positive_mask):
    positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
    return -(
        torch.logsumexp(positive_logits, dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()


def test_bidirectional_multi_positive_supports_rectangular_ids_and_gradients():
    logits = torch.tensor(
        [[0.2, -0.3], [1.1, 0.4], [-0.7, 1.3]], requires_grad=True
    )
    text_group_ids = torch.tensor([1, 1, 2])
    video_group_ids = torch.tensor([1, 2])

    loss, positive_mask = MultiPositiveCrossEn().bidirectional(
        logits, text_group_ids, video_group_ids
    )

    expected_mask = torch.tensor(
        [[True, False], [True, False], [False, True]]
    )
    expected_loss = 0.5 * (
        _masked_cross_entropy(logits, expected_mask)
        + _masked_cross_entropy(logits.T, expected_mask.T)
    )
    torch.testing.assert_close(positive_mask, expected_mask)
    torch.testing.assert_close(loss, expected_loss)

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_bidirectional_multi_positive_square_ids_mark_all_same_group_pairs():
    logits = torch.eye(3)

    _, positive_mask = MultiPositiveCrossEn().bidirectional(
        logits, torch.tensor([1, 1, 2])
    )

    torch.testing.assert_close(
        positive_mask,
        torch.tensor(
            [
                [True, True, False],
                [True, True, False],
                [False, False, True],
            ]
        ),
    )


@pytest.mark.parametrize(
    ("text_group_ids", "video_group_ids", "missing_direction"),
    (
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2]), "text"),
        (torch.tensor([1, 2]), torch.tensor([1, 2, 3]), "video"),
    ),
)
def test_bidirectional_multi_positive_validates_positives_in_both_directions(
    text_group_ids, video_group_ids, missing_direction
):
    logits = torch.zeros(text_group_ids.numel(), video_group_ids.numel())

    with pytest.raises(ValueError, match=missing_direction):
        MultiPositiveCrossEn().bidirectional(
            logits, text_group_ids, video_group_ids
        )
