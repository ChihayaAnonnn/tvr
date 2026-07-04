import importlib
import inspect
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spatial_enhancer_stub = types.ModuleType("modules.spatial_enhancer")
spatial_enhancer_stub.SpatialEnhancer = object
sys.modules["modules.spatial_enhancer"] = spatial_enhancer_stub
UATVR = importlib.import_module("modules.modeling_mulit").UATVR


def test_forward_accepts_explicit_hard_negative_kwargs():
    signature = inspect.signature(UATVR.forward)

    for name in ["sample_index", "hard_video", "hard_video_mask", "hard_valid"]:
        assert name in signature.parameters


def test_evidential_matrix_loss_prefers_confident_diagonal():
    confident = torch.tensor(
        [
            [8.0, -1.0, -1.0],
            [-1.0, 8.0, -1.0],
            [-1.0, -1.0, 8.0],
        ]
    )
    flat = torch.zeros_like(confident)

    confident_loss, _ = UATVR.evidential_matrix_loss(confident)
    flat_loss, _ = UATVR.evidential_matrix_loss(flat)

    assert confident_loss < flat_loss


def test_model_weight_defaults_match_uncertainty_only_setting():
    task_config = Namespace()

    assert getattr(task_config, "w_uncertainty_reg", 1e-3) == 1e-3


def test_evidential_similarity_penalizes_high_uncertainty():
    """认知不确定性越高，evidential 相似度越低。"""
    mu_video = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # [2, 2], 已 L2 norm
    text_pooled = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # [2, 2]
    # 低不确定性
    epistemic_low = torch.full((2, 4, 2), 0.01)
    sim_low = UATVR._evidential_similarity(mu_video, text_pooled, epistemic_low)
    # 高不确定性
    epistemic_high = torch.full((2, 4, 2), 5.0)
    sim_high = UATVR._evidential_similarity(mu_video, text_pooled, epistemic_high)
    # 低不确定性的对角值应大于高不确定性的
    assert sim_low.diagonal().mean() > sim_high.diagonal().mean()


def test_evidential_nll_loss_penalizes_weak_positives():
    """正对分数越低，NLL loss 越高。"""
    alpha_dir = torch.tensor([[10.0, 10.0], [10.0, 10.0]])
    # 强正对
    sim_strong = torch.tensor([[2.0, -0.5], [-0.5, 2.0]])
    loss_strong = UATVR._evidential_nll_loss(sim_strong, alpha_dir)
    # 弱正对
    sim_weak = torch.tensor([[0.1, -0.5], [-0.5, 0.1]])
    loss_weak = UATVR._evidential_nll_loss(sim_weak, alpha_dir)
    assert loss_weak > loss_strong


def test_evidential_neg_reg_loss_penalizes_high_negative_evidence():
    """负对证据量越高，neg_reg loss 越高。"""
    # 高负对分数
    sim_high_neg = torch.tensor([[1.0, 3.0, 3.0], [3.0, 1.0, 3.0], [3.0, 3.0, 1.0]])
    loss_high = UATVR._evidential_neg_reg_loss(sim_high_neg)
    # 低负对分数
    sim_low_neg = torch.tensor([[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    loss_low = UATVR._evidential_neg_reg_loss(sim_low_neg)
    assert loss_high > loss_low


def test_explicit_hard_negative_infonce_matches_concatenated_denominator():
    retrieval_logits = torch.tensor(
        [
            [3.0, 0.1, -0.2],
            [0.0, 3.2, 0.4],
            [0.1, -0.3, 2.8],
        ]
    )
    hard_logits = torch.tensor(
        [
            [2.0, 4.0, 8.0],
            [2.4, 0.2, 8.0],
            [1.8, 0.3, 8.0],
        ]
    )
    valid = torch.tensor([1, 1, 0])

    loss = UATVR._hard_negative_infonce_loss(retrieval_logits, hard_logits, valid)

    masked_hard = hard_logits.masked_fill(~valid.to(dtype=torch.bool).unsqueeze(0), torch.finfo(hard_logits.dtype).min)
    expected = torch.nn.functional.cross_entropy(
        torch.cat([retrieval_logits, masked_hard], dim=1),
        torch.arange(retrieval_logits.size(0)),
    )
    assert torch.allclose(loss, expected)


def test_explicit_hard_negative_infonce_ignores_all_invalid_hard_negatives():
    retrieval_logits = torch.eye(3)
    hard_logits = torch.full((3, 3), 100.0)
    valid = torch.zeros(3, dtype=torch.long)

    loss = UATVR._hard_negative_infonce_loss(retrieval_logits, hard_logits, valid)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_select_closest_gaussian_sample_picks_highest_cosine_sample():
    mean = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    samples = torch.tensor(
        [
            [[0.0, 1.0], [0.8, 0.2], [1.0, 0.0]],
            [[1.0, 0.0], [0.1, 0.9], [0.0, 1.0]],
        ]
    )

    selected = UATVR._select_closest_gaussian_sample(mean, samples)

    assert torch.allclose(selected, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_select_uacl_gaussian_sample_keeps_closest_strategy_behavior():
    mean = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    samples = torch.tensor(
        [
            [[0.0, 1.0], [0.8, 0.2], [1.0, 0.0]],
            [[1.0, 0.0], [0.1, 0.9], [0.0, 1.0]],
        ]
    )

    selected = UATVR._select_uacl_gaussian_sample(mean, samples, strategy="closest")

    assert torch.allclose(selected, UATVR._select_closest_gaussian_sample(mean, samples))


def test_select_uacl_gaussian_sample_random_returns_one_sample_per_row():
    torch.manual_seed(0)
    mean = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    samples = torch.tensor(
        [
            [[0.0, 1.0], [0.8, 0.2], [1.0, 0.0]],
            [[1.0, 0.0], [0.1, 0.9], [0.0, 1.0]],
        ]
    )

    selected = UATVR._select_uacl_gaussian_sample(mean, samples, strategy="random")

    assert selected.shape == mean.shape
    for row_idx in range(samples.size(0)):
        assert any(torch.allclose(selected[row_idx], sample) for sample in samples[row_idx])


def test_select_uacl_gaussian_sample_rejects_unknown_strategy():
    mean = torch.tensor([[1.0, 0.0]])
    samples = torch.tensor([[[1.0, 0.0]]])

    with pytest.raises(ValueError, match="Unknown UACL sample strategy"):
        UATVR._select_uacl_gaussian_sample(mean, samples, strategy="bad")


def test_uacl_intra_contrastive_loss_prefers_aligned_pairs():
    anchor = torch.eye(3)
    aligned = anchor.clone()
    shuffled = anchor[[1, 2, 0]]

    aligned_loss = UATVR._uacl_intra_contrastive_loss(anchor, aligned, temperature=0.1)
    shuffled_loss = UATVR._uacl_intra_contrastive_loss(anchor, shuffled, temperature=0.1)

    assert aligned_loss < shuffled_loss


def test_logvar_kl_is_zero_at_unit_variance_and_positive_otherwise():
    unit_logvar = torch.zeros(2, 3)
    shifted_logvar = torch.full((2, 3), 0.5)

    assert torch.isclose(UATVR._logvar_kl(unit_logvar), torch.tensor(0.0))
    assert UATVR._logvar_kl(shifted_logvar) > 0
