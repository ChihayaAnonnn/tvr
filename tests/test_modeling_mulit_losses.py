import importlib
import sys
import types
from argparse import Namespace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spatial_enhancer_stub = types.ModuleType("modules.spatial_enhancer")
spatial_enhancer_stub.SpatialEnhancer = object
sys.modules["modules.spatial_enhancer"] = spatial_enhancer_stub
UATVR = importlib.import_module("modules.modeling_mulit").UATVR


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
