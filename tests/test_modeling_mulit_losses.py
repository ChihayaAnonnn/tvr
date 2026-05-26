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


def test_lightweight_tas_entropy_is_higher_for_ambiguous_neighbors():
    text_pooled = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    tas = UATVR._compute_lightweight_tas(text_pooled, top_k=3, temperature=0.05)

    assert tas[0] > tas[2]
    assert torch.all((tas >= 0.0) & (tas <= 1.0))


def test_dynamic_vib_loss_relaxes_text_kl_for_high_tas():
    sampled_video = torch.zeros(2, 1, 2)
    video_sigma = torch.zeros(2, 2)
    sampled_text = torch.zeros(2, 1, 2)
    text_sigma = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    high_then_low_tas = torch.tensor([1.0, 0.0])
    low_then_high_tas = torch.tensor([0.0, 1.0])

    high_first_loss = UATVR._dynamic_vib_loss(
        sampled_video,
        video_sigma,
        sampled_text,
        text_sigma,
        query_weight=high_then_low_tas,
        tas_kl_scale=0.5,
    )
    low_first_loss = UATVR._dynamic_vib_loss(
        sampled_video,
        video_sigma,
        sampled_text,
        text_sigma,
        query_weight=low_then_high_tas,
        tas_kl_scale=0.5,
    )

    assert high_first_loss == low_first_loss

    uneven_text_sigma = torch.tensor([[1.0, 1.0], [0.1, 0.1]])
    high_uncertain_loss = UATVR._dynamic_vib_loss(
        sampled_video,
        video_sigma,
        sampled_text,
        uneven_text_sigma,
        query_weight=high_then_low_tas,
        tas_kl_scale=0.5,
    )
    low_uncertain_loss = UATVR._dynamic_vib_loss(
        sampled_video,
        video_sigma,
        sampled_text,
        uneven_text_sigma,
        query_weight=low_then_high_tas,
        tas_kl_scale=0.5,
    )

    assert high_uncertain_loss < low_uncertain_loss
