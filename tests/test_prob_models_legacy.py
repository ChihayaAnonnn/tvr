import torch

from prob_models.pie_model import PIENet
from prob_models.uncertainty_module import UncertaintyModuleText


def test_pienet_padding_mask_removes_padding_from_attention():
    torch.manual_seed(0)
    pie = PIENet(1, 4, 4, 2)
    out = torch.zeros(1, 4)
    x = torch.randn(1, 4, 4)
    pad_mask = torch.tensor([[False, False, True, True]])

    _out, attn, _residual = pie(out, x, pad_mask=pad_mask)

    assert torch.allclose(attn[0, 2:, 0], torch.zeros(2), atol=1e-6)
    assert torch.isclose(attn[0, :2, 0].sum(), torch.tensor(1.0), atol=1e-6)


def test_uncertainty_module_text_uses_true_padding_mask_for_lengths():
    torch.manual_seed(0)
    module = UncertaintyModuleText(4, 4, 2)
    out = torch.randn(2, 4)
    x = torch.randn(2, 5, 4)
    pad_mask = torch.tensor(
        [
            [False, False, False, True, True],
            [False, True, True, True, True],
        ]
    )

    result = module(out, x, pad_mask=pad_mask)

    assert result["logsigma"].shape == (2, 4)
    assert torch.allclose(result["attention"][0, 3:, 0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(result["attention"][1, 1:, 0], torch.zeros(4), atol=1e-6)
