import pytest
import torch
import torch.nn.functional as F
from torch import nn

from modules import module_clip

LayerNorm = module_clip.LayerNorm


def _legacy_fp32(layer, value):
    return F.layer_norm(
        value.float(),
        layer.normalized_shape,
        layer.weight.float() if layer.weight is not None else None,
        layer.bias.float() if layer.bias is not None else None,
        layer.eps,
    ).to(value.dtype)


def test_layer_norm_defaults_to_fp16_and_accepts_fp32_fallback():
    layer = LayerNorm(8)
    assert layer.precision == "fp16"
    layer.set_precision("fp32")
    assert layer.precision == "fp32"


def test_layer_norm_rejects_unknown_precision():
    with pytest.raises(ValueError, match="layer norm precision"):
        LayerNorm(8, precision="tf32")


def test_set_layer_norm_precision_only_updates_custom_layers():
    module = nn.Sequential(LayerNorm(8), nn.LayerNorm(8), LayerNorm(8))
    assert module_clip.set_layer_norm_precision(module, "fp32") == 2
    assert module[0].precision == "fp32"
    assert module[2].precision == "fp32"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cpu_inputs_use_legacy_fp32_computation(dtype):
    layer = LayerNorm(8, precision="fp16")
    value = torch.randn(4, 8, dtype=dtype, requires_grad=True)
    torch.testing.assert_close(layer(value), _legacy_fp32(layer, value), rtol=0, atol=0)


def test_fp32_mode_matches_legacy_formula():
    layer = LayerNorm(8, precision="fp32")
    value = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    torch.testing.assert_close(layer(value), _legacy_fp32(layer, value))


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@CUDA_REQUIRED
@pytest.mark.parametrize("scale,offset", [(1.0, 0.0), (1e-4, 0.0), (1e4, 0.0), (1.0, 1e4)])
def test_cuda_fp16_is_finite_and_close_to_fp32(scale, offset):
    fp16_layer = LayerNorm(768, precision="fp16").cuda()
    fp32_layer = LayerNorm(768, precision="fp32").cuda()
    with torch.no_grad():
        fp16_layer.weight.uniform_(0.5, 1.5)
        fp16_layer.bias.uniform_(-0.2, 0.2)
    fp32_layer.load_state_dict(fp16_layer.state_dict())
    value_fp16 = (
        torch.randn(32, 16, 768, device="cuda", dtype=torch.float16) * scale + offset
    ).requires_grad_(True)
    value_fp32 = value_fp16.detach().clone().requires_grad_(True)
    actual = fp16_layer(value_fp16)
    expected = fp32_layer(value_fp32)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1e-2)
    probe = torch.randn_like(actual, dtype=torch.float32)
    (actual.float() * probe).sum().backward()
    (expected.float() * probe).sum().backward()
    assert torch.isfinite(value_fp16.grad).all()
    cosine = F.cosine_similarity(
        value_fp16.grad.float().flatten()[None],
        value_fp32.grad.float().flatten()[None],
    ).item()
    assert cosine >= 0.9999


@CUDA_REQUIRED
def test_cuda_fp16_does_not_save_full_fp32_input_copy():
    layer = LayerNorm(768, precision="fp16").cuda()
    value = torch.randn(64, 16, 768, device="cuda", dtype=torch.float16, requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: saved.append((tensor.dtype, tensor.numel())) or tensor,
        lambda tensor: tensor,
    ):
        layer(value).float().sum().backward()
    assert (torch.float32, value.numel()) not in saved
    assert layer.weight.dtype == torch.float32
    assert layer.weight.grad is not None
    assert layer.weight.grad.dtype == torch.float32
