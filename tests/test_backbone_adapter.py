import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

import modules.backbone_adapter as backbone_adapter
from modules.backbone_adapter import (
    EvaClipBackboneAdapter,
    get_eva_clip_backbone_spec,
    load_eva_clip_pretrained,
    normalize_eva_clip_state_dict_for_adapter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeEvaVisual(nn.Module):
    image_size = 224

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 3, bias=False)
        self.rope = nn.Module()
        self.rope.register_buffer("freqs_cos", torch.ones(2, 2))
        self.rope.register_buffer("freqs_sin", torch.zeros(2, 2))

    def forward_features(self, image, return_all_features=False):
        assert return_all_features is True
        batch = image.size(0)
        return torch.arange(batch * 2 * 4, dtype=image.dtype, device=image.device).view(batch, 2, 4)


class FakeEvaTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_attn_mask_shape = None

    def get_cast_dtype(self):
        return torch.float32

    def forward(self, x, attn_mask=None):
        self.last_attn_mask_shape = tuple(attn_mask.shape) if attn_mask is not None else None
        return x + 1


class FakeEvaClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = FakeEvaVisual()
        self.transformer = FakeEvaTransformer()
        self.vocab_size = 11
        self.token_embedding = nn.Embedding(11, 4)
        self.positional_embedding = nn.Parameter(torch.zeros(5, 4))
        self.ln_final = nn.LayerNorm(4)
        self.text_projection = nn.Parameter(torch.ones(4, 3))
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.register_buffer("attn_mask", torch.zeros(5, 5), persistent=False)


class FakeEvaVisualWithoutHidden(nn.Module):
    def forward(self, image):
        return torch.zeros(image.size(0), 3, dtype=image.dtype, device=image.device)


def test_eva_clip_backbone_spec_matches_b16_interface_dimensions():
    spec = get_eva_clip_backbone_spec(
        "EVA02-CLIP-B-16",
        PROJECT_ROOT / "ref/EVA/EVA-CLIP/rei",
    )

    assert spec.backbone_type == "eva_clip"
    assert spec.embed_dim == 512
    assert spec.image_resolution == 224
    assert spec.vision_patch_size == 16
    assert spec.vision_layers == 12
    assert spec.vision_width == 768
    assert spec.context_length == 77
    assert spec.vocab_size == 49408
    assert spec.transformer_width == 512
    assert spec.transformer_heads == 8
    assert spec.transformer_layers == 12


def test_eva_clip_adapter_returns_projected_text_and_visual_hidden_tokens():
    adapter = EvaClipBackboneAdapter(FakeEvaClip())
    text = torch.tensor([[1, 2, 10], [3, 10, 0]])
    image = torch.zeros(2, 3, 224, 224)

    text_pooled, text_hidden = adapter.encode_text(text, return_hidden=True)
    image_pooled, image_hidden = adapter.encode_image(image, return_hidden=True, video_frame=1)

    assert text_pooled.shape == (2, 3)
    assert text_hidden.shape == (2, 3, 3)
    assert adapter.transformer.last_attn_mask_shape == (3, 3)
    assert image_pooled.shape == (2, 3)
    assert image_hidden.shape == (2, 2, 3)
    assert adapter.output_dim == 3
    assert adapter.supports_text_hidden is True
    assert adapter.supports_visual_hidden is True


def test_eva_clip_adapter_rejects_visual_hidden_when_forward_features_is_unavailable():
    eva_model = FakeEvaClip()
    eva_model.visual = FakeEvaVisualWithoutHidden()
    adapter = EvaClipBackboneAdapter(eva_model)
    image = torch.zeros(2, 3, 224, 224)

    assert adapter.supports_visual_hidden is False
    assert adapter.encode_image(image).shape == (2, 3)
    with pytest.raises(RuntimeError, match="visual hidden.*forward_features"):
        adapter.encode_image(image, return_hidden=True)


def test_eva_clip_pretrained_loader_requires_existing_checkpoint(tmp_path):
    adapter = EvaClipBackboneAdapter(FakeEvaClip())

    with pytest.raises(FileNotFoundError):
        load_eva_clip_pretrained(
            adapter,
            backbone_name="EVA02-CLIP-B-16",
            backbone_path=tmp_path / "missing.pt",
            eva_clip_root=PROJECT_ROOT / "ref/EVA/EVA-CLIP/rei",
        )


def test_normalize_eva_clip_state_dict_maps_custom_text_tower_keys_to_adapter_keys():
    state_dict = {
        "logit_scale": torch.tensor(1.0),
        "visual.patch_embed.proj.weight": torch.ones(2, 3, 1, 1),
        "text.token_embedding.weight": torch.full((3, 4), 2.0),
        "text.positional_embedding": torch.full((5, 4), 3.0),
        "text.transformer.resblocks.0.attn.in_proj_weight": torch.full((12, 4), 4.0),
        "text.ln_final.weight": torch.full((4,), 5.0),
        "text.text_projection": torch.full((4, 3), 6.0),
    }

    normalized = normalize_eva_clip_state_dict_for_adapter(state_dict)

    assert "text.token_embedding.weight" not in normalized
    assert torch.equal(normalized["token_embedding.weight"], state_dict["text.token_embedding.weight"])
    assert torch.equal(normalized["positional_embedding"], state_dict["text.positional_embedding"])
    assert torch.equal(
        normalized["transformer.resblocks.0.attn.in_proj_weight"],
        state_dict["text.transformer.resblocks.0.attn.in_proj_weight"],
    )
    assert torch.equal(normalized["ln_final.weight"], state_dict["text.ln_final.weight"])
    assert torch.equal(normalized["text_projection"], state_dict["text.text_projection"])
    assert torch.equal(normalized["visual.patch_embed.proj.weight"], state_dict["visual.patch_embed.proj.weight"])


def _install_fake_eva_checkpoint_loader(monkeypatch, state_dict):
    eva_clip_package = types.ModuleType("eva_clip")
    eva_clip_package.__path__ = []
    factory_module = types.ModuleType("eva_clip.factory")
    factory_module.load_state_dict = lambda _path, is_openai=False: state_dict
    monkeypatch.setitem(sys.modules, "eva_clip", eva_clip_package)
    monkeypatch.setitem(sys.modules, "eva_clip.factory", factory_module)
    monkeypatch.setattr(backbone_adapter, "resolve_eva_clip_root", lambda _root=None: PROJECT_ROOT)
    monkeypatch.setattr(backbone_adapter, "_prepare_eva_model_config", lambda *_args, **_kwargs: {})


def test_eva_clip_pretrained_loader_accepts_complete_checkpoint(monkeypatch, tmp_path):
    adapter = EvaClipBackboneAdapter(FakeEvaClip())
    state_dict = {key: value.clone() for key, value in adapter.state_dict().items()}
    checkpoint_path = tmp_path / "complete.pt"
    checkpoint_path.touch()
    _install_fake_eva_checkpoint_loader(monkeypatch, state_dict)

    load_eva_clip_pretrained(adapter, backbone_path=checkpoint_path, eva_clip_root=PROJECT_ROOT)


def test_eva_clip_pretrained_loader_allows_only_rebuildable_rope_buffers_missing(monkeypatch, tmp_path):
    adapter = EvaClipBackboneAdapter(FakeEvaClip())
    state_dict = {
        key: value.clone()
        for key, value in adapter.state_dict().items()
        if not key.endswith(("freqs_cos", "freqs_sin"))
    }
    checkpoint_path = tmp_path / "without-rope.pt"
    checkpoint_path.touch()
    _install_fake_eva_checkpoint_loader(monkeypatch, state_dict)

    load_eva_clip_pretrained(adapter, backbone_path=checkpoint_path, eva_clip_root=PROJECT_ROOT)


def test_eva_clip_pretrained_loader_rejects_non_rope_missing_key(monkeypatch, tmp_path):
    adapter = EvaClipBackboneAdapter(FakeEvaClip())
    state_dict = {key: value.clone() for key, value in adapter.state_dict().items()}
    state_dict.pop("token_embedding.weight")
    checkpoint_path = tmp_path / "missing-text.pt"
    checkpoint_path.touch()
    _install_fake_eva_checkpoint_loader(monkeypatch, state_dict)

    with pytest.raises(RuntimeError, match=r"missing-text\.pt.*token_embedding\.weight"):
        load_eva_clip_pretrained(adapter, backbone_path=checkpoint_path, eva_clip_root=PROJECT_ROOT)


def test_eva_clip_pretrained_loader_rejects_unexpected_key(monkeypatch, tmp_path):
    adapter = EvaClipBackboneAdapter(FakeEvaClip())
    state_dict = {key: value.clone() for key, value in adapter.state_dict().items()}
    state_dict["unexpected.weight"] = torch.ones(1)
    checkpoint_path = tmp_path / "unexpected.pt"
    checkpoint_path.touch()
    _install_fake_eva_checkpoint_loader(monkeypatch, state_dict)

    with pytest.raises(RuntimeError, match=r"unexpected\.pt.*unexpected\.weight"):
        load_eva_clip_pretrained(adapter, backbone_path=checkpoint_path, eva_clip_root=PROJECT_ROOT)
