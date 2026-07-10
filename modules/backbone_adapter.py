from __future__ import annotations

import copy
import importlib.machinery
import json
import logging
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVA_CLIP_ROOT = PROJECT_ROOT / "ref/EVA/EVA-CLIP/rei"
DEFAULT_EVA_CLIP_B16_PATH = PROJECT_ROOT / "ref/model_weights/eva_clip/EVA02_CLIP_B_psz16_s8B.pt"


@dataclass(frozen=True)
class BackboneSpec:
    backbone_type: str
    backbone_name: str
    embed_dim: int
    image_resolution: int
    vision_layers: int
    vision_width: int
    vision_patch_size: int
    context_length: int
    vocab_size: int
    transformer_width: int
    transformer_heads: int
    transformer_layers: int
    supports_text_hidden: bool = True
    supports_visual_hidden: bool = True


def _resolve_path(path: str | os.PathLike | None, default: Path | None = None) -> Path:
    if path in (None, ""):
        if default is None:
            raise ValueError("A path is required.")
        path = default
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def resolve_eva_clip_root(eva_clip_root: str | os.PathLike | None = None) -> Path:
    root = _resolve_path(eva_clip_root, DEFAULT_EVA_CLIP_ROOT)
    candidates = [
        root,
        root / "rei",
        root / "EVA-CLIP" / "rei",
    ]
    for candidate in candidates:
        if (candidate / "eva_clip" / "model_configs").is_dir():
            return candidate
    raise FileNotFoundError(f"EVA-CLIP python root not found under {root}")


def get_eva_clip_backbone_spec(
    backbone_name: str = "EVA02-CLIP-B-16",
    eva_clip_root: str | os.PathLike | None = None,
) -> BackboneSpec:
    root = resolve_eva_clip_root(eva_clip_root)
    config_path = root / "eva_clip" / "model_configs" / f"{backbone_name}.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"EVA-CLIP config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    vision_cfg = cfg["vision_cfg"]
    text_cfg = cfg["text_cfg"]
    return BackboneSpec(
        backbone_type="eva_clip",
        backbone_name=backbone_name,
        embed_dim=int(cfg["embed_dim"]),
        image_resolution=int(vision_cfg["image_size"]),
        vision_layers=int(vision_cfg["layers"]),
        vision_width=int(vision_cfg["width"]),
        vision_patch_size=int(vision_cfg["patch_size"]),
        context_length=int(text_cfg["context_length"]),
        vocab_size=int(text_cfg["vocab_size"]),
        transformer_width=int(text_cfg["width"]),
        transformer_heads=int(text_cfg["heads"]),
        transformer_layers=int(text_cfg["layers"]),
        supports_text_hidden=True,
        supports_visual_hidden=True,
    )


def _drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def _to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def _ensure_timm_layer_fallback():
    try:
        import timm  # noqa: F401
        return
    except ImportError:
        pass

    timm_module = types.ModuleType("timm")
    timm_layers = types.ModuleType("timm.layers")
    timm_loss = types.ModuleType("timm.loss")
    timm_models = types.ModuleType("timm.models")
    timm_models_layers = types.ModuleType("timm.models.layers")
    timm_module.__spec__ = importlib.machinery.ModuleSpec("timm", loader=None)
    timm_layers.__spec__ = importlib.machinery.ModuleSpec("timm.layers", loader=None)
    timm_loss.__spec__ = importlib.machinery.ModuleSpec("timm.loss", loader=None)
    timm_models.__spec__ = importlib.machinery.ModuleSpec("timm.models", loader=None)
    timm_models_layers.__spec__ = importlib.machinery.ModuleSpec("timm.models.layers", loader=None)
    timm_module.__path__ = []

    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing

        def forward(self, x, target):
            confidence = 1.0 - self.smoothing
            logprobs = F.log_softmax(x, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            return (confidence * nll_loss + self.smoothing * smooth_loss).mean()

    for layer_module in (timm_layers, timm_models_layers):
        layer_module.drop_path = _drop_path
        layer_module.to_2tuple = _to_2tuple
        layer_module.trunc_normal_ = torch.nn.init.trunc_normal_

    timm_loss.LabelSmoothingCrossEntropy = LabelSmoothingCrossEntropy
    timm_module.layers = timm_layers
    timm_module.loss = timm_loss
    timm_module.models = timm_models
    timm_models.layers = timm_models_layers

    sys.modules.setdefault("timm", timm_module)
    sys.modules.setdefault("timm.layers", timm_layers)
    sys.modules.setdefault("timm.loss", timm_loss)
    sys.modules.setdefault("timm.models", timm_models)
    sys.modules.setdefault("timm.models.layers", timm_models_layers)


def _prepare_eva_model_config(backbone_name: str, eva_clip_root: Path, use_xattn: bool):
    _ensure_timm_layer_fallback()
    if str(eva_clip_root) not in sys.path:
        sys.path.insert(0, str(eva_clip_root))

    from eva_clip.factory import get_model_config

    model_cfg = get_model_config(backbone_name)
    if model_cfg is None:
        raise RuntimeError(f"EVA-CLIP model config not found: {backbone_name}")

    model_cfg = copy.deepcopy(model_cfg)
    if not use_xattn:
        model_cfg.get("vision_cfg", {})["xattn"] = False
        model_cfg.get("text_cfg", {})["xattn"] = False

    os.environ["RoPE"] = "1" if model_cfg.get("vision_cfg", {}).get("rope", False) else "0"
    return model_cfg


def normalize_eva_clip_state_dict_for_adapter(state_dict):
    normalized = {}
    text_prefixes = {
        "text.token_embedding.": "token_embedding.",
        "text.positional_embedding": "positional_embedding",
        "text.transformer.": "transformer.",
        "text.ln_final.": "ln_final.",
        "text.text_projection": "text_projection",
    }
    for key, value in state_dict.items():
        new_key = key
        for source_prefix, target_prefix in text_prefixes.items():
            if key.startswith(source_prefix):
                new_key = target_prefix + key[len(source_prefix):]
                break
        normalized[new_key] = value
    return normalized


class EvaClipBackboneAdapter(nn.Module):
    """Expose EVA-CLIP through the CLIP-like interface used by UATVR."""

    def __init__(self, eva_model: nn.Module, spec: BackboneSpec | None = None):
        super().__init__()
        self.spec = spec
        self.visual = eva_model.visual
        self.transformer = eva_model.transformer
        self.vocab_size = eva_model.vocab_size
        self.token_embedding = eva_model.token_embedding
        self.positional_embedding = eva_model.positional_embedding
        self.ln_final = eva_model.ln_final
        self.text_projection = eva_model.text_projection
        self.logit_scale = eva_model.logit_scale
        if self.text_projection.ndim != 2:
            raise ValueError(
                "EVA-CLIP text_projection must be rank 2 to declare the adapter output dimension; "
                f"got shape={tuple(self.text_projection.shape)}"
            )
        self.output_dim = int(self.text_projection.shape[-1])
        self.supports_text_hidden = True
        self.supports_visual_hidden = callable(getattr(self.visual, "forward_features", None))

        if spec is not None and spec.embed_dim != self.output_dim:
            raise ValueError(
                f"EVA-CLIP adapter output_dim={self.output_dim} does not match spec.embed_dim={spec.embed_dim}"
            )
        visual_head_dim = getattr(getattr(self.visual, "head", None), "out_features", None)
        if visual_head_dim is not None and int(visual_head_dim) != self.output_dim:
            raise ValueError(
                f"EVA-CLIP visual head output_dim={visual_head_dim} does not match "
                f"text output_dim={self.output_dim}"
            )

        attn_mask = getattr(eva_model, "attn_mask", None)
        if attn_mask is not None:
            self.register_buffer("attn_mask", attn_mask, persistent=False)
        else:
            self.attn_mask = None

    @property
    def dtype(self):
        try:
            return next(self.visual.parameters()).dtype
        except StopIteration:
            return self.token_embedding.weight.dtype

    def encode_text(self, text, return_hidden=False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding[: x.size(1), :].to(cast_dtype)
        x = x.permute(1, 0, 2)

        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[: x.size(0), : x.size(0)].to(device=x.device)

        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        hidden = x @ self.text_projection
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), text.argmax(dim=-1)]

        if return_hidden:
            return pooled, hidden
        return pooled

    def encode_image(self, image, return_hidden=False, video_frame=-1):
        image = image.type(self.dtype)
        if return_hidden:
            if not self.supports_visual_hidden:
                raise RuntimeError(
                    "EVA-CLIP visual hidden states were requested, but this visual tower does not "
                    "provide forward_features; pooled output cannot substitute for patch tokens."
                )
            tokens = self.visual.forward_features(image, return_all_features=True)
            if hasattr(self.visual, "norm"):
                tokens = self.visual.norm(tokens)
            hidden = self.visual.head(tokens) if hasattr(self.visual, "head") else tokens
            if getattr(self.visual, "fc_norm", None) is not None:
                pooled = self.visual.head(self.visual.fc_norm(tokens.mean(1)))
            else:
                pooled = hidden[:, 0, :]
        else:
            pooled = self.visual(image)
            hidden = None

        if return_hidden:
            return pooled, hidden
        return pooled

    def forward(self, image, text):
        image_features = F.normalize(self.encode_image(image), dim=-1)
        text_features = F.normalize(self.encode_text(text), dim=-1)
        logit_scale = self.logit_scale.exp()
        return logit_scale * image_features @ text_features.t(), logit_scale * text_features @ image_features.t()


def build_eva_clip_backbone(
    backbone_name: str = "EVA02-CLIP-B-16",
    backbone_path: str | os.PathLike | None = None,
    eva_clip_root: str | os.PathLike | None = None,
    use_xattn: bool = False,
    load_pretrained: bool = True,
) -> EvaClipBackboneAdapter:
    root = resolve_eva_clip_root(eva_clip_root)
    model_cfg = _prepare_eva_model_config(backbone_name, root, use_xattn=use_xattn)

    from eva_clip.model import CLIP as EvaCLIP

    eva_model = EvaCLIP(**model_cfg, cast_dtype=None)
    spec = get_eva_clip_backbone_spec(backbone_name, root)
    adapter = EvaClipBackboneAdapter(eva_model, spec=spec)
    if not load_pretrained:
        return adapter

    load_eva_clip_pretrained(adapter, backbone_name, backbone_path, root, use_xattn=use_xattn)
    return adapter


def load_eva_clip_pretrained(
    adapter: EvaClipBackboneAdapter,
    backbone_name: str = "EVA02-CLIP-B-16",
    backbone_path: str | os.PathLike | None = None,
    eva_clip_root: str | os.PathLike | None = None,
    use_xattn: bool = False,
):
    root = resolve_eva_clip_root(eva_clip_root)
    _prepare_eva_model_config(backbone_name, root, use_xattn=use_xattn)

    from eva_clip.factory import load_state_dict

    checkpoint_path = _resolve_path(backbone_path, DEFAULT_EVA_CLIP_B16_PATH)
    if checkpoint_path.is_file():
        state_dict = load_state_dict(str(checkpoint_path), is_openai=False)
        state_dict = normalize_eva_clip_state_dict_for_adapter(state_dict)
        try:
            incompatible_keys = adapter.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to load EVA-CLIP checkpoint {checkpoint_path}: {exc}") from exc

        allowed_missing = {
            key
            for key in incompatible_keys.missing_keys
            if key.endswith(".freqs_cos") or key.endswith(".freqs_sin")
        }
        invalid_missing = sorted(set(incompatible_keys.missing_keys) - allowed_missing)
        unexpected = sorted(incompatible_keys.unexpected_keys)
        if invalid_missing or unexpected:
            details = []
            if invalid_missing:
                details.append(f"missing_keys={invalid_missing[:10]}")
            if unexpected:
                details.append(f"unexpected_keys={unexpected[:10]}")
            raise RuntimeError(
                f"Invalid EVA-CLIP checkpoint {checkpoint_path}: " + "; ".join(details)
            )
        if allowed_missing:
            logging.info(
                "Rebuilding %d omitted EVA-CLIP RoPE buffers: %s",
                len(allowed_missing),
                sorted(allowed_missing)[:10],
            )
        return
    raise FileNotFoundError(f"EVA-CLIP checkpoint not found: {checkpoint_path}")
