"""ViT/text CLIP primitives migrated from OpenAI CLIP via module_clip.py (MIT)."""

from collections import OrderedDict
from dataclasses import dataclass, fields

import torch
from torch import nn

from semantic_retrieval.types import EncoderInputs, QueryInputs


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks((x, -1))[0]


@dataclass(frozen=True)
class ClipConfig:
    embed_dim: int
    image_size: int
    vision_width: int
    vision_layers: int
    patch_size: int
    context_length: int
    vocab_size: int
    text_width: int
    text_heads: int
    text_layers: int

    def __post_init__(self):
        if any(type(getattr(self, f.name)) is not int or getattr(self, f.name) <= 0 for f in fields(self)):
            raise ValueError("CLIP dimensions must be positive integers; only ViT is supported")
        if self.vision_width % 64 or self.text_width % self.text_heads or self.image_size % self.patch_size:
            raise ValueError("incompatible CLIP head/patch dimensions")
        if self.context_length < 2 or self.vocab_size < 3:
            raise ValueError("CLIP needs BOS/EOS and a context of at least two tokens")


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)  # rewrite it? check
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, "__call__"):
            attn_mask_ = self.attn_mask(x.size(0))  # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x_tuple: tuple):
        x, video_frame = x_tuple
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return (x, video_frame)


class VisualTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution  # 224
        self.output_dim = output_dim  # 512

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width**-0.5  # 768 ** -0.5 = 0.03608..
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # (768)
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )  # 197 (224*224/16*16), 768
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        x = self.conv1(x)

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD    frames * (tokens+1) * Dimension

        # Move the three lines below to `encode_image` for entire hidden sequence
        # x = self.ln_post(x[:, 0, :])
        # if self.proj is not None:
        #     x = x @ self.proj

        # whwu update
        # [bs*T 50 768]

        # x = self.ln_post(x) @ self.proj         # frames * (tokens+1) * 768->512
        # x = x[:, 0, :]          # BS*T, 512 -----> embedding for each frame

        return x


class ClipBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = VisualTransformer(
            config.image_size,
            config.patch_size,
            config.vision_width,
            config.vision_layers,
            config.vision_width // 64,
            config.embed_dim,
        )
        self.transformer = Transformer(config.text_width, config.text_layers, config.text_heads, self.causal_mask)
        self.token_embedding = nn.Embedding(config.vocab_size, config.text_width)
        self.positional_embedding = nn.Parameter(torch.randn(config.context_length, config.text_width) * 0.01)
        self.ln_final = LayerNorm(config.text_width)
        self.text_projection = nn.Parameter(torch.randn(config.text_width, config.embed_dim) * config.text_width**-0.5)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    @staticmethod
    def causal_mask(length):
        return torch.full((length, length), float("-inf")).triu_(1)

    def encode_text(self, input_ids, text_mask):
        QueryInputs(input_ids, text_mask)
        if input_ids.shape[1] > self.config.context_length or (input_ids >= self.config.vocab_size).any():
            raise ValueError("text exceeds CLIP context or vocabulary")
        # CLIP padding is a suffix; EOS is the highest vocabulary ID.
        lengths = text_mask.sum(1)
        expected = torch.arange(input_ids.shape[1], device=input_ids.device)[None] < lengths[:, None]
        if not torch.equal(expected, text_mask) or (lengths < 2).any():
            raise ValueError("text must use suffix padding and include BOS/EOS")
        rows = torch.arange(len(input_ids), device=input_ids.device)
        if (input_ids[:, 0] != self.config.vocab_size - 2).any() or (
            input_ids[rows, lengths - 1] != self.config.vocab_size - 1
        ).any():
            raise ValueError("text must start with BOS and end with EOS")
        hidden = self.token_embedding(input_ids) + self.positional_embedding[: input_ids.shape[1]]
        hidden = self.transformer(hidden.permute(1, 0, 2)).permute(1, 0, 2)
        tokens = (self.ln_final(hidden) @ self.text_projection).float()
        return tokens.masked_fill(~text_mask[..., None], 0), tokens[rows, lengths - 1]

    def encode_frames(self, video, frame_mask):
        # Reuse input validation without introducing an encoder-model dependency.
        EncoderInputs(
            torch.tensor([[self.config.vocab_size - 2, self.config.vocab_size - 1]], device=video.device),
            torch.ones((1, 2), dtype=torch.bool, device=video.device),
            video,
            frame_mask,
        )
        if video.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError("video image size does not match CLIP")
        result = video.new_zeros((*video.shape[:2], self.config.embed_dim), dtype=torch.float32)
        if frame_mask.any():
            hidden = self.visual(video[frame_mask].float())
            result[frame_mask] = (self.visual.ln_post(hidden[:, 0]) @ self.visual.proj).float()
        return result
