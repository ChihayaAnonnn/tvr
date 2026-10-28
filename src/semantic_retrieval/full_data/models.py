"""Authenticated trainable baselines; official UATVR and CLIP4Clip meanP."""
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from semantic_retrieval.models.clip import ClipBackbone
from semantic_retrieval.runtime.distributed import gather_with_grad, gather_metadata
from semantic_retrieval.types import QueryInputs, EncoderInputs


def _freeze_official_bottom(clip):
    # Exact upstream freeze_layer_num=0, linear_patch=2d policy.
    train_prefixes = ('ln_final.', 'text_projection', 'logit_scale', 'visual.ln_post.',
                      'visual.proj', 'visual.transformer.resblocks.', 'transformer.resblocks.')
    for name, parameter in clip.named_parameters():
        parameter.requires_grad_(name.startswith(train_prefixes))


class BaselineModel(nn.Module):
    def __init__(self, method, clip_config, clip_state, *, temporal_layers=4,
                 extra_video_tokens=2, extra_text_tokens=2, samples=7, batching='unique_video_all_captions_v1'):
        super().__init__()
        if batching not in {'unique_video_all_captions_v1', 'caption_shuffle_v1'}:
            raise ValueError('invalid registered batch policy')
        self.batching = batching
        self.method, self.clip_config = method, clip_config
        reference = ClipBackbone(clip_config)
        expected = reference.state_dict()
        if (clip_state.keys() != expected.keys()
                or any(not isinstance(v, torch.Tensor) or v.shape != expected[k].shape
                       or v.dtype != torch.float32 or not torch.isfinite(v).all() for k, v in clip_state.items())):
            raise ValueError('invalid strict CLIP initialization state')
        if method == 'clip4clip':
            reference.load_state_dict(clip_state, strict=True)
            self.core = reference.float()
        elif method == 'uatvr':
            from .upstream_uatvr.modeling import UATVR
            from .upstream_uatvr.nn_components.module_cross import CrossConfig
            if not 1 <= temporal_layers <= clip_config.text_layers:
                raise ValueError('invalid pretrained temporal layer count')
            task = SimpleNamespace(max_words=min(32, clip_config.context_length-12), max_frames=12,
                                   loose_type=True, linear_patch='2d', sim_header='seqTransf',
                                   cross_num_hidden_layers=temporal_layers, n_video_embeddings=samples,
                                   n_text_embeddings=samples, extra_video_cls_num=extra_video_tokens,
                                   extra_text_cls_num=extra_text_tokens, local_rank=0)
            cross = CrossConfig(clip_config.vocab_size, hidden_size=clip_config.text_width,
                                max_position_embeddings=clip_config.context_length,
                                num_attention_heads=clip_config.text_heads)
            self.core = UATVR(cross, dict(clip_state), task).float()
            state = self.core.state_dict()
            # Explicit complete initialization, rather than upstream's tolerant loader.
            mapped = {'clip.'+k: v for k, v in clip_state.items()}
            mapped['frame_position_embeddings.weight'] = clip_state['positional_embedding']
            for k, v in clip_state.items():
                if k.startswith('transformer.resblocks.') and int(k.split('.')[2]) < temporal_layers:
                    mapped[k.replace('transformer.', 'transformerClip.', 1)] = v
            for k, v in mapped.items():
                if k not in state or state[k].shape != v.shape:
                    raise ValueError('CLIP/temporal initialization shape mismatch')
                state[k] = v.float()
            self.core.load_state_dict(state, strict=True)
        else:
            raise ValueError('unregistered baseline method')
        _freeze_official_bottom(self.clip)

    @property
    def clip(self):
        return self.core.clip if self.method == 'uatvr' else self.core

    def encode_text(self, input_ids, mask):
        QueryInputs(input_ids, mask)
        cfg = self.clip_config
        lengths = mask.sum(1)
        if (input_ids.shape[1] > min(32, cfg.context_length) or (input_ids >= cfg.vocab_size).any()
                or (lengths < 2).any() or (input_ids[:, 0] != cfg.vocab_size-2).any()
                or not torch.equal(mask, torch.arange(input_ids.shape[1], device=mask.device)[None] < lengths[:, None])
                or (input_ids[torch.arange(len(input_ids), device=input_ids.device), lengths-1] != cfg.vocab_size-1).any()):
            raise ValueError('invalid baseline BOS/EOS/context/padding')
        if self.method == 'uatvr':
            pooled, tokens = self.core.get_sequence_output(input_ids, torch.zeros_like(input_ids), mask.long(), shaped=True)
        else:
            tokens, pooled = self.clip.encode_text(input_ids, mask)
            pooled = pooled[:, None]
        return dict(pooled=pooled, tokens=tokens, mask=mask)

    def encode_video(self, video, mask):
        cfg = self.clip_config
        EncoderInputs(torch.tensor([[cfg.vocab_size-2, cfg.vocab_size-1]], device=video.device),
                      torch.ones((1, 2), dtype=torch.bool, device=video.device), video, mask)
        if video.shape[-2:] != (cfg.image_size, cfg.image_size) or video.shape[1] > 12:
            raise ValueError('invalid baseline video spatial/frame dimensions')
        if self.method == 'uatvr':
            tokens = self.core.get_visual_output(video.flatten(0, 1), mask.long(), shaped=True, video_frame=video.shape[1])
        else:
            tokens = self.clip.encode_frames(video, mask)
        return dict(tokens=tokens, mask=mask)

    def _logits(self, text, video):
        if self.method == 'uatvr':
            result = self.core.get_similarity_logits(text['pooled'], text['tokens'], video['tokens'],
                text['mask'].long(), video['mask'].long(), shaped=True, loose_type=True)
            return result if self.training else (result[0], None, None)
        pooled, frames, mask = text['pooled'].squeeze(1), video['tokens'], video['mask']
        if self.training:
            pooled, frames, mask = gather_with_grad(pooled), gather_with_grad(frames), gather_metadata(mask)
        pooled = F.normalize(pooled, dim=-1)
        frames = F.normalize(frames, dim=-1)
        video_pooled = (frames*mask[..., None]).sum(1)/mask.sum(1)[:, None]
        logits = self.clip.logit_scale.exp()*pooled @ F.normalize(video_pooled, dim=-1).T
        zero = logits.sum()*0
        return logits, zero, zero

    def score(self, text, video):
        if self.training:
            raise ValueError('cross-product scoring requires evaluation mode')
        devices = [video['tokens'].device.index] if video['tokens'].is_cuda else []
        # Upstream computes unused Gaussian samples even at inference. Preserve
        # training RNG across scoring chunks, without changing the WTI ranking.
        with torch.random.fork_rng(devices=devices):
            logits = self._logits(text, video)[0]
        if not torch.isfinite(logits).all():
            raise ValueError('nonfinite baseline scores')
        return logits

    def forward(self, input_ids, text_mask, video, frame_mask, video_ids):
        if not self.training or video_ids.dtype != torch.long or video_ids.shape != (len(input_ids),) or len(video) != len(input_ids):
            raise ValueError('paired baseline training input contract mismatch')
        global_ids = gather_metadata(video_ids)
        if self.batching == 'unique_video_all_captions_v1' and len(global_ids.unique()) != len(global_ids):
            raise ValueError('official paired loss requires unique video identities per global batch')
        logits, mil, kl = self._logits(self.encode_text(input_ids, text_mask), self.encode_video(video, frame_mask))
        if logits.shape != (len(global_ids), len(global_ids)):
            raise ValueError('global paired loss shape mismatch')
        target = torch.arange(len(logits), device=logits.device)
        contrastive = (F.cross_entropy(logits, target)+F.cross_entropy(logits.T, target))/2
        loss = contrastive+mil+kl
        if not torch.isfinite(loss):
            raise ValueError('nonfinite baseline loss')
        return dict(loss=loss, contrastive=contrastive, mil=mil, kl=kl, logits=logits)
