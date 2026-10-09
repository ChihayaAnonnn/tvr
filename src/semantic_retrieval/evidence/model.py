"""Frozen CLIP, observation-only temporal encoder and Gaussian pair matcher."""
from dataclasses import replace
import hashlib

import torch
from torch import nn

from semantic_retrieval.models.clip import ClipBackbone
from semantic_retrieval.models.temporal import TemporalEncoder
from semantic_retrieval.models.weights import load_clip_state
from semantic_retrieval.types import QueryFeatures
from semantic_retrieval.evidence.encoding import pack_observations
from semantic_retrieval.evidence.frame_cache import FrameCacheIdentity
from semantic_retrieval.evidence.matcher import GaussianPairMatcher
from semantic_retrieval.evidence.types import digest
from semantic_retrieval.config import config_payload


class EvidenceMatcherModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stage = 'matcher'
        base = config.base
        self.clip = ClipBackbone(base.clip)
        load_clip_state(base.clip_weights, base.clip_sha256, self.clip)
        self.clip.requires_grad_(False).eval()
        self.temporal = TemporalEncoder(base.dim, base.num_layers, base.num_heads, base.max_frames)
        self.matcher = GaussianPairMatcher(base.dim, scale_floor=config.scale_floor)

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    def temporal_features(self, observed):
        frames, pooled = self.temporal(observed.features.frame_tokens, observed.features.frame_mask,
                                       position_ids=observed.position_ids)
        return replace(observed.features, frame_tokens=frames, pooled_video=pooled)

    def point_scores(self, observed):
        return self.matcher.base(self.temporal_features(observed), observed.pair_indices)

    def forward(self, observed, *, sampling):
        return self.matcher(self.temporal_features(observed), observed.pair_indices, observed.query_offsets,
                            pair_ids=observed.pair_ids, coverage=observed.pair_coverage, sampling=sampling)

    def encode_inputs(self, query, states, frame_source, cache):
        device = next(self.parameters()).device
        with torch.no_grad():
            tokens, pooled = self.clip.encode_text(query.input_ids.to(device), query.text_mask.to(device))
        queries = QueryFeatures(tokens, pooled, query.text_mask.to(device))
        base = self.config.base
        preprocessing_sha256 = frame_source.preprocessing_sha256
        clip_config_sha256 = digest(config_payload(base)['clip'])

        def reader(candidate, ids):
            identity = FrameCacheIdentity(candidate.content_sha256, base.clip_sha256, clip_config_sha256, preprocessing_sha256)
            # Only successful acquired observations reach this reader. Cache misses can encode those frames.
            features = []
            for i in ids:
                path = cache._path(candidate.canonical_id, i, identity)
                if path.exists():
                    feature = cache.read(candidate.canonical_id, i, identity=identity)
                else:
                    image = frame_source(candidate, (i,)).to(device)
                    if image.shape != (1, 3, base.clip.image_size, base.clip.image_size):
                        raise ValueError('frame source returned invalid CLIP image shape')
                    with torch.no_grad():
                        feature = self.clip.encode_frames(image[None], torch.ones((1,1), device=device, dtype=torch.bool))[0,0]
                    cache.write(candidate.canonical_id, i, feature, identity=identity)
                features.append(feature.to(device))
            return torch.stack(features)
        reader.preprocessing_sha256 = preprocessing_sha256
        reader.clip_sha256 = base.clip_sha256
        return pack_observations(states, queries, reader)


def load_reference_parts(path, student, *, expected_sha256):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('reference file hash mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if set(payload) != {'clip', 'temporal', 'matcher.base'}:
        raise ValueError('reference snapshot must contain exactly clip/temporal/matcher.base')
    student.clip.load_state_dict(payload['clip'], strict=True)
    student.temporal.load_state_dict(payload['temporal'], strict=True)
    student.matcher.base.load_state_dict(payload['matcher.base'], strict=True)
    student.clip.requires_grad_(False)
