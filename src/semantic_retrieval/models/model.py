"""New retrieval model composed from explicitly configured primitives."""

from dataclasses import replace

from torch import nn

from semantic_retrieval.models.answerability import AnswerabilityHead, ScoreOnlyAnswerabilityHead
from semantic_retrieval.models.clip import ClipBackbone
from semantic_retrieval.models.coarse import CoarseProjector
from semantic_retrieval.models.grouping import EquivalencePredictor
from semantic_retrieval.models.pair_matcher import PairMatcher
from semantic_retrieval.models.temporal import TemporalEncoder
from semantic_retrieval.models.weights import load_clip_state
from semantic_retrieval.types import EncodedFeatures, QueryFeatures, VideoFeatures


class SemanticRetrievalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.clip = ClipBackbone(config.clip)
        load_clip_state(config.clip_weights, config.clip_sha256, self.clip)
        self.temporal = TemporalEncoder(config.dim, config.num_layers, config.num_heads, config.max_frames)
        self.coarse = CoarseProjector(config.dim, config.coarse_dim)
        self.matcher = PairMatcher(config.dim, wti_only=config.pilot_arm == "A0")
        self.answerability = (
            ScoreOnlyAnswerabilityHead() if config.pilot_arm == "A4" else AnswerabilityHead(self.matcher.feature_dim)
        )
        self.grouping = EquivalencePredictor(self.matcher.feature_dim) if config.grouping is not None else None

    def encode_query(self, inputs):
        tokens, pooled = self.clip.encode_text(inputs.input_ids, inputs.text_mask)
        return QueryFeatures(tokens, pooled, inputs.text_mask)

    def encode_gallery(self, video, frame_mask):
        frames = self.clip.encode_frames(video, frame_mask)
        frames, pooled = self.temporal(frames, frame_mask)
        return VideoFeatures(frames, pooled, frame_mask)

    def encode(self, inputs):
        q = self.encode_query(inputs)
        v = self.encode_gallery(inputs.video, inputs.frame_mask)
        return EncodedFeatures(q.text_tokens, q.pooled_text, v.frame_tokens, v.pooled_video, q.text_mask, v.frame_mask)

    def project_query(self, features):
        return self.coarse.project_text(features.pooled_text)

    def project_gallery(self, features):
        return self.coarse.project_video(features.pooled_video)

    def coarse_embeddings(self, features):
        return self.project_query(features), self.project_gallery(features)

    def score_pairs(self, features, pair_indices, *, return_alignment=False):
        return self.matcher(features, pair_indices, return_alignment=return_alignment)

    def score_presence(self, pair_scores, query_offsets):
        if self.config.pilot_arm == "A2":
            pair_scores = replace(
                pair_scores, features=pair_scores.features.detach(), logits=pair_scores.logits.detach()
            )
        return self.answerability(pair_scores, query_offsets)

    def forward(self, inputs, pair_indices, query_offsets, *, group_pairs=None, return_alignment=False):
        features = self.encode(inputs)
        pairs = self.score_pairs(features, pair_indices, return_alignment=return_alignment)
        q, v = self.coarse_embeddings(features)
        group_logits = (
            self.grouping(pairs.features[group_pairs[:, 0]], pairs.features[group_pairs[:, 1]])
            if self.grouping is not None and group_pairs is not None
            else None
        )
        return features, pairs, self.score_presence(pairs, query_offsets), q, v, group_logits
