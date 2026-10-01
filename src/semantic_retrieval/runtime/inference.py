"""Label-free, frozen, identity-checked bounded retrieval."""

import hashlib
from dataclasses import asdict

import torch

from experiment_artifacts import artifact_json_bytes
from semantic_retrieval.evaluation.episodes import EpisodePrediction
from semantic_retrieval.models.grouping import ValidatedGrouper
from semantic_retrieval.retrieval.policy import RetrievalPolicy
from semantic_retrieval.retrieval.semantic_set import SemanticSet
from semantic_retrieval.runtime.checkpoint import model_sha256
from semantic_retrieval.types import EncodedFeatures, PairScores, QueryInputs


def policy_sha256(config):
    return hashlib.sha256(artifact_json_bytes(asdict(config))).hexdigest()


class RetrievalService:
    def __init__(self, model, index, cache, calibration, policy, *, content_sha256):
        self.model = model
        self.index = index
        self.cache = cache
        self.calibration = calibration
        self.policy = RetrievalPolicy(policy)
        self.content_sha256 = dict(content_sha256)
        self.last_prediction = None
        self._verify()
        self.grouper = ValidatedGrouper(model.grouping, model.config.grouping) if model.grouping is not None else None
        if set(self.content_sha256) != set(index.ids):
            raise ValueError("content inventory must cover the canonical gallery")

    def _verify(self):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("deployment model must be frozen and in eval mode")
        identity = self.index.identity
        cal = self.calibration.identity
        if model_sha256(self.model) != identity.model_sha256 or (
            cal.model_sha256,
            cal.preprocessing_sha256,
            cal.gallery_version,
            cal.candidate_policy_sha256,
        ) != (
            identity.model_sha256,
            identity.preprocessing_sha256,
            identity.gallery_version,
            policy_sha256(self.policy.config),
        ):
            raise ValueError("deployment identity mismatch")

    @torch.no_grad()
    def retrieve(self, inputs, *, budget):
        self._verify()
        if type(inputs) is not QueryInputs or len(inputs.input_ids) != 1:
            raise ValueError("online retrieval requires one label-free query")
        if type(budget) is not int or budget < 0 or budget > self.policy.config.max_budget:
            raise ValueError("invalid request budget")
        query = self.model.encode_query(inputs)
        embedding = self.model.project_query(query)
        if embedding.device != self.index.embeddings.device:
            raise ValueError("query/index device mismatch")
        cached = {}
        attempted = []
        initial = ()
        used = 0
        for k in self.policy.config.k_sequence:
            attempted.append(k)
            ids = self.index.search(embedding, k, expected_identity=self.index.identity)[0]
            new = [i for i in ids if i not in cached][: budget - used]
            if new:
                videos = [
                    self.cache.read(i, expected_identity=self.index.identity, content_sha256=self.content_sha256[i])
                    for i in new
                ]
                shape = videos[0].frame_tokens.shape[1:]
                if any(v.frame_tokens.shape[1:] != shape for v in videos):
                    raise ValueError("cache frame capacities disagree")
                frames = torch.cat([v.frame_tokens for v in videos]).to(embedding.device)
                pooled = torch.cat([v.pooled_video for v in videos]).to(embedding.device)
                mask = torch.cat([v.frame_mask for v in videos]).to(embedding.device)
                f = EncodedFeatures(query.text_tokens, query.pooled_text, frames, pooled, query.text_mask, mask)
                pairs = torch.stack(
                    [
                        torch.zeros(len(new), dtype=torch.long, device=embedding.device),
                        torch.arange(len(new), device=embedding.device),
                    ],
                    -1,
                )
                scores = self.model.score_pairs(f, pairs)
                for j, i in enumerate(new):
                    cached[i] = PairScores(
                        scores.logits[j : j + 1], scores.features[j : j + 1], scores.dustbin_mass[j : j + 1]
                    )
                used += len(new)
            ordered = tuple(sorted(cached))
            if not initial:
                initial = ordered
            if ordered:
                p = PairScores(
                    torch.cat([cached[i].logits for i in ordered]),
                    torch.cat([cached[i].features for i in ordered]),
                    torch.cat([cached[i].dustbin_mass for i in ordered]),
                )
            else:
                z = embedding.new_empty(0)
                p = PairScores(z, embedding.new_empty((0, self.model.matcher.feature_dim)), z)
            presence = self.model.score_presence(p, torch.tensor([0, len(ordered)], device=embedding.device))
            probabilities = self.calibration.pair_probabilities(p.logits)
            semantic = SemanticSet.build(
                ordered,
                PairScores(torch.logit(probabilities.clamp(1e-7, 1 - 1e-7)), p.features, p.dustbin_mass),
                support_threshold=self.policy.config.pair_threshold,
                group_predictor=self.grouper,
            )
            remaining = budget - used
            # No additional canonical candidates remain after covering this library.
            if used == len(self.index.ids):
                remaining = 0
            decision = self.policy.decide(
                probabilities,
                float(self.calibration.presence_probabilities(presence)[0]),
                semantic,
                attempted_k=attempted,
                remaining_budget=remaining,
            )
            if decision.action != "expand":
                self.last_prediction = EpisodePrediction("", decision, ordered, initial, tuple(attempted), used)
                return decision
        raise RuntimeError("bounded policy must terminate")
