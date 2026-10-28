"""Frozen-threshold decision policy with finite expansion budget."""

import math

import torch

from semantic_retrieval.types import Decision


class RetrievalPolicy:
    def __init__(self, config):
        self.config = config

    def decide(self, pair_probabilities, presence_probability, semantic_set, *, attempted_k, remaining_budget):
        ids = semantic_set.canonical_ids
        if (
            pair_probabilities.shape != (len(ids),)
            or not torch.isfinite(pair_probabilities).all()
            or ((pair_probabilities < 0) | (pair_probabilities > 1)).any()
            or not math.isfinite(presence_probability)
            or not 0 <= presence_probability <= 1
        ):
            raise ValueError("invalid probabilities")
        if (
            type(remaining_budget) is not int
            or remaining_budget < 0
            or remaining_budget > self.config.max_budget
            or not attempted_k
            or tuple(attempted_k) != self.config.k_sequence[: len(attempted_k)]
        ):
            raise ValueError("invalid expansion history/budget")
        supported = tuple(i for i, p in zip(ids, pair_probabilities.tolist()) if p >= self.config.pair_threshold)
        if presence_probability < self.config.presence_threshold or not supported:
            n = len(attempted_k)
            if n < len(self.config.k_sequence):
                k = self.config.k_sequence[n]
                if k - attempted_k[-1] <= remaining_budget:
                    return Decision("expand", (), "candidate support insufficient", k)
            return Decision("abstain", (), "candidate support insufficient; expansion budget exhausted", None)
        if semantic_set.conflict_signal:
            return Decision("clarify", supported, "validated conflicting explanations", None)
        return Decision("return", supported, "supported canonical answers", None)
