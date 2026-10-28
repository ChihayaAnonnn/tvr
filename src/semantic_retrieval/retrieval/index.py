"""Canonical exact search with deterministic tie-breaking."""

import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IndexIdentity:
    model_sha256: str
    gallery_version: str
    preprocessing_sha256: str

    def __post_init__(self):
        if (
            not all(
                isinstance(x, str) and re.fullmatch("[0-9a-f]{64}", x)
                for x in (self.model_sha256, self.preprocessing_sha256)
            )
            or not isinstance(self.gallery_version, str)
            or not self.gallery_version
        ):
            raise ValueError("invalid index identity")


class CanonicalIndex:
    def __init__(self, ids, embeddings, identity):
        if (
            embeddings.ndim != 2
            or not embeddings.is_floating_point()
            or not torch.isfinite(embeddings).all()
            or len(ids) != len(embeddings)
            or embeddings.requires_grad
        ):
            raise ValueError("index requires finite frozen embeddings")
        if not isinstance(identity, IndexIdentity) or not all(isinstance(i, str) and i for i in ids):
            raise ValueError("invalid canonical identity")
        rows = {}
        for i, row in zip(ids, embeddings):
            if i in rows and not torch.equal(rows[i], row):
                raise ValueError("canonical alias embeddings conflict")
            rows[i] = row
        self.ids = tuple(sorted(rows))
        self.embeddings = torch.stack([rows[i] for i in self.ids]).clone() if rows else embeddings.clone()
        self.identity = identity
        self.chunk_size = 1024

    def search(self, query, k, *, expected_identity):
        if expected_identity != self.identity:
            raise ValueError("index identity mismatch")
        if type(k) is not int or k <= 0 or type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError("positive k/chunk required")
        if (
            query.ndim != 2
            or query.shape[1] != self.embeddings.shape[1]
            or not torch.isfinite(query).all()
            or query.device != self.embeddings.device
        ):
            raise ValueError("invalid query embeddings")
        best = [[] for _ in query]
        for start in range(0, len(self.ids), self.chunk_size):
            scores = (
                (query.float() @ self.embeddings[start : start + self.chunk_size].float().T).detach().cpu().tolist()
            )
            for b, row in zip(best, scores):
                b.extend((score, i) for score, i in zip(row, self.ids[start : start + self.chunk_size]))
                b.sort(key=lambda x: (-x[0], x[1]))
                del b[k:]
        return [[i for _, i in row] for row in best]
