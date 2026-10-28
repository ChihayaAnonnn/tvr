"""Named tensor contracts; inference features never contain supervision."""

from dataclasses import dataclass

import torch
from torch import Tensor


def _mask(mask: Tensor, shape: tuple[int, ...], name: str) -> None:
    if mask.dtype != torch.bool or tuple(mask.shape) != shape:
        raise ValueError(f"{name} must be a boolean mask of shape {shape}")
    if mask.shape[0] and not mask.any(dim=-1).all():
        raise ValueError(f"{name} contains a sample with no valid positions")


@dataclass(frozen=True)
class QueryInputs:
    input_ids: Tensor
    text_mask: Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.input_ids.dtype != torch.long or not all(self.input_ids.shape):
            raise ValueError("input_ids must be a nonempty [Q,L] int64 tensor")
        if (self.input_ids < 0).any():
            raise ValueError("input_ids cannot be negative")
        _mask(self.text_mask, tuple(self.input_ids.shape), "text_mask")


@dataclass(frozen=True)
class EncoderInputs(QueryInputs):
    video: Tensor
    frame_mask: Tensor

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_inputs(self)


def validate_inputs(inputs: EncoderInputs) -> None:
    QueryInputs(inputs.input_ids, inputs.text_mask)
    video = inputs.video
    if video.ndim != 5 or video.shape[2] != 3 or not all(video.shape[1:]):
        raise ValueError("video must have shape [V,T,3,H,W] with nonzero spatial/frame dimensions")
    if not video.is_floating_point() or not torch.isfinite(video).all():
        raise ValueError("video must contain finite floating point values")
    _mask(inputs.frame_mask, tuple(video.shape[:2]), "frame_mask")


@dataclass(frozen=True)
class QueryFeatures:
    text_tokens: Tensor
    pooled_text: Tensor
    text_mask: Tensor


@dataclass(frozen=True)
class VideoFeatures:
    frame_tokens: Tensor
    pooled_video: Tensor
    frame_mask: Tensor


@dataclass(frozen=True)
class EncodedFeatures:
    text_tokens: Tensor
    pooled_text: Tensor
    frame_tokens: Tensor
    pooled_video: Tensor
    text_mask: Tensor
    frame_mask: Tensor


@dataclass(frozen=True)
class PairScores:
    logits: Tensor
    features: Tensor
    dustbin_mass: Tensor | None
    attention: Tensor | None = None
    alignment_similarity: Tensor | None = None


@dataclass(frozen=True)
class Targets:
    relevance: Tensor
    candidate_presence: Tensor
    group_pairs: Tensor | None = None
    group_labels: Tensor | None = None

    def __post_init__(self) -> None:
        validate_targets(self)


def validate_targets(targets: Targets) -> None:
    if targets.relevance.ndim != 2 or targets.candidate_presence.shape != targets.relevance.shape[:1]:
        raise ValueError("targets require relevance [Q,V] and candidate_presence [Q]")
    values = [targets.relevance, targets.candidate_presence]
    if (targets.group_pairs is None) != (targets.group_labels is None):
        raise ValueError("group_pairs and group_labels must be supplied together")
    if targets.group_pairs is not None:
        pairs = targets.group_pairs
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
            raise ValueError("group_pairs must be [M,2] int64 indices")
        if (pairs < 0).any() or targets.group_labels.shape != pairs.shape[:1]:
            raise ValueError("group supervision indices or labels are invalid")
        values.append(targets.group_labels)
    integer_types = (torch.int8, torch.int16, torch.int32, torch.int64)
    for value in values:
        if value.dtype not in integer_types or not ((value == -1) | (value == 0) | (value == 1)).all():
            raise ValueError("labels must be integer -1 (unknown), 0 or 1")


@dataclass(frozen=True)
class EpisodeBatch:
    inputs: EncoderInputs
    targets: Targets
    query_ids: tuple[str, ...]
    canonical_ids: tuple[str, ...]
    pair_indices: Tensor
    query_offsets: Tensor
    group_provenance: tuple[str, ...] | None = None
    benchmark_positive_ids: tuple[tuple[str, ...], ...] | None = None

    def inference_queries(self) -> QueryInputs:
        return QueryInputs(self.inputs.input_ids, self.inputs.text_mask)


@dataclass(frozen=True)
class Decision:
    action: str
    answer_ids: tuple[str, ...]
    reason: str
    next_k: int | None

    def __post_init__(self):
        if self.action not in {"return", "expand", "clarify", "abstain"}:
            raise ValueError("invalid decision action")
        if (self.action == "expand") != (self.next_k is not None):
            raise ValueError("expansion requires next K")
