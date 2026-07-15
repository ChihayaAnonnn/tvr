from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import modules.pair_evidence_refiner as refiner_module
from modules.pair_evidence_refiner import DualSourcePairEvidenceRefiner

_DIAGNOSTIC_NAMES = {
    "data_ambiguity_mean",
    "view_disagreement_mean",
    "gate_mean",
    "representation_update_norm",
    "representation_update_rate",
}


def _normalized_features(*shape: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(*shape, generator=generator), dim=-1)


def _uniform_weights(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    weight = mask.to(dtype=dtype)
    return weight / weight.sum(dim=-1, keepdim=True)


def _reference_wti(
    text: torch.Tensor,
    video: torch.Tensor,
    text_mask: torch.Tensor,
    video_mask: torch.Tensor,
    text_weight: torch.Tensor,
    video_weight: torch.Tensor,
) -> torch.Tensor:
    similarities = torch.einsum("atd,bvd->abtv", text, video)
    pair_valid = (
        text_mask[:, None, :, None].bool()
        & video_mask[None, :, None, :].bool()
    )
    similarities = similarities.masked_fill(
        ~pair_valid, torch.finfo(similarities.dtype).min
    )
    text_score = similarities.max(dim=-1).values
    text_score = text_score.masked_fill(~text_mask[:, None, :].bool(), 0.0)
    text_score = torch.einsum("abt,at->ab", text_score, text_weight)
    video_score = similarities.max(dim=-2).values
    video_score = video_score.masked_fill(
        ~video_mask[None, :, :].bool(), 0.0
    )
    video_score = torch.einsum("abv,bv->ab", video_score, video_weight)
    return (text_score + video_score) / 2.0


def _rectangular_inputs(*, requires_grad: bool = False):
    text = _normalized_features(3, 4, 12, seed=1)
    video = _normalized_features(5, 3, 12, seed=2)
    text_mask = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
    )
    video_mask = torch.tensor(
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]]
    )
    text_weight = _uniform_weights(text_mask, text.dtype)
    video_weight = _uniform_weights(video_mask, video.dtype)
    if requires_grad:
        text.requires_grad_()
        video.requires_grad_()
        text_weight.requires_grad_()
        video_weight.requires_grad_()
    return text, video, text_mask, video_mask, text_weight, video_weight


def _clone_inputs_with_grad(inputs):
    return tuple(
        value.detach().clone().requires_grad_(value.requires_grad)
        if value.is_floating_point()
        else value.clone()
        for value in inputs
    )


def _independent_dense_pair_oracle(
    text_token: torch.Tensor,
    frame_token: torch.Tensor,
    attention_mask: torch.Tensor,
    video_mask: torch.Tensor,
    text_weight: torch.Tensor,
    video_weight: torch.Tensor,
    *,
    num_views: int = 4,
    lambda_max: float = 0.1,
    alignment_temperature: float = 0.07,
):
    """Direct P2-006 definition, intentionally independent of block code."""

    def entropy(probabilities: torch.Tensor, dim: int) -> torch.Tensor:
        tiny = torch.finfo(probabilities.dtype).tiny
        return -(
            probabilities * probabilities.clamp_min(tiny).log()
        ).sum(dim=dim)

    diagnostic_sums = {
        name: text_token.new_zeros((), dtype=torch.float32)
        for name in _DIAGNOSTIC_NAMES
    }
    representation_count = 0
    score_rows = []

    # Work one pair at a time and physically slice padding out.  This is
    # deliberately a different organization from the production block path.
    for query_index in range(text_token.size(0)):
        valid_text = attention_mask[query_index].bool()
        pair_text = text_token[query_index, valid_text]
        pair_text_weight = text_weight[query_index, valid_text]
        row_scores = []

        for candidate_index in range(frame_token.size(0)):
            valid_frame = video_mask[candidate_index].bool()
            pair_frame = frame_token[candidate_index, valid_frame]
            pair_video_weight = video_weight[candidate_index, valid_frame]
            text_count = pair_text.size(0)
            frame_count = pair_frame.size(0)

            text_to_frame_views = []
            frame_to_text_views = []
            text_view_entropies = []
            frame_view_entropies = []
            for view_index in range(num_views):
                text_view = F.normalize(
                    pair_text[..., view_index::num_views].float(),
                    dim=-1,
                    eps=1e-6,
                )
                frame_view = F.normalize(
                    pair_frame[..., view_index::num_views].float(),
                    dim=-1,
                    eps=1e-6,
                )
                logits = (
                    text_view @ frame_view.transpose(0, 1)
                ) / alignment_temperature
                text_distribution = torch.softmax(logits, dim=1)
                frame_distribution = torch.softmax(logits, dim=0)
                text_to_frame_views.append(text_distribution)
                frame_to_text_views.append(frame_distribution)
                text_view_entropies.append(entropy(text_distribution, 1))
                frame_view_entropies.append(entropy(frame_distribution, 0))

            text_alignments = torch.stack(text_to_frame_views)
            frame_alignments = torch.stack(frame_to_text_views)
            text_entropies = torch.stack(text_view_entropies)
            frame_entropies = torch.stack(frame_view_entropies)
            mean_text_alignment = text_alignments.mean(dim=0)
            mean_frame_alignment = frame_alignments.mean(dim=0)

            if frame_count == 1:
                text_ambiguity = pair_text.new_zeros(text_count)
                text_disagreement = pair_text.new_zeros(text_count)
            else:
                text_ambiguity = (
                    text_entropies.mean(dim=0) / math.log(frame_count)
                ).clamp(0.0, 1.0)
                text_js = entropy(mean_text_alignment, 1) - (
                    text_entropies.mean(dim=0)
                )
                text_disagreement = (
                    text_js / math.log(min(num_views, frame_count))
                ).clamp(0.0, 1.0)

            if text_count == 1:
                frame_ambiguity = pair_frame.new_zeros(frame_count)
                frame_disagreement = pair_frame.new_zeros(frame_count)
            else:
                frame_ambiguity = (
                    frame_entropies.mean(dim=0) / math.log(text_count)
                ).clamp(0.0, 1.0)
                frame_js = entropy(mean_frame_alignment, 0) - (
                    frame_entropies.mean(dim=0)
                )
                frame_disagreement = (
                    frame_js / math.log(min(num_views, text_count))
                ).clamp(0.0, 1.0)

            text_gate = (
                (1.0 - text_ambiguity) * text_disagreement
            ).clamp(0.0, 1.0).detach()
            frame_gate = (
                (1.0 - frame_ambiguity) * frame_disagreement
            ).clamp(0.0, 1.0).detach()

            # Alignment is low-dimensional, but the routed context and the
            # residual update are explicitly performed in the original D.
            text_context = mean_text_alignment @ pair_frame
            frame_context = mean_frame_alignment.transpose(0, 1) @ pair_text
            refined_text = F.normalize(
                pair_text
                + lambda_max
                * text_gate.unsqueeze(-1)
                * (text_context - pair_text),
                dim=-1,
            )
            refined_frame = F.normalize(
                pair_frame
                + lambda_max
                * frame_gate.unsqueeze(-1)
                * (frame_context - pair_frame),
                dim=-1,
            )

            refined_similarity = refined_text @ refined_frame.transpose(0, 1)
            text_score = (
                refined_similarity.max(dim=1).values * pair_text_weight
            ).sum()
            frame_score = (
                refined_similarity.max(dim=0).values * pair_video_weight
            ).sum()
            row_scores.append((text_score + frame_score) / 2.0)

            text_update_norm = torch.linalg.vector_norm(
                (refined_text - pair_text).float(), dim=-1
            )
            frame_update_norm = torch.linalg.vector_norm(
                (refined_frame - pair_frame).float(), dim=-1
            )
            diagnostic_sums["data_ambiguity_mean"] += (
                text_ambiguity.float().sum()
                + frame_ambiguity.float().sum()
            ).detach()
            diagnostic_sums["view_disagreement_mean"] += (
                text_disagreement.float().sum()
                + frame_disagreement.float().sum()
            ).detach()
            diagnostic_sums["gate_mean"] += (
                text_gate.float().sum() + frame_gate.float().sum()
            ).detach()
            diagnostic_sums["representation_update_norm"] += (
                text_update_norm.sum() + frame_update_norm.sum()
            ).detach()
            diagnostic_sums["representation_update_rate"] += (
                (text_update_norm > 1e-6).float().sum()
                + (frame_update_norm > 1e-6).float().sum()
            ).detach()
            representation_count += text_count + frame_count

        score_rows.append(torch.stack(row_scores))

    diagnostics = {
        name: (value / representation_count).detach()
        for name, value in diagnostic_sums.items()
    }
    return torch.stack(score_rows), diagnostics


def test_refiner_has_no_parameters_buffers_or_rng_state_and_repr_is_explicit():
    refiner = DualSourcePairEvidenceRefiner()
    inputs = _rectangular_inputs(requires_grad=True)
    cpu_rng_state = torch.random.get_rng_state()

    refiner(*inputs)

    assert list(refiner.parameters()) == []
    assert list(refiner.buffers()) == []
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_state)
    assert "num_views=4" in repr(refiner)
    assert "alignment_temperature=0.07" in repr(refiner)
    assert "query_block_size=16" in repr(refiner)
    assert "candidate_block_size=32" in repr(refiner)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for FP16 coverage"
)
def test_fp16_zero_feature_subspaces_remain_finite():
    device = torch.device("cuda")
    dtype = torch.float16
    text = torch.zeros(2, 3, 8, device=device, dtype=dtype)
    video = torch.zeros(3, 2, 8, device=device, dtype=dtype)
    text[..., 0] = 1.0
    video[..., 0] = 1.0
    text.requires_grad_()
    video.requires_grad_()
    text_mask = torch.ones(2, 3, device=device, dtype=torch.long)
    video_mask = torch.ones(3, 2, device=device, dtype=torch.long)
    text_weight = _uniform_weights(text_mask, dtype).requires_grad_()
    video_weight = _uniform_weights(video_mask, dtype).requires_grad_()
    refiner = DualSourcePairEvidenceRefiner(
        query_block_size=1,
        candidate_block_size=2,
        use_checkpoint=False,
    ).to(device)

    output = refiner(
        text,
        video,
        text_mask,
        video_mask,
        text_weight,
        video_weight,
    )
    output.scores.sum().backward()

    assert torch.isfinite(output.scores).all()
    assert all(torch.isfinite(value) for value in output.diagnostics.values())
    for tensor in (text, video, text_weight, video_weight):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for FP16 coverage"
)
def test_fp16_one_sided_zero_subspace_has_finite_nonzero_gradients():
    device = torch.device("cuda")
    dtype = torch.float16
    generator = torch.Generator().manual_seed(73)
    text = F.normalize(torch.randn(2, 3, 8, generator=generator), dim=-1)
    video = F.normalize(torch.randn(3, 2, 8, generator=generator), dim=-1)
    text[..., 1::4] = 0.0
    text = F.normalize(text, dim=-1).to(device=device, dtype=dtype)
    video = video.to(device=device, dtype=dtype)
    text.requires_grad_()
    video.requires_grad_()
    text_mask = torch.ones(2, 3, device=device, dtype=torch.long)
    video_mask = torch.ones(3, 2, device=device, dtype=torch.long)
    text_weight = _uniform_weights(text_mask, dtype).requires_grad_()
    video_weight = _uniform_weights(video_mask, dtype).requires_grad_()
    refiner = DualSourcePairEvidenceRefiner(
        query_block_size=1,
        candidate_block_size=2,
        use_checkpoint=False,
    ).to(device)

    output = refiner(
        text,
        video,
        text_mask,
        video_mask,
        text_weight,
        video_weight,
    )
    coefficients = torch.arange(
        1, output.scores.numel() + 1, device=device, dtype=dtype
    ).reshape_as(output.scores)
    (output.scores * coefficients).sum().backward()

    assert output.diagnostics["gate_mean"] > 0.0
    for tensor in (text, video, text_weight, video_weight):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad).item() > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for FP16 coverage"
)
def test_fp16_masked_zero_padding_is_finite_and_padding_invariant():
    device = torch.device("cuda")
    dtype = torch.float16
    text, video, text_mask, video_mask, text_weight, video_weight = (
        _rectangular_inputs()
    )
    text_mask = text_mask.to(device)
    video_mask = video_mask.to(device)
    text = text.to(device=device, dtype=dtype)
    video = video.to(device=device, dtype=dtype)
    text[~text_mask.bool()] = 0.0
    video[~video_mask.bool()] = 0.0
    text.requires_grad_()
    video.requires_grad_()
    text_weight = text_weight.to(device=device, dtype=dtype).requires_grad_()
    video_weight = video_weight.to(device=device, dtype=dtype).requires_grad_()
    refiner = DualSourcePairEvidenceRefiner(
        query_block_size=2,
        candidate_block_size=2,
        use_checkpoint=False,
    ).to(device)

    output = refiner(
        text,
        video,
        text_mask,
        video_mask,
        text_weight,
        video_weight,
    )
    changed_text = text.detach().clone()
    changed_video = video.detach().clone()
    padding_value = F.normalize(
        torch.arange(1, text.size(-1) + 1, device=device, dtype=dtype),
        dim=0,
    )
    changed_text[~text_mask.bool()] = padding_value
    changed_video[~video_mask.bool()] = -padding_value
    changed_padding = refiner(
        changed_text,
        changed_video,
        text_mask,
        video_mask,
        text_weight.detach(),
        video_weight.detach(),
    )

    assert torch.isfinite(output.scores).all()
    assert all(torch.isfinite(value) for value in output.diagnostics.values())
    torch.testing.assert_close(output.scores, changed_padding.scores)
    for name in _DIAGNOSTIC_NAMES:
        torch.testing.assert_close(
            output.diagnostics[name], changed_padding.diagnostics[name]
        )

    coefficients = torch.linspace(
        -0.8,
        1.2,
        output.scores.numel(),
        device=device,
        dtype=dtype,
    ).reshape_as(output.scores)
    (output.scores * coefficients).sum().backward()
    for tensor in (text, video, text_weight, video_weight):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_lambda_zero_exactly_recovers_weighted_wti_for_rectangular_batches():
    inputs = _rectangular_inputs()
    refiner = DualSourcePairEvidenceRefiner(
        lambda_max=0.0,
        query_block_size=2,
        candidate_block_size=2,
        use_checkpoint=False,
    )

    output = refiner(*inputs)
    expected = _reference_wti(*inputs)

    assert output.scores.shape == (3, 5)
    torch.testing.assert_close(output.scores, expected, atol=1e-6, rtol=1e-6)


def test_dense_eager_and_small_blocks_match_all_outputs_and_input_gradients():
    dense_inputs = _rectangular_inputs(requires_grad=True)
    blocked_inputs = _clone_inputs_with_grad(dense_inputs)
    dense = DualSourcePairEvidenceRefiner(
        query_block_size=3,
        candidate_block_size=5,
        use_checkpoint=False,
    )
    blocked = DualSourcePairEvidenceRefiner(
        query_block_size=1,
        candidate_block_size=2,
        use_checkpoint=False,
    )

    dense_output = dense(*dense_inputs)
    blocked_output = blocked(*blocked_inputs)
    score_coefficients = torch.linspace(-0.7, 1.1, 15).reshape(3, 5)
    (dense_output.scores * score_coefficients).sum().backward()
    (blocked_output.scores * score_coefficients).sum().backward()

    torch.testing.assert_close(
        blocked_output.scores,
        dense_output.scores,
        atol=2e-7,
        rtol=2e-6,
    )
    assert dense_output.diagnostics.keys() == _DIAGNOSTIC_NAMES
    assert blocked_output.diagnostics.keys() == _DIAGNOSTIC_NAMES
    for name in _DIAGNOSTIC_NAMES:
        torch.testing.assert_close(
            blocked_output.diagnostics[name],
            dense_output.diagnostics[name],
            atol=2e-7,
            rtol=2e-6,
        )

    differentiable_inputs = {
        "text_token": 0,
        "frame_token": 1,
        "text_weight": 4,
        "video_weight": 5,
    }
    for name, index in differentiable_inputs.items():
        dense_gradient = dense_inputs[index].grad
        blocked_gradient = blocked_inputs[index].grad
        assert dense_gradient is not None, name
        assert blocked_gradient is not None, name
        assert torch.isfinite(dense_gradient).all(), name
        assert torch.isfinite(blocked_gradient).all(), name
        torch.testing.assert_close(
            blocked_gradient,
            dense_gradient,
            atol=2e-7,
            rtol=2e-6,
        )


def test_independent_dense_oracle_matches_scores_diagnostics_and_gradients():
    text = _normalized_features(2, 3, 8, seed=101).requires_grad_()
    video = _normalized_features(3, 3, 8, seed=102).requires_grad_()
    text_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    video_mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]])
    text_weight = torch.tensor(
        [[0.25, 0.75, 0.0], [0.15, 0.35, 0.50]]
    ).requires_grad_()
    video_weight = torch.tensor(
        [[0.65, 0.35, 0.0], [0.20, 0.30, 0.50], [1.0, 0.0, 0.0]]
    ).requires_grad_()
    production_inputs = (
        text,
        video,
        text_mask,
        video_mask,
        text_weight,
        video_weight,
    )
    oracle_inputs = _clone_inputs_with_grad(production_inputs)
    refiner = DualSourcePairEvidenceRefiner(
        num_views=4,
        lambda_max=0.1,
        alignment_temperature=0.07,
        query_block_size=1,
        candidate_block_size=2,
        use_checkpoint=False,
    )

    production = refiner(*production_inputs)
    oracle_scores, oracle_diagnostics = _independent_dense_pair_oracle(
        *oracle_inputs,
        num_views=4,
        lambda_max=0.1,
        alignment_temperature=0.07,
    )
    score_coefficients = torch.tensor(
        [[-1.1, -0.4, 0.3], [0.7, 1.2, 1.8]]
    )
    (production.scores * score_coefficients).sum().backward()
    (oracle_scores * score_coefficients).sum().backward()

    torch.testing.assert_close(
        production.scores, oracle_scores, atol=8e-7, rtol=3e-6
    )
    assert production.diagnostics.keys() == _DIAGNOSTIC_NAMES
    assert oracle_diagnostics.keys() == _DIAGNOSTIC_NAMES
    for name in _DIAGNOSTIC_NAMES:
        torch.testing.assert_close(
            production.diagnostics[name],
            oracle_diagnostics[name],
            atol=8e-7,
            rtol=3e-6,
        )

    for name, index in {
        "text_token": 0,
        "frame_token": 1,
        "text_weight": 4,
        "video_weight": 5,
    }.items():
        production_gradient = production_inputs[index].grad
        oracle_gradient = oracle_inputs[index].grad
        assert production_gradient is not None, name
        assert oracle_gradient is not None, name
        assert torch.isfinite(production_gradient).all(), name
        assert torch.isfinite(oracle_gradient).all(), name
        torch.testing.assert_close(
            production_gradient,
            oracle_gradient,
            atol=1e-6,
            rtol=5e-6,
        )


@pytest.mark.parametrize(
    "mask_dtype",
    [torch.bool, torch.int64, torch.float32],
    ids=["bool", "int", "float"],
)
def test_binary_bool_integer_and_float_masks_are_equivalent(mask_dtype):
    inputs = list(_rectangular_inputs())
    refiner = DualSourcePairEvidenceRefiner(
        query_block_size=2,
        candidate_block_size=2,
        use_checkpoint=False,
    )
    reference = refiner(*inputs)
    inputs[2] = inputs[2].to(dtype=mask_dtype)
    inputs[3] = inputs[3].to(dtype=mask_dtype)

    output = refiner(*inputs)

    torch.testing.assert_close(output.scores, reference.scores)
    for name in _DIAGNOSTIC_NAMES:
        torch.testing.assert_close(
            output.diagnostics[name], reference.diagnostics[name]
        )


def test_singleton_token_and_frame_have_finite_exact_wti_scores():
    text = _normalized_features(2, 1, 8, seed=41)
    video = _normalized_features(3, 1, 8, seed=42)
    text_mask = torch.ones(2, 1, dtype=torch.bool)
    video_mask = torch.ones(3, 1, dtype=torch.bool)
    text_weight = torch.ones(2, 1)
    video_weight = torch.ones(3, 1)
    refiner = DualSourcePairEvidenceRefiner(
        query_block_size=1,
        candidate_block_size=2,
        use_checkpoint=False,
    )

    output = refiner(
        text, video, text_mask, video_mask, text_weight, video_weight
    )
    expected = _reference_wti(
        text, video, text_mask, video_mask, text_weight, video_weight
    )

    assert torch.isfinite(output.scores).all()
    torch.testing.assert_close(output.scores, expected, atol=1e-6, rtol=1e-6)
    for name in _DIAGNOSTIC_NAMES:
        assert torch.isfinite(output.diagnostics[name]), name
    for name in (
        "data_ambiguity_mean",
        "view_disagreement_mean",
        "gate_mean",
        "representation_update_norm",
        "representation_update_rate",
    ):
        torch.testing.assert_close(
            output.diagnostics[name], torch.tensor(0.0), atol=1e-7, rtol=0.0
        )


def test_all_valid_negative_similarities_remain_finite_and_negative():
    text = F.normalize(torch.ones(2, 3, 8), dim=-1)
    video = -F.normalize(torch.ones(4, 2, 8), dim=-1)
    text_mask = torch.ones(2, 3, dtype=torch.long)
    video_mask = torch.ones(4, 2, dtype=torch.long)
    text_weight = _uniform_weights(text_mask, text.dtype)
    video_weight = _uniform_weights(video_mask, video.dtype)
    refiner = DualSourcePairEvidenceRefiner(
        query_block_size=1,
        candidate_block_size=3,
        use_checkpoint=False,
    )

    output = refiner(
        text, video, text_mask, video_mask, text_weight, video_weight
    )
    expected = _reference_wti(
        text, video, text_mask, video_mask, text_weight, video_weight
    )

    assert torch.isfinite(output.scores).all()
    assert bool((output.scores < 0).all())
    torch.testing.assert_close(output.scores, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        output.scores,
        -torch.ones_like(output.scores),
        atol=1e-6,
        rtol=1e-6,
    )


def test_refiner_is_deterministic_padding_invariant_and_diagnostics_are_detached():
    inputs = _rectangular_inputs()
    text, video, text_mask, video_mask, text_weight, video_weight = inputs
    refiner = DualSourcePairEvidenceRefiner(
        query_block_size=2,
        candidate_block_size=3,
        use_checkpoint=False,
    )

    first = refiner(*inputs)
    second = refiner(*inputs)
    changed_text = text.clone()
    changed_video = video.clone()
    changed_text[~text_mask.bool()] = _normalized_features(
        int((~text_mask.bool()).sum()), text.size(-1), seed=31
    )
    changed_video[~video_mask.bool()] = _normalized_features(
        int((~video_mask.bool()).sum()), video.size(-1), seed=32
    )
    changed_padding = refiner(
        changed_text,
        changed_video,
        text_mask,
        video_mask,
        text_weight,
        video_weight,
    )

    assert torch.equal(first.scores, second.scores)
    torch.testing.assert_close(first.scores, changed_padding.scores)
    assert first.diagnostics.keys() == {
        "data_ambiguity_mean",
        "view_disagreement_mean",
        "gate_mean",
        "representation_update_norm",
        "representation_update_rate",
    }
    for name, value in first.diagnostics.items():
        assert value.shape == (), name
        assert not value.requires_grad, name
        assert torch.isfinite(value), name
        torch.testing.assert_close(value, changed_padding.diagnostics[name])
    assert 0.0 <= first.diagnostics["data_ambiguity_mean"] <= 1.0
    assert 0.0 <= first.diagnostics["view_disagreement_mean"] <= 1.0
    assert 0.0 <= first.diagnostics["gate_mean"] <= 1.0
    assert 0.0 <= first.diagnostics["representation_update_rate"] <= 1.0


def test_uniform_identical_evidence_is_ambiguous_but_has_no_view_disagreement():
    text = F.normalize(torch.ones(2, 3, 8), dim=-1)
    video = F.normalize(torch.ones(3, 2, 8), dim=-1)
    text_mask = torch.ones(2, 3, dtype=torch.long)
    video_mask = torch.ones(3, 2, dtype=torch.long)
    text_weight = _uniform_weights(text_mask, text.dtype)
    video_weight = _uniform_weights(video_mask, video.dtype)
    refiner = DualSourcePairEvidenceRefiner(use_checkpoint=False)

    output = refiner(
        text, video, text_mask, video_mask, text_weight, video_weight
    )

    torch.testing.assert_close(
        output.diagnostics["data_ambiguity_mean"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        output.diagnostics["view_disagreement_mean"],
        torch.tensor(0.0),
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.diagnostics["gate_mean"],
        torch.tensor(0.0),
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.diagnostics["representation_update_norm"],
        torch.tensor(0.0),
        atol=1e-7,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
def test_cpu_supported_precision_forward_and_all_input_gradients_are_finite(
    dtype,
):
    inputs = tuple(
        value.to(dtype=dtype).detach().requires_grad_(True)
        if value.is_floating_point()
        else value
        for value in _rectangular_inputs()
    )
    refiner = DualSourcePairEvidenceRefiner(
        query_block_size=2,
        candidate_block_size=2,
        use_checkpoint=False,
    )

    try:
        output = refiner(*inputs)
        output.scores.float().square().mean().backward()
    except RuntimeError as error:
        unsupported = any(
            marker in str(error).lower()
            for marker in ("not implemented", "unsupported")
        )
        if dtype == torch.bfloat16 and unsupported:
            pytest.skip(f"CPU bfloat16 operator unavailable: {error}")
        raise

    assert output.scores.dtype == dtype
    assert torch.isfinite(output.scores).all()
    for name, value in output.diagnostics.items():
        assert name in _DIAGNOSTIC_NAMES
        assert torch.isfinite(value), name
    for name, index in {
        "text_token": 0,
        "frame_token": 1,
        "text_weight": 4,
        "video_weight": 5,
    }.items():
        gradient = inputs[index].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name


def test_random_nonzero_gate_preserves_nonzero_representation_gradients():
    inputs = _rectangular_inputs(requires_grad=True)
    refiner = DualSourcePairEvidenceRefiner(
        query_block_size=2,
        candidate_block_size=3,
        use_checkpoint=False,
    )

    output = refiner(*inputs)

    assert output.diagnostics["gate_mean"] > 1e-4
    assert output.diagnostics["representation_update_rate"] > 0.0
    score_coefficients = torch.arange(1, 16, dtype=torch.float32).reshape(3, 5)
    (output.scores * score_coefficients).sum().backward()
    for name, representation in (
        ("text_token", inputs[0]),
        ("frame_token", inputs[1]),
    ):
        assert representation.grad is not None, name
        assert torch.isfinite(representation.grad).all(), name
        assert torch.count_nonzero(representation.grad).item() > 0, name


def test_checkpointed_blocks_match_eager_scores_and_gradients(monkeypatch):
    eager_inputs = _rectangular_inputs(requires_grad=True)
    checkpoint_inputs = tuple(
        value.detach().clone().requires_grad_(value.requires_grad)
        if value.is_floating_point()
        else value.clone()
        for value in eager_inputs
    )
    eager = DualSourcePairEvidenceRefiner(
        query_block_size=2,
        candidate_block_size=3,
        use_checkpoint=False,
    )
    checkpointed = DualSourcePairEvidenceRefiner(
        query_block_size=2,
        candidate_block_size=3,
        use_checkpoint=True,
    )
    real_checkpoint = refiner_module.checkpoint
    checkpoint_calls = []

    def recording_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append((args[0].shape[0], args[1].shape[0]))
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(refiner_module, "checkpoint", recording_checkpoint)

    eager_output = eager(*eager_inputs)
    checkpoint_output = checkpointed(*checkpoint_inputs)
    eager_output.scores.square().sum().backward()
    checkpoint_output.scores.square().sum().backward()

    assert checkpoint_calls == [(2, 3), (2, 2), (1, 3), (1, 2)]
    torch.testing.assert_close(checkpoint_output.scores, eager_output.scores)
    for eager_value, checkpoint_value in zip(
        eager_inputs, checkpoint_inputs, strict=True
    ):
        if eager_value.requires_grad:
            assert eager_value.grad is not None
            assert checkpoint_value.grad is not None
            assert torch.isfinite(eager_value.grad).all()
            assert torch.isfinite(checkpoint_value.grad).all()
            torch.testing.assert_close(
                checkpoint_value.grad, eager_value.grad, atol=1e-6, rtol=1e-5
            )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_views": 3}, "exactly four"),
        ({"lambda_max": -0.1}, r"lambda_max must be.*\[0, 1\]"),
        ({"lambda_max": 1.1}, r"lambda_max must be.*\[0, 1\]"),
        ({"alignment_temperature": 0.0}, "temperature must be"),
        ({"query_block_size": 0}, "query_block_size must be positive"),
        (
            {"candidate_block_size": 0},
            "candidate_block_size must be positive",
        ),
    ],
)
def test_refiner_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DualSourcePairEvidenceRefiner(**kwargs)


def test_refiner_rejects_too_few_feature_dimensions_and_empty_samples():
    text = F.normalize(torch.ones(2, 2, 3), dim=-1)
    video = F.normalize(torch.ones(3, 2, 3), dim=-1)
    text_mask = torch.ones(2, 2, dtype=torch.long)
    video_mask = torch.ones(3, 2, dtype=torch.long)
    text_weight = _uniform_weights(text_mask, text.dtype)
    video_weight = _uniform_weights(video_mask, video.dtype)
    refiner = DualSourcePairEvidenceRefiner()

    with pytest.raises(ValueError, match="at least num_views"):
        refiner(
            text, video, text_mask, video_mask, text_weight, video_weight
        )

    text = _normalized_features(2, 2, 8, seed=7)
    video = _normalized_features(3, 2, 8, seed=8)
    text_mask[1] = 0
    text_weight = _uniform_weights(text_mask.clamp_min(1), text.dtype)
    with pytest.raises(ValueError, match=r"text token.*indices=\[1\]"):
        refiner(
            text, video, text_mask, video_mask, text_weight, video_weight
        )
