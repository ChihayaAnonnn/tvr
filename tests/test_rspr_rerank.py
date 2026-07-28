import logging
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import main_task_retrieval
import modules.rspr_rerank as rspr_rerank
from main_task_retrieval import _run_on_single_gpu, validate_rspr_cli
from metrics import compute_metrics
from modules.modeling import UATVR
from modules.rspr_rerank import rerank_top_r
from modules.stochastic_prototype_ranking import (
    BidirectionalSoftPrototypeMatcher,
    RSPRCore,
)


class _SpyMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.pair_batch_sizes = []
        self.forward_calls = 0

    def forward(self, *_args, **_kwargs):
        self.forward_calls += 1
        raise AssertionError("full-matrix matcher.forward must not be called")

    def score_pairs(self, text_samples, video_samples):
        self.pair_batch_sizes.append(text_samples.size(0))
        scores = (text_samples * video_samples).sum(dim=-1).mean(dim=-1)
        uncertainty = (text_samples - video_samples).square().mean(dim=(1, 2))
        return SimpleNamespace(logits=scores, pair_uncertainty=uncertainty)


def _inputs(seed=7):
    generator = torch.Generator().manual_seed(seed)
    text_mean = torch.randn(3, 8, generator=generator)
    video_mean = torch.randn(5, 8, generator=generator)
    text_samples = torch.randn(3, 4, 8, generator=generator)
    video_samples = torch.randn(5, 4, 8, generator=generator)
    deterministic_logits = torch.randn(3, 5, generator=generator)
    return (
        deterministic_logits,
        text_mean,
        video_mean,
        text_samples,
        video_samples,
    )


def _rerank(matcher, *, top_r=2, pair_chunk_size=3, **overrides):
    kwargs = {
        "top_r": top_r,
        "deterministic_temperature": 2.0,
        "probabilistic_temperature": 0.5,
        "probabilistic_weight": 0.25,
        "pair_chunk_size": pair_chunk_size,
    }
    kwargs.update(overrides)
    return rerank_top_r(*_inputs(), matcher, **kwargs)


def _mean_matrix(text_mean, video_mean):
    return F.normalize(text_mean, dim=-1) @ F.normalize(video_mean, dim=-1).T


def _candidate_mask(scores, top_r, dim):
    mask = torch.zeros(scores.shape, dtype=torch.bool)
    mask.scatter_(dim, scores.topk(top_r, dim=dim).indices, True)
    return mask


def test_top_r_scores_only_aligned_candidate_pairs_in_bounded_chunks():
    matcher = _SpyMatcher()

    _rerank(matcher, top_r=2, pair_chunk_size=3)

    assert matcher.forward_calls == 0
    assert sum(matcher.pair_batch_sizes) <= 3 * 2 + 5 * 2
    assert max(matcher.pair_batch_sizes) <= 3


def test_top_r_candidate_masks_are_direction_independent():
    matcher = _SpyMatcher()
    output = _rerank(matcher, top_r=1)

    expected_t2v_mask = _candidate_mask(output.recall_logits, 1, dim=1)
    expected_v2t_mask = _candidate_mask(output.recall_logits, 1, dim=0)

    assert torch.equal(torch.isfinite(output.text_to_video_logits), expected_t2v_mask)
    assert torch.equal(torch.isfinite(output.video_to_text_logits), expected_v2t_mask)
    assert not torch.equal(expected_t2v_mask, expected_v2t_mask)
    assert torch.equal(
        torch.isfinite(output.pair_uncertainty),
        expected_t2v_mask | expected_v2t_mask,
    )


def test_top_r_recalls_candidates_by_deterministic_logits_by_default():
    """The RSPR mean retrieves worse than DSA, so it must not gate the pool."""

    deterministic_logits, text_mean, video_mean, *_ = _inputs()

    output = _rerank(_SpyMatcher(), top_r=1)

    torch.testing.assert_close(output.recall_logits, deterministic_logits)
    deterministic_mask = _candidate_mask(deterministic_logits, 1, dim=1)
    mean_mask = _candidate_mask(_mean_matrix(text_mean, video_mean), 1, dim=1)
    assert torch.equal(
        torch.isfinite(output.text_to_video_logits), deterministic_mask
    )
    # the fixture is only meaningful if the two scorers actually disagree
    assert not torch.equal(deterministic_mask, mean_mask)


def test_top_r_can_still_recall_by_probability_mean_for_ablation():
    deterministic_logits, text_mean, video_mean, *_ = _inputs()
    expected_mean = _mean_matrix(text_mean, video_mean)

    output = _rerank(_SpyMatcher(), top_r=1, recall_source="mean")

    torch.testing.assert_close(output.recall_logits, expected_mean)
    assert torch.equal(
        torch.isfinite(output.text_to_video_logits),
        _candidate_mask(expected_mean, 1, dim=1),
    )
    assert not torch.equal(
        torch.isfinite(output.text_to_video_logits),
        _candidate_mask(deterministic_logits, 1, dim=1),
    )


def test_rerank_rejects_an_unknown_recall_source():
    with pytest.raises(ValueError, match="recall_source"):
        _rerank(_SpyMatcher(), recall_source="probabilistic")


def test_recall_breaks_complete_ties_by_original_candidate_index():
    deterministic_logits = torch.zeros(5, 5)
    text_mean = torch.zeros(5, 8)
    video_mean = torch.zeros(5, 8)
    text_samples = torch.zeros(5, 4, 8)
    video_samples = torch.zeros(5, 4, 8)

    output = rerank_top_r(
        deterministic_logits,
        text_mean,
        video_mean,
        text_samples,
        video_samples,
        _SpyMatcher(),
        top_r=2,
        deterministic_temperature=1.0,
        probabilistic_temperature=1.0,
        probabilistic_weight=1.0,
        pair_chunk_size=4,
    )

    expected_t2v = torch.zeros(5, 5, dtype=torch.bool)
    expected_t2v[:, :2] = True
    expected_v2t = torch.zeros(5, 5, dtype=torch.bool)
    expected_v2t[:2, :] = True
    assert torch.equal(torch.isfinite(output.text_to_video_logits), expected_t2v)
    assert torch.equal(torch.isfinite(output.video_to_text_logits), expected_v2t)


def test_top_r_uses_probability_formula_on_deterministic_candidates():
    matcher = _SpyMatcher()
    inputs = _inputs()
    output = rerank_top_r(
        *inputs,
        matcher,
        top_r=2,
        deterministic_temperature=2.0,
        probabilistic_temperature=0.5,
        probabilistic_weight=0.25,
        pair_chunk_size=4,
    )

    deterministic_logits, text_mean, video_mean, text_samples, video_samples = inputs
    # the RSPR mean stays available as a diagnostic even when it no longer recalls
    torch.testing.assert_close(
        output.mean_logits, _mean_matrix(text_mean, video_mean)
    )

    row = torch.arange(3).unsqueeze(1).expand(-1, 2)
    col = deterministic_logits.topk(2, dim=1).indices
    probability = (text_samples[row] * video_samples[col]).sum(dim=-1).mean(dim=-1)
    expected = deterministic_logits[row, col] / 2.0 + 0.25 * probability / 0.5
    torch.testing.assert_close(output.text_to_video_logits[row, col], expected)


def test_probabilistic_scale_lifts_the_matcher_onto_the_deterministic_scale():
    """DSA logits carry logit_scale (~100); the matcher returns bare cosines."""

    deterministic_logits, _text_mean, _video_mean, text_samples, video_samples = (
        _inputs()
    )

    unscaled = _rerank(_SpyMatcher(), top_r=2)
    scaled = _rerank(_SpyMatcher(), top_r=2, probabilistic_scale=100.0)

    row = torch.arange(3).unsqueeze(1).expand(-1, 2)
    col = deterministic_logits.topk(2, dim=1).indices
    probability = (text_samples[row] * video_samples[col]).sum(dim=-1).mean(dim=-1)
    deterministic = deterministic_logits[row, col] / 2.0

    torch.testing.assert_close(
        unscaled.text_to_video_logits[row, col],
        deterministic + 0.25 * probability / 0.5,
    )
    torch.testing.assert_close(
        scaled.text_to_video_logits[row, col],
        deterministic + 0.25 * 100.0 * probability / 0.5,
    )


@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf")))
def test_rerank_rejects_nonpositive_or_nonfinite_probabilistic_scale(value):
    with pytest.raises(ValueError, match="probabilistic_scale"):
        _rerank(_SpyMatcher(), probabilistic_scale=value)


def test_rerank_exposes_the_raw_matcher_score_on_the_candidate_pairs():
    """The bare matcher score is the only way to audit what reranking adds.

    Without it a permutation control cannot be run offline, so a reranking gain
    cannot be told apart from a tie-breaking artefact.
    """

    deterministic_logits, _text_mean, _video_mean, text_samples, video_samples = (
        _inputs()
    )

    output = _rerank(_SpyMatcher(), top_r=2)

    row = torch.arange(3).unsqueeze(1).expand(-1, 2)
    col = deterministic_logits.topk(2, dim=1).indices
    probability = (text_samples[row] * video_samples[col]).sum(dim=-1).mean(dim=-1)
    torch.testing.assert_close(output.probability_logits[row, col], probability)


def test_unscored_pairs_leave_the_probability_matrix_undefined():
    output = _rerank(_SpyMatcher(), top_r=2)

    scored = torch.isfinite(output.probability_logits)
    torch.testing.assert_close(scored, torch.isfinite(output.pair_uncertainty))
    assert not scored.all()


def test_zero_top_r_scores_no_pair_probabilities():
    output = _rerank(_SpyMatcher(), top_r=0)

    assert torch.isnan(output.probability_logits).all()


def test_top_r_is_elementwise_deterministic_for_fixed_inputs():
    matcher = _SpyMatcher()

    first = _rerank(matcher)
    second = _rerank(matcher)

    for name in (
        "text_to_video_logits",
        "video_to_text_logits",
        "mean_logits",
        "pair_uncertainty",
        "probability_logits",
    ):
        torch.testing.assert_close(
            getattr(first, name), getattr(second, name), equal_nan=True
        )


def test_zero_top_r_returns_the_recall_matrix_without_calling_matcher():
    matcher = _SpyMatcher()
    deterministic_logits, *_ = _inputs()

    output = _rerank(matcher, top_r=0)

    assert matcher.pair_batch_sizes == []
    assert matcher.forward_calls == 0
    torch.testing.assert_close(output.text_to_video_logits, deterministic_logits)
    torch.testing.assert_close(output.video_to_text_logits, deterministic_logits)
    assert torch.isnan(output.pair_uncertainty).all()


def test_zero_top_r_with_mean_recall_reproduces_pure_rspr_retrieval():
    matcher = _SpyMatcher()
    _deterministic, text_mean, video_mean, *_ = _inputs()

    output = _rerank(matcher, top_r=0, recall_source="mean")

    expected = _mean_matrix(text_mean, video_mean)
    assert matcher.pair_batch_sizes == []
    torch.testing.assert_close(output.text_to_video_logits, expected)
    torch.testing.assert_close(output.video_to_text_logits, expected)


def _distribution_model(mode="stochastic", seed=23):
    model = UATVR.__new__(UATVR)
    nn.Module.__init__(model)
    model.rspr_mode = mode
    model.rspr = RSPRCore(
        dim=8,
        sample_count=4 if mode == "stochastic" else 1,
        eval_sample_count=4,
        eval_seed=seed,
    )
    model.extra_value = 100.0
    model.refinement_calls = []

    def refine_text(self, tokens, mask):
        self.refinement_calls.append("text")
        extra = tokens.new_full((tokens.size(0), 2, tokens.size(2)), self.extra_value)
        extra_mask = mask.new_ones((mask.size(0), 2))
        return torch.cat((tokens + 0.5, extra), dim=1), torch.cat((mask, extra_mask), dim=1)

    def refine_video(self, tokens, mask):
        self.refinement_calls.append("video")
        extra = tokens.new_full((tokens.size(0), 1, tokens.size(2)), self.extra_value)
        extra_mask = mask.new_ones((mask.size(0), 1))
        return torch.cat((tokens - 0.25, extra), dim=1), torch.cat((mask, extra_mask), dim=1)

    model._refine_text_tokens = MethodType(refine_text, model)
    model._refine_video_tokens = MethodType(refine_video, model)
    model.eval()
    return model


def test_distribution_interfaces_refine_modalities_and_exclude_extra_tokens():
    torch.manual_seed(3)
    model = _distribution_model()
    text_tokens = torch.randn(2, 3, 8)
    video_tokens = torch.randn(2, 4, 8)
    text_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    video_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

    text_first = model.get_rspr_text_distribution(text_tokens, text_mask)
    video_first = model.get_rspr_video_distribution(video_tokens, video_mask)
    model.extra_value = -100.0
    text_second = model.get_rspr_text_distribution(text_tokens, text_mask)
    video_second = model.get_rspr_video_distribution(video_tokens, video_mask)

    assert model.refinement_calls == ["text", "video", "text", "video"]
    assert text_first.samples.shape == (2, 4, 8)
    assert video_first.samples.shape == (2, 4, 8)
    for first, second in ((text_first, text_second), (video_first, video_second)):
        torch.testing.assert_close(first.mean, second.mean)
        torch.testing.assert_close(first.logvar, second.logvar)
        torch.testing.assert_close(first.samples, second.samples)


def test_distribution_interfaces_flatten_dataloader_masks():
    model = _distribution_model()
    text_tokens = torch.randn(2, 3, 8)
    video_tokens = torch.randn(2, 4, 8)
    text_mask = torch.tensor([[[1, 1, 0]], [[1, 1, 1]]])
    video_mask = torch.tensor([[[1, 1, 0, 0]], [[1, 1, 1, 1]]])

    text_output = model.get_rspr_text_distribution(text_tokens, text_mask)
    video_output = model.get_rspr_video_distribution(video_tokens, video_mask)

    assert text_output.samples.shape == (2, 4, 8)
    assert video_output.samples.shape == (2, 4, 8)


def test_mean_distribution_interface_returns_one_normalized_mean_sample():
    model = _distribution_model(mode="mean")
    text_tokens = torch.randn(2, 3, 8)
    text_mask = torch.ones(2, 3, dtype=torch.long)

    output = model.get_rspr_text_distribution(text_tokens, text_mask)

    assert output.samples.shape == (2, 1, 8)
    torch.testing.assert_close(output.samples[:, 0], F.normalize(output.mean, dim=-1))


def test_mean_pooling_distribution_interfaces_use_raw_modal_tokens():
    model = _distribution_model()
    model.sim_header = "meanP"

    def forbidden_refinement(*_args, **_kwargs):
        raise AssertionError("meanP must not require seqTransf refinement modules")

    model._refine_text_tokens = forbidden_refinement
    model._refine_video_tokens = forbidden_refinement
    text_tokens = torch.randn(2, 3, 8)
    video_tokens = torch.randn(2, 4, 8)

    text_output = model.get_rspr_text_distribution(
        text_tokens,
        torch.ones(2, 3, dtype=torch.long),
    )
    video_output = model.get_rspr_video_distribution(
        video_tokens,
        torch.ones(2, 4, dtype=torch.long),
    )

    assert text_output.samples.shape == (2, 4, 8)
    assert video_output.samples.shape == (2, 4, 8)


class _EvaluationModel(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(()))
        self.loose_type = True
        self.rspr_mode = "stochastic"
        self.rspr = SimpleNamespace(
            matcher=BidirectionalSoftPrototypeMatcher(temperature=0.3)
        )
        generator = torch.Generator().manual_seed(seed)
        self.text_noise = torch.randn(4, 4, 8, generator=generator) * 0.2
        self.video_noise = torch.randn(5, 4, 8, generator=generator) * 0.2
        self.text_offset = 0
        self.video_offset = 0
        self.text_distribution_batches = []
        self.video_distribution_batches = []

    def get_similarity_logits(
        self,
        sequence_output,
        _text_token,
        visual_output,
        _input_mask,
        _video_mask,
        *,
        loose_type,
    ):
        assert loose_type is True
        return sequence_output[:, 0] @ visual_output[:, 0].T, None

    def get_rspr_text_distribution(self, text_token, _attention_mask):
        batch_size = text_token.size(0)
        start, end = self.text_offset, self.text_offset + batch_size
        self.text_offset = end
        self.text_distribution_batches.append(batch_size)
        # The RSPR head learns its own embedding, so its mean ranks candidates
        # differently from the DSA similarity; flipping keeps that deterministic.
        mean = text_token[:, 0].flip(-1)
        samples = F.normalize(
            mean.unsqueeze(1) + self.text_noise[start:end], dim=-1
        )
        return SimpleNamespace(mean=mean, samples=samples)

    def get_rspr_video_distribution(self, visual_output, _video_mask):
        batch_size = visual_output.size(0)
        start, end = self.video_offset, self.video_offset + batch_size
        self.video_offset = end
        self.video_distribution_batches.append(batch_size)
        mean = visual_output[:, 0]
        samples = F.normalize(
            mean.unsqueeze(1) + self.video_noise[start:end], dim=-1
        )
        return SimpleNamespace(mean=mean, samples=samples)


def _evaluation_inputs(seed):
    generator = torch.Generator().manual_seed(101)
    text = F.normalize(torch.randn(4, 8, generator=generator), dim=-1)
    video = F.normalize(torch.randn(5, 8, generator=generator), dim=-1)
    model = _EvaluationModel(seed)
    args = SimpleNamespace(
        rspr_mode="stochastic",
        rspr_top_r=2,
        rspr_det_temperature=1.5,
        rspr_rerank_temperature=0.7,
        rspr_rerank_weight=0.4,
        rspr_pair_chunk_size=3,
        rspr_recall_source="deterministic",
        rspr_rerank_scale="logit_scale",
        eval_vid_chunk_size=2,
    )
    text_sizes = (2, 2)
    batch_list_t = [
        (
            torch.ones(size, 1, dtype=torch.long),
            torch.zeros(size, 1, dtype=torch.long),
        )
        for size in text_sizes
    ]
    batch_sequence_output_list = []
    start = 0
    for size in text_sizes:
        values = text[start : start + size]
        batch_sequence_output_list.append(
            (values.unsqueeze(1), values.unsqueeze(1))
        )
        start += size
    batch_list_v = [
        (torch.ones(3, 1, dtype=torch.long),),
        (torch.ones(2, 1, dtype=torch.long),),
    ]
    batch_visual_output_list = [
        video[:3].unsqueeze(1),
        video[3:].unsqueeze(1),
    ]
    inputs = (
        model,
        args,
        batch_list_t,
        batch_list_v,
        batch_sequence_output_list,
        batch_visual_output_list,
    )
    return model, inputs


def test_rspr_evaluation_returns_full_directional_matrices_and_encodes_once():
    model, inputs = _evaluation_inputs(seed=17)

    output = _run_on_single_gpu(*inputs)

    assert set(output) == {
        "t2v",
        "v2t",
        "mean",
        "recall",
        "uncertainty",
        "probability",
    }
    assert all(matrix.shape == (4, 5) for matrix in output.values())
    assert model.text_distribution_batches == [2, 2]
    assert model.video_distribution_batches == [2, 2, 1]
    assert (torch.from_numpy(output["v2t"]).isfinite().sum(dim=0) == 2).all()


def test_evaluation_reports_the_raw_probability_matrix():
    """Auditing what reranking adds needs the matcher score, not just S_final."""

    model, inputs = _evaluation_inputs(seed=17)

    output = _run_on_single_gpu(*inputs)

    assert output["probability"].shape == (4, 5)
    scored = np.isfinite(output["probability"])
    assert scored.any() and not scored.all()
    np.testing.assert_array_equal(scored, np.isfinite(output["uncertainty"]))


def test_score_dump_round_trips_every_evaluation_matrix(tmp_path):
    model, inputs = _evaluation_inputs(seed=17)
    destination = tmp_path / "nested" / "scores.npz"
    inputs[1].rspr_dump_scores = str(destination)

    output = _run_on_single_gpu(*inputs)
    main_task_retrieval.dump_rspr_scores(inputs[1], output)

    restored = np.load(destination)
    assert set(restored) == set(output)
    for name, matrix in output.items():
        np.testing.assert_array_equal(restored[name], matrix)


def test_score_dump_is_skipped_when_no_destination_is_configured(tmp_path):
    model, inputs = _evaluation_inputs(seed=17)
    inputs[1].rspr_dump_scores = ""

    main_task_retrieval.dump_rspr_scores(inputs[1], _run_on_single_gpu(*inputs))

    assert list(tmp_path.iterdir()) == []


def test_evaluation_recall_matrix_is_the_deterministic_similarity():
    model, inputs = _evaluation_inputs(seed=17)

    output = _run_on_single_gpu(*inputs)

    text = torch.cat([values for values, _ in inputs[4]])[:, 0]
    video = torch.cat(inputs[5])[:, 0]
    torch.testing.assert_close(
        torch.from_numpy(output["recall"]), text @ video.T
    )
    # the RSPR mean is still reported, it just no longer gates the candidates
    assert not torch.allclose(
        torch.from_numpy(output["recall"]), torch.from_numpy(output["mean"])
    )


def test_evaluation_scales_the_probabilistic_term_by_clip_logit_scale(monkeypatch):
    _model, inputs = _evaluation_inputs(seed=17)
    inputs[0].clip = SimpleNamespace(logit_scale=torch.tensor(4.0).log())
    seen = {}
    original = main_task_retrieval.rerank_top_r

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(main_task_retrieval, "rerank_top_r", spy)
    _run_on_single_gpu(*inputs)

    assert seen["probabilistic_scale"] == pytest.approx(4.0)
    assert seen["recall_source"] == "deterministic"


def test_evaluation_can_disable_scale_matching_for_ablation(monkeypatch):
    _model, inputs = _evaluation_inputs(seed=17)
    inputs[0].clip = SimpleNamespace(logit_scale=torch.tensor(4.0).log())
    inputs[1].rspr_rerank_scale = "none"
    seen = {}
    original = main_task_retrieval.rerank_top_r
    monkeypatch.setattr(
        main_task_retrieval,
        "rerank_top_r",
        lambda *args, **kwargs: (seen.update(kwargs), original(*args, **kwargs))[1],
    )

    _run_on_single_gpu(*inputs)

    assert seen["probabilistic_scale"] == pytest.approx(1.0)


def test_evaluation_without_a_clip_trunk_falls_back_to_unit_scale(monkeypatch):
    _model, inputs = _evaluation_inputs(seed=17)
    seen = {}
    original = main_task_retrieval.rerank_top_r
    monkeypatch.setattr(
        main_task_retrieval,
        "rerank_top_r",
        lambda *args, **kwargs: (seen.update(kwargs), original(*args, **kwargs))[1],
    )

    _run_on_single_gpu(*inputs)

    assert seen["probabilistic_scale"] == pytest.approx(1.0)


def test_rspr_evaluation_is_repeatable_and_seed_only_changes_reranking():
    _, first_inputs = _evaluation_inputs(seed=17)
    _, repeated_inputs = _evaluation_inputs(seed=17)
    _, changed_inputs = _evaluation_inputs(seed=19)

    first = _run_on_single_gpu(*first_inputs)
    repeated = _run_on_single_gpu(*repeated_inputs)
    changed = _run_on_single_gpu(*changed_inputs)

    for key in first:
        torch.testing.assert_close(
            torch.from_numpy(first[key]),
            torch.from_numpy(repeated[key]),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    assert torch.equal(
        torch.from_numpy(first["t2v"]).argsort(dim=-1),
        torch.from_numpy(repeated["t2v"]).argsort(dim=-1),
    )
    assert torch.equal(
        torch.from_numpy(first["v2t"]).T.argsort(dim=-1),
        torch.from_numpy(repeated["v2t"]).T.argsort(dim=-1),
    )
    torch.testing.assert_close(
        torch.from_numpy(first["mean"]), torch.from_numpy(changed["mean"])
    )
    assert not torch.equal(
        torch.from_numpy(first["t2v"]), torch.from_numpy(changed["t2v"])
    )


def test_dsl_with_actual_rspr_top_r_is_rejected_immediately():
    args = SimpleNamespace(
        rspr_mode="stochastic",
        rspr_freeze_clip=False,
        rspr_freeze_dsa=False,
        rspr_sample_count=4,
        rspr_eval_sample_count=4,
        rspr_match_temperature=0.1,
        rspr_prob_temperature=0.1,
        rspr_rank_temperature=0.1,
        rspr_prior_std=0.1,
        rspr_pair_chunk_size=4,
        rspr_prob_weight=0.1,
        rspr_rank_weight=0.1,
        rspr_anchor_weight=0.1,
        rspr_rerank_weight=0.1,
        rspr_warmup_epochs=0.0,
        rspr_top_r=2,
        DSL=True,
    )

    with pytest.raises(ValueError, match="DSL.*RSPR Top-R"):
        validate_rspr_cli(args)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("deterministic_temperature", 0.0),
        ("deterministic_temperature", float("nan")),
        ("probabilistic_temperature", -1.0),
        ("probabilistic_temperature", float("inf")),
    ),
)
def test_rerank_rejects_nonpositive_or_nonfinite_temperatures(name, value):
    kwargs = {
        "top_r": 2,
        "deterministic_temperature": 1.0,
        "probabilistic_temperature": 1.0,
        "probabilistic_weight": 0.25,
        "pair_chunk_size": 3,
    }
    kwargs[name] = value

    with pytest.raises(ValueError, match=name):
        rerank_top_r(*_inputs(), _SpyMatcher(), **kwargs)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("rspr_det_temperature", 0.0),
        ("rspr_det_temperature", float("nan")),
        ("rspr_rerank_temperature", -1.0),
        ("rspr_rerank_temperature", float("inf")),
    ),
)
def test_cli_rejects_invalid_reranking_temperatures(name, value):
    args = SimpleNamespace(
        rspr_mode="off",
        rspr_freeze_clip=False,
        rspr_freeze_dsa=False,
        rspr_sample_count=4,
        rspr_eval_sample_count=4,
        rspr_match_temperature=0.1,
        rspr_prob_temperature=0.1,
        rspr_rank_temperature=0.1,
        rspr_prior_std=0.1,
        rspr_pair_chunk_size=4,
        rspr_prob_weight=0.1,
        rspr_rank_weight=0.1,
        rspr_anchor_weight=0.1,
        rspr_rerank_weight=0.1,
        rspr_warmup_epochs=0.0,
        rspr_top_r=2,
        rspr_det_temperature=1.0,
        rspr_rerank_temperature=1.0,
        DSL=False,
    )
    setattr(args, name, value)

    with pytest.raises(ValueError, match=name):
        validate_rspr_cli(args)


def test_multi_caption_metrics_reshape_each_direction_independently(monkeypatch):
    t2v = torch.arange(8, dtype=torch.float32).reshape(4, 2).numpy()
    v2t = (100 + torch.arange(8, dtype=torch.float32)).reshape(4, 2).numpy()
    seen = {}

    def fake_text_metrics(matrix):
        seen["t2v"] = matrix.copy()
        return {"direction": "t2v"}

    def fake_video_sim(matrix):
        seen["v2t"] = matrix.copy()
        return matrix.sum(axis=1)

    def fake_compute(matrix):
        seen["computed_v2t"] = matrix.copy()
        return {"direction": "v2t"}

    monkeypatch.setattr(
        main_task_retrieval,
        "tensor_text_to_video_metrics",
        fake_text_metrics,
    )
    monkeypatch.setattr(
        main_task_retrieval,
        "tensor_video_to_text_sim",
        fake_video_sim,
    )
    monkeypatch.setattr(main_task_retrieval, "compute_metrics", fake_compute)

    tv_metrics, vt_metrics = main_task_retrieval._compute_directional_metrics(
        t2v,
        v2t,
        cut_off_points=[2, 4],
    )

    assert tv_metrics == {"direction": "t2v"}
    assert vt_metrics == {"direction": "v2t"}
    assert seen["t2v"].shape == (2, 2, 2)
    assert seen["v2t"].shape == (2, 2, 2)
    assert (seen["t2v"] < 100).all()
    assert (seen["v2t"] >= 100).all()


def test_multi_caption_metrics_can_preserve_legacy_single_matrix(monkeypatch):
    t2v = torch.arange(8, dtype=torch.float32).reshape(4, 2).numpy()
    v2t = (100 + torch.arange(8, dtype=torch.float32)).reshape(4, 2).numpy()
    seen = {}

    monkeypatch.setattr(
        main_task_retrieval,
        "tensor_text_to_video_metrics",
        lambda _matrix: {},
    )

    def fake_video_sim(matrix):
        seen["v2t"] = matrix.copy()
        return matrix.sum(axis=1)

    monkeypatch.setattr(
        main_task_retrieval,
        "tensor_video_to_text_sim",
        fake_video_sim,
    )
    monkeypatch.setattr(main_task_retrieval, "compute_metrics", lambda _matrix: {})

    main_task_retrieval._compute_directional_metrics(
        t2v,
        v2t,
        cut_off_points=[2, 4],
        independent_directions=False,
    )

    assert (seen["v2t"] < 100).all()


def _complete_metric_scores(sparse_scores, mean_scores):
    return rspr_rerank.build_full_ranking_scores(
        torch.as_tensor(sparse_scores),
        torch.as_tensor(mean_scores),
    ).numpy()


def test_full_metric_ranking_counts_partial_top_r_misses_once_each():
    sparse = torch.tensor(
        [
            [5.0, -torch.inf, -torch.inf],
            [5.0, -torch.inf, -torch.inf],
            [5.0, -torch.inf, -torch.inf],
        ]
    )
    mean = torch.tensor(
        [
            [3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
        ]
    )

    metrics = compute_metrics(_complete_metric_scores(sparse, mean))

    assert metrics["R1"] == pytest.approx(100 / 3)
    assert metrics["cols"] == [0, 1, 2]


def test_full_metric_ranking_all_top_r_misses_return_zero_recall():
    sparse = torch.tensor(
        [
            [-torch.inf, 5.0, -torch.inf],
            [-torch.inf, -torch.inf, 5.0],
            [5.0, -torch.inf, -torch.inf],
        ]
    )
    mean = torch.tensor(
        [
            [3.0, 2.0, 1.0],
            [1.0, 3.0, 2.0],
            [2.0, 1.0, 3.0],
        ]
    )

    metrics = compute_metrics(_complete_metric_scores(sparse, mean))

    assert metrics["R1"] == 0.0
    assert metrics["R5"] == 100.0
    assert metrics["cols"] == [1, 1, 1]


def test_multi_caption_real_misses_count_but_reshape_padding_does_not():
    sparse_t2v = torch.tensor(
        [
            [5.0, -torch.inf],
            [-torch.inf, 5.0],
            [5.0, -torch.inf],
        ]
    )
    sparse_v2t = torch.tensor(
        [
            [5.0, 5.0],
            [-torch.inf, -torch.inf],
            [-torch.inf, -torch.inf],
        ]
    )
    mean = torch.tensor(
        [
            [2.0, 1.0],
            [2.0, 1.0],
            [2.0, 1.0],
        ]
    )
    t2v_metric, v2t_metric = main_task_retrieval._build_rspr_metric_matrices(
        sparse_t2v.numpy(),
        sparse_v2t.numpy(),
        mean.numpy(),
    )

    tv_metrics, vt_metrics = main_task_retrieval._compute_directional_metrics(
        t2v_metric,
        v2t_metric,
        cut_off_points=[2, 3],
    )

    assert tv_metrics["R1"] == pytest.approx(100 / 3)
    assert tv_metrics["R5"] == 100.0
    assert vt_metrics["R1"] == 50.0
    assert vt_metrics["R5"] == 100.0


def test_directional_metric_matrices_count_all_misses_in_both_directions():
    sparse_t2v = torch.tensor(
        [
            [-torch.inf, 5.0, -torch.inf],
            [-torch.inf, -torch.inf, 5.0],
            [5.0, -torch.inf, -torch.inf],
        ]
    )
    sparse_v2t = sparse_t2v.T.contiguous()
    mean = torch.tensor(
        [
            [3.0, 2.0, 1.0],
            [1.0, 3.0, 2.0],
            [2.0, 1.0, 3.0],
        ]
    )

    t2v_metric, v2t_metric = main_task_retrieval._build_rspr_metric_matrices(
        sparse_t2v.numpy(),
        sparse_v2t.numpy(),
        mean.numpy(),
    )
    tv_metrics, vt_metrics = main_task_retrieval._compute_directional_metrics(
        t2v_metric,
        v2t_metric,
    )

    assert tv_metrics["R1"] == 0.0
    assert vt_metrics["R1"] == 0.0
    assert tv_metrics["cols"] == [1, 1, 1]
    assert vt_metrics["cols"] == [1, 1, 1]


def test_full_metric_ranking_breaks_mean_and_final_ties_by_original_index():
    sparse = torch.tensor([[7.0, -torch.inf, 7.0, -torch.inf]])
    mean = torch.zeros(1, 4)

    metric_scores = torch.from_numpy(_complete_metric_scores(sparse, mean))

    assert metric_scores.argsort(dim=1, descending=True).tolist() == [
        [0, 2, 1, 3]
    ]


def test_fully_tied_metric_matrix_counts_each_query_exactly_once():
    tied = torch.zeros(3, 3)

    metrics = compute_metrics(_complete_metric_scores(tied, tied))

    assert metrics["cols"] == [0, 1, 2]
    assert metrics["R1"] == pytest.approx(100 / 3)


def test_mean_only_eval_counts_each_fully_tied_query_once(monkeypatch):
    tied = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ).numpy()
    retrieval_output = {
        "t2v": tied.copy(),
        "v2t": tied.copy(),
        "mean": tied.copy(),
        "recall": tied.copy(),
        "uncertainty": torch.full((3, 3), torch.nan).numpy(),
    }
    monkeypatch.setattr(
        main_task_retrieval,
        "_run_on_single_gpu",
        lambda *_args, **_kwargs: retrieval_output,
    )
    monkeypatch.setattr(
        main_task_retrieval,
        "logger",
        logging.getLogger("test.rspr.mean_only_ties"),
        raising=False,
    )
    model = nn.Linear(1, 1)

    class _EmptyLoader:
        dataset = SimpleNamespace(multi_sentence_per_video=False)

        def __iter__(self):
            return iter(())

    dataloader = _EmptyLoader()
    args = SimpleNamespace(
        rspr_mode="mean",
        rspr_top_r=0,
        DSL=False,
        local_rank=0,
        log_mus_scores=False,
    )

    metrics = main_task_retrieval.eval_epoch(
        args,
        model,
        dataloader,
        torch.device("cpu"),
        n_gpu=1,
    )

    assert metrics["t2v"]["R1"] == pytest.approx(200 / 3)
    assert metrics["v2t"]["R1"] == pytest.approx(200 / 3)
    assert metrics["t2v"]["cols"] == [0, 0, 2]
    assert metrics["v2t"]["cols"] == [0, 0, 2]
