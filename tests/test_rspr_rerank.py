from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import main_task_retrieval
from main_task_retrieval import _run_on_single_gpu, validate_rspr_cli
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


def _rerank(matcher, *, top_r=2, pair_chunk_size=3):
    return rerank_top_r(
        *_inputs(),
        matcher,
        top_r=top_r,
        deterministic_temperature=2.0,
        probabilistic_temperature=0.5,
        probabilistic_weight=0.25,
        pair_chunk_size=pair_chunk_size,
    )


def test_top_r_scores_only_aligned_candidate_pairs_in_bounded_chunks():
    matcher = _SpyMatcher()

    _rerank(matcher, top_r=2, pair_chunk_size=3)

    assert matcher.forward_calls == 0
    assert sum(matcher.pair_batch_sizes) <= 3 * 2 + 5 * 2
    assert max(matcher.pair_batch_sizes) <= 3


def test_top_r_candidate_masks_are_direction_independent():
    matcher = _SpyMatcher()
    output = _rerank(matcher, top_r=1)

    t2v_candidates = output.mean_logits.topk(1, dim=1).indices
    expected_t2v_mask = torch.zeros(3, 5, dtype=torch.bool)
    expected_t2v_mask.scatter_(1, t2v_candidates, True)
    v2t_candidates = output.mean_logits.topk(1, dim=0).indices
    expected_v2t_mask = torch.zeros(3, 5, dtype=torch.bool)
    expected_v2t_mask.scatter_(0, v2t_candidates, True)

    assert torch.equal(torch.isfinite(output.text_to_video_logits), expected_t2v_mask)
    assert torch.equal(torch.isfinite(output.video_to_text_logits), expected_v2t_mask)
    assert not torch.equal(expected_t2v_mask, expected_v2t_mask)
    assert torch.equal(
        torch.isfinite(output.pair_uncertainty),
        expected_t2v_mask | expected_v2t_mask,
    )


def test_top_r_uses_probability_formula_and_mean_recall_candidates():
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
    expected_mean = F.normalize(text_mean, dim=-1) @ F.normalize(video_mean, dim=-1).T
    torch.testing.assert_close(output.mean_logits, expected_mean)

    row = torch.arange(3).unsqueeze(1).expand(-1, 2)
    col = expected_mean.topk(2, dim=1).indices
    probability = (text_samples[row] * video_samples[col]).sum(dim=-1).mean(dim=-1)
    expected = deterministic_logits[row, col] / 2.0 + 0.25 * probability / 0.5
    torch.testing.assert_close(output.text_to_video_logits[row, col], expected)


def test_top_r_is_elementwise_deterministic_for_fixed_inputs():
    matcher = _SpyMatcher()

    first = _rerank(matcher)
    second = _rerank(matcher)

    for name in (
        "text_to_video_logits",
        "video_to_text_logits",
        "mean_logits",
        "pair_uncertainty",
    ):
        torch.testing.assert_close(
            getattr(first, name), getattr(second, name), equal_nan=True
        )


def test_zero_top_r_returns_mean_only_without_calling_matcher():
    matcher = _SpyMatcher()

    output = _rerank(matcher, top_r=0)

    assert matcher.pair_batch_sizes == []
    assert matcher.forward_calls == 0
    torch.testing.assert_close(output.text_to_video_logits, output.mean_logits)
    torch.testing.assert_close(output.video_to_text_logits, output.mean_logits)
    assert torch.isnan(output.pair_uncertainty).all()


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
        mean = text_token[:, 0]
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

    assert set(output) == {"t2v", "v2t", "mean", "uncertainty"}
    assert all(matrix.shape == (4, 5) for matrix in output.values())
    assert model.text_distribution_batches == [2, 2]
    assert model.video_distribution_batches == [2, 2, 1]
    assert (torch.from_numpy(output["v2t"]).isfinite().sum(dim=0) == 2).all()


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
