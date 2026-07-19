import pytest
import torch
import torch.nn.functional as F

from modules.stochastic_prototype_ranking import (
    BidirectionalSoftPrototypeMatcher,
    StochasticRankLoss,
)


def test_soft_matcher_supports_rectangular_batches():
    matcher = BidirectionalSoftPrototypeMatcher(temperature=0.07)

    output = matcher(torch.randn(2, 4, 8), torch.randn(3, 4, 8))

    assert output.logits.shape == (2, 3)
    assert output.pair_uncertainty.shape == (2, 3)
    assert output.text_prototype_scores.shape == (2, 3, 4)
    assert output.video_prototype_scores.shape == (2, 3, 4)
    assert output.stochastic_pair_scores.shape == (2, 3, 4)


def test_soft_matcher_supports_single_prototype():
    matcher = BidirectionalSoftPrototypeMatcher(temperature=0.07)

    output = matcher(torch.randn(2, 1, 8), torch.randn(3, 1, 8))

    assert output.logits.shape == (2, 3)
    assert output.stochastic_pair_scores.shape == (2, 3, 1)


def test_soft_matcher_is_invariant_to_duplicate_identical_prototypes():
    text = F.normalize(torch.randn(2, 1, 8), dim=-1)
    video = F.normalize(torch.randn(3, 1, 8), dim=-1)
    matcher = BidirectionalSoftPrototypeMatcher(temperature=0.07)

    one = matcher(text, video).logits
    four = matcher(text.expand(-1, 4, -1), video.expand(-1, 4, -1)).logits

    torch.testing.assert_close(one, four)


def test_soft_matcher_outputs_are_finite_for_normalized_inputs_at_small_temperature():
    text = F.normalize(torch.randn(2, 4, 8), dim=-1)
    video = F.normalize(torch.randn(3, 4, 8), dim=-1)
    matcher = BidirectionalSoftPrototypeMatcher(temperature=1e-6)

    output = matcher(text, video)

    for value in (
        output.logits,
        output.pair_uncertainty,
        output.text_prototype_scores,
        output.video_prototype_scores,
        output.stochastic_pair_scores,
    ):
        assert torch.isfinite(value).all()


def test_soft_matcher_reports_nonnegative_and_zero_uncertainty_for_identical_prototypes():
    prototype_text = F.normalize(torch.randn(2, 1, 8), dim=-1)
    prototype_video = F.normalize(torch.randn(3, 1, 8), dim=-1)
    matcher = BidirectionalSoftPrototypeMatcher(temperature=0.07)

    output = matcher(
        prototype_text.expand(-1, 4, -1),
        prototype_video.expand(-1, 4, -1),
    )

    assert torch.all(output.pair_uncertainty >= 0)
    torch.testing.assert_close(output.pair_uncertainty, torch.zeros_like(output.pair_uncertainty))


def test_soft_matcher_requires_equal_numbers_of_text_and_video_prototypes():
    matcher = BidirectionalSoftPrototypeMatcher(temperature=0.07)

    with pytest.raises(ValueError, match="same number of prototypes"):
        matcher(torch.randn(2, 3, 8), torch.randn(3, 4, 8))


def test_score_pairs_matches_diagonal_of_full_matcher():
    text = torch.randn(3, 4, 8)
    video = torch.randn(3, 4, 8)
    matcher = BidirectionalSoftPrototypeMatcher(temperature=0.07)

    paired = matcher.score_pairs(text, video)
    full = matcher(text, video)

    torch.testing.assert_close(paired.logits, full.logits.diagonal())
    torch.testing.assert_close(
        paired.stochastic_pair_scores,
        full.stochastic_pair_scores[torch.arange(text.size(0)), torch.arange(text.size(0))],
    )


@pytest.mark.parametrize("hard_max", [False, True])
def test_soft_and_hard_max_modes_backpropagate(hard_max):
    text = torch.randn(2, 4, 8, requires_grad=True)
    video = torch.randn(3, 4, 8, requires_grad=True)
    matcher = BidirectionalSoftPrototypeMatcher(temperature=0.07, hard_max=hard_max)

    output = matcher(text, video)
    output.logits.sum().backward()

    assert text.grad is not None
    assert video.grad is not None
    assert torch.isfinite(text.grad).all()
    assert torch.isfinite(video.grad).all()


def test_rank_loss_excludes_every_same_group_candidate_and_matches_formula():
    scores = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0], [2.0, 4.0], [0.0, 2.0]],
            [[4.0, 2.0], [6.0, 8.0], [1.0, 3.0], [2.0, 0.0]],
            [[3.0, 1.0], [0.0, 2.0], [7.0, 5.0], [4.0, 6.0]],
            [[2.0, 4.0], [3.0, 1.0], [5.0, 3.0], [8.0, 6.0]],
        ]
    )
    group_ids = torch.tensor([10, 10, 20, 30])
    mining_logits = torch.tensor(
        [[0.0, 100.0, 3.0, 2.0], [100.0, 0.0, 3.0, 2.0], [3.0, 2.0, 0.0, 1.0], [2.0, 1.0, 3.0, 0.0]]
    )
    rank_loss = StochasticRankLoss(temperature=0.5, hard_negative_count=1, margin=0.25)

    output = rank_loss(scores, group_ids, mining_logits)

    assert not torch.isin(output.negative_indices[0], torch.tensor([0, 1])).any()
    expected_indices = torch.tensor([[2], [2], [0], [2]])
    torch.testing.assert_close(output.negative_indices, expected_indices)
    positive_mask = group_ids[:, None].eq(group_ids[None, :])
    positive_scores = (scores * positive_mask.unsqueeze(-1)).sum(dim=1) / positive_mask.sum(dim=1, keepdim=True)
    gathered = scores.gather(
        1,
        expected_indices.unsqueeze(-1).expand(-1, -1, scores.size(-1)),
    )
    difference = gathered - positive_scores.unsqueeze(1)
    expected = F.softplus((difference + rank_loss.margin) / rank_loss.temperature).mean()
    expected_inversion = torch.sigmoid(difference / rank_loss.temperature).mean(dim=-1)
    torch.testing.assert_close(output.loss, expected)
    torch.testing.assert_close(output.inversion_probability, expected_inversion)
    assert torch.all((0 <= output.inversion_probability) & (output.inversion_probability <= 1))


def test_rank_loss_clips_hard_negative_count_to_available_candidates():
    scores = torch.randn(3, 3, 2)
    group_ids = torch.tensor([0, 1, 2])
    mining_logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, 2.0], [1.0, 3.0, 0.0]])

    output = StochasticRankLoss(hard_negative_count=8)(scores, group_ids, mining_logits)

    assert output.negative_indices.shape == (3, 2)
    assert torch.all(output.negative_indices != torch.arange(3).unsqueeze(1))


def test_rank_loss_bidirectional_uses_transposed_mining_logits_independently():
    scores = torch.arange(9, dtype=torch.float32).reshape(3, 3, 1)
    group_ids = torch.tensor([0, 1, 2])
    mining_logits = torch.tensor([[0.0, 0.0, 1.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rank_loss = StochasticRankLoss(hard_negative_count=1)

    loss, text_to_video, video_to_text = rank_loss.bidirectional(scores, group_ids, mining_logits)

    torch.testing.assert_close(text_to_video.negative_indices, torch.tensor([[2], [0], [1]]))
    torch.testing.assert_close(video_to_text.negative_indices, torch.tensor([[1], [2], [0]]))
    torch.testing.assert_close(loss, 0.5 * (text_to_video.loss + video_to_text.loss))


def test_rank_loss_backpropagates_only_through_stochastic_scores():
    scores = torch.randn(3, 3, 2, requires_grad=True)
    group_ids = torch.tensor([0, 1, 2])
    mining_logits = torch.tensor(
        [[0.0, 2.0, 1.0], [3.0, 0.0, 2.0], [1.0, 3.0, 0.0]],
        requires_grad=True,
    )

    output = StochasticRankLoss(hard_negative_count=1)(scores, group_ids, mining_logits)
    output.loss.backward()

    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert mining_logits.grad is None


@pytest.mark.parametrize(
    ("scores", "group_ids", "mining_logits", "message"),
    [
        (torch.randn(2, 3, 1), torch.tensor([0, 1]), torch.randn(2, 2), "square"),
        (torch.empty(2, 2, 0), torch.tensor([0, 1]), torch.randn(2, 2), "nonempty"),
        (torch.randn(2, 2, 1), torch.tensor([0.0, 1.0]), torch.randn(2, 2), "integer"),
        (torch.randn(2, 2, 1), torch.tensor([True, False]), torch.randn(2, 2), "integer"),
        (torch.randn(2, 2, 1), torch.tensor([0, 1, 2]), torch.randn(2, 2), "shape"),
        (torch.randn(2, 2, 1), torch.tensor([0, 1]), torch.randn(2, 3), "shape"),
    ],
)
def test_rank_loss_validates_input_contracts(scores, group_ids, mining_logits, message):
    with pytest.raises(ValueError, match=message):
        StochasticRankLoss()(scores, group_ids, mining_logits)


def test_rank_loss_requires_a_negative_for_every_query():
    scores = torch.randn(2, 2, 1)
    group_ids = torch.tensor([0, 0])

    with pytest.raises(ValueError, match="valid negative"):
        StochasticRankLoss()(scores, group_ids, torch.randn(2, 2))


def test_rank_loss_validates_constructor_and_matching_score_dtypes():
    with pytest.raises(ValueError, match="temperature"):
        StochasticRankLoss(temperature=0)
    with pytest.raises(ValueError, match="hard_negative_count"):
        StochasticRankLoss(hard_negative_count=0)
    with pytest.raises(ValueError, match="margin"):
        StochasticRankLoss(margin=float("inf"))
    with pytest.raises(ValueError, match="same dtype"):
        StochasticRankLoss()(
            torch.randn(2, 2, 1, dtype=torch.float32),
            torch.tensor([0, 1]),
            torch.randn(2, 2, dtype=torch.float64),
        )
