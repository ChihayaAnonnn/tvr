import pytest
import torch
import torch.nn.functional as F

from prob_models.contextual_retrieval_risk import (
    CONTEXTUAL_FEATURE_SETS,
    ContextualRetrievalRiskSidecar,
    contextual_topk_features,
)


@pytest.mark.parametrize("feature_set", sorted(CONTEXTUAL_FEATURE_SETS))
def test_feature_ablation_masks_named_blocks_without_changing_capacity(feature_set):
    torch.manual_seed(7)
    full = ContextualRetrievalRiskSidecar(
        embed_dim=512, projection_dim=128, hidden_dim=128, topk=16,
        dropout=0.0, feature_set="full",
    )
    variant = ContextualRetrievalRiskSidecar(
        embed_dim=512, projection_dim=128, hidden_dim=128, topk=16,
        dropout=0.0, feature_set=feature_set,
    )
    variant.load_state_dict(full.state_dict())
    logits = torch.tensor([[4.0, 3.0, 1.0], [0.0, 2.0, 5.0]])
    text = torch.randn(2, 512)
    video = torch.randn(3, 512)

    full_output = full(logits, text, video, "t2v")
    variant_output = variant(logits, text, video, "t2v")
    full_features = full_output["features"]
    variant_features = variant_output["features"]

    assert sum(parameter.numel() for parameter in variant.parameters()) == 186_113
    assert variant_features.shape == full_features.shape == (2, 426)
    assert variant_output.get("feature_set") == feature_set
    assert variant_output.get("active_feature_blocks") == tuple(
        sorted(CONTEXTUAL_FEATURE_SETS[feature_set])
    )
    assert variant_output.get("feature_block_offsets") == {
        name: (feature_slice.start, feature_slice.stop)
        for name, feature_slice in variant.feature_slices.items()
    }
    for block, feature_slice in variant.feature_slices.items():
        if block in CONTEXTUAL_FEATURE_SETS[feature_set]:
            torch.testing.assert_close(
                variant_features[:, feature_slice], full_features[:, feature_slice]
            )
        else:
            torch.testing.assert_close(
                variant_features[:, feature_slice],
                torch.zeros_like(variant_features[:, feature_slice]),
            )


def test_topk_features_follow_direction_and_pad_to_registered_width():
    logits = torch.tensor([[4.0, 3.0, 1.0], [0.0, 2.0, 5.0]])
    text = F.normalize(torch.randn(2, 4), dim=-1)
    video = F.normalize(torch.randn(3, 4), dim=-1)

    t2v = contextual_topk_features(logits, text, video, "t2v", topk=4)
    v2t = contextual_topk_features(logits, video, text, "v2t", topk=4)

    assert t2v["top_indices"].shape == (2, 4)
    assert v2t["top_indices"].shape == (3, 4)
    assert t2v["top_valid"].sum(1).tolist() == [3, 3]
    assert v2t["top_valid"].sum(1).tolist() == [2, 2, 2]
    torch.testing.assert_close(t2v["top1_mean"], video[t2v["top_indices"][:, 0]])
    assert t2v["normalized_scores"].shape == (2, 4)
    assert t2v["score_gaps"].shape == (2, 4)
    assert torch.all(t2v["attention"][~t2v["top_valid"]] == 0.0)
    torch.testing.assert_close(t2v["attention"].sum(1), torch.ones(2))


def test_contextual_head_detaches_retrieval_inputs():
    logits = torch.randn(3, 4, requires_grad=True)
    text = torch.randn(3, 8, requires_grad=True)
    video = torch.randn(4, 8, requires_grad=True)
    head = ContextualRetrievalRiskSidecar(
        embed_dim=8, projection_dim=5, topk=2
    )

    head(logits, text, video, "t2v")["failure_logit"].sum().backward()

    assert logits.grad is None and text.grad is None and video.grad is None
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_contextual_head_uses_binding_projection_default():
    head = ContextualRetrievalRiskSidecar(embed_dim=4)

    assert head.text_projection.out_features == 128
    assert head.video_projection.out_features == 128


def test_topk_one_defines_zero_finite_entropy():
    features = contextual_topk_features(
        torch.tensor([[4.0, 1.0], [0.0, 2.0]]),
        torch.randn(2, 3),
        torch.randn(2, 3),
        "t2v",
        topk=1,
    )

    assert torch.isfinite(features["topk_entropy"]).all()
    torch.testing.assert_close(features["topk_entropy"], torch.zeros(2))


def test_head_uses_one_shared_network_and_direction_specific_projections():
    head = ContextualRetrievalRiskSidecar(
        embed_dim=4, projection_dim=3, hidden_dim=7, topk=2, dropout=0.0
    )
    logits = torch.randn(2, 3)
    text = torch.randn(2, 4)
    video = torch.randn(3, 4)

    t2v = head(logits, text, video, "t2v")
    v2t = head(logits, video, text, "v2t")

    assert head.network[0].in_features == 3 * 3 + 8 + 2 * 2 + 2
    assert t2v["failure_logit"].shape == (2,)
    assert v2t["failure_logit"].shape == (3,)
    assert t2v["features"].shape == (2, 3 * 3 + 8 + 2 * 2 + 2)
    assert v2t["features"].shape == (3, 3 * 3 + 8 + 2 * 2 + 2)
    torch.testing.assert_close(t2v["features"][:, -2:], torch.tensor([[1.0, 0.0]]).expand(2, -1))
    torch.testing.assert_close(v2t["features"][:, -2:], torch.tensor([[0.0, 1.0]]).expand(3, -1))
    torch.testing.assert_close(t2v["failure_risk"], t2v["failure_logit"].sigmoid())
    assert t2v["topk_entropy"].shape == (2,)


@pytest.mark.parametrize(
    ("logits", "query", "candidate", "direction", "topk", "temperature"),
    [
        (torch.ones(2), torch.ones(2, 3), torch.ones(2, 3), "t2v", 1, 1.0),
        (torch.ones(2, 2), torch.ones(3, 3), torch.ones(2, 3), "t2v", 1, 1.0),
        (torch.ones(2, 2), torch.ones(2, 3), torch.ones(2, 4), "t2v", 1, 1.0),
        (torch.ones(2, 2), torch.ones(2, 3), torch.ones(2, 3), "bad", 1, 1.0),
        (torch.ones(2, 2), torch.ones(2, 3), torch.ones(2, 3), "t2v", 0, 1.0),
        (torch.ones(2, 2), torch.ones(2, 3), torch.ones(2, 3), "t2v", 1, 0.0),
        (torch.tensor([[1.0, float("nan")], [0.0, 1.0]]), torch.ones(2, 3), torch.ones(2, 3), "t2v", 1, 1.0),
    ],
)
def test_contextual_features_reject_invalid_inputs(
    logits, query, candidate, direction, topk, temperature
):
    with pytest.raises(ValueError):
        contextual_topk_features(logits, query, candidate, direction, topk, temperature)
