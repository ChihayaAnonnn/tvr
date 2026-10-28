"""Regressions for the registered pilot's policy, verdict and panel contracts."""

from types import SimpleNamespace

import numpy as np
import pytest


def gallery(size=33):
    ids = [f"c{i:03}" for i in range(size)]
    return dict(
        version="gallery",
        manifest_sha256="a" * 64,
        canonical_ids=ids,
        aliases={c: [f"v{i:03}"] for i, c in enumerate(ids)},
    )


def rows(*, track="controlled", size=33):
    truth = np.zeros((2, size), dtype=np.int8)
    truth[:, 0] = 1
    mask = np.zeros_like(truth, dtype=bool)
    mask[0, :32] = True
    mask[1, 1:33] = True
    if track == "natural":
        mask[1] = mask[0]
    scores = np.full(truth.shape, -8.0)
    scores[:, 0] = 8
    return dict(
        query_ids=np.array(["q", "q"] if track == "controlled" else ["q", "r"]),
        component_ids=np.array(["component", "component"]),
        pair_logits=scores,
        relevance=truth,
        official_relevance=truth.copy(),
        candidate_mask=mask,
        presence_logits=np.array([8.0, -8.0] if track == "controlled" else [8.0, 8.0]),
    )


def group(track="controlled"):
    return dict(
        arm="A3",
        seed=0,
        split="val_development",
        track=track,
        mechanism="fixed",
        K=32,
        gallery="gallery",
        model_sha256="b" * 64,
        split_sha256="c" * 64,
    )


def load(tmp_path, data, descriptor=None, size=33):
    from semantic_retrieval.evaluation.pilot_analysis import _load_group, file_sha256

    path = tmp_path / "scores.npz"
    np.savez(path, **data)
    descriptor = dict(descriptor or group(), arrays=dict(path=str(path), sha256=file_sha256(path)))
    return _load_group(descriptor, dict(galleries={"gallery": gallery(size)}))


def test_natural_actions_use_the_feasible_controlled_development_policy(tmp_path):
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evaluation.pilot_analysis import _evaluate_group, _freeze

    provenance = tmp_path / "provenance"
    provenance.mkdir()
    context = SimpleNamespace(
        paths=SimpleNamespace(provenance=provenance),
        manifest={"git_sha": "1" * 40},
        array_store=ArtifactArrayStore(tmp_path / "arrays"),
    )
    bundle = dict(
        galleries={"locked": gallery()},
        relevance_space="benchmark_annotation",
        label_protocol={"protocol_id": "official"},
        preprocessing_sha256="d" * 64,
        policy_sha256="e" * 64,
    )
    controlled = dict(rows(), group=group(), gallery=gallery())
    natural = dict(rows(track="natural"), group=group("natural"), gallery=gallery())
    frozen = _freeze(context, bundle, [controlled, natural], [])
    assert set(frozen["policies"]) == {"A3_seed0_controlled_fixed_K32"}
    assert frozen["policies"]["A3_seed0_controlled_fixed_K32"]["status"] == "feasible"
    evaluated = _evaluate_group(context, bundle, frozen, natural, [], raw_rc=True)
    assert evaluated["answers"] is not None
    assert evaluated["answers"][:, 0].tolist() == [True, True]
    assert evaluated["answers"].sum() == 2


def test_policy_freeze_rejects_development_with_only_natural_panels(tmp_path):
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evaluation.pilot_analysis import _freeze

    provenance = tmp_path / "provenance"
    provenance.mkdir()
    context = SimpleNamespace(
        paths=SimpleNamespace(provenance=provenance),
        manifest={"git_sha": "1" * 40},
        array_store=ArtifactArrayStore(tmp_path / "arrays"),
    )
    bundle = dict(
        galleries={"locked": gallery()},
        relevance_space="benchmark_annotation",
        label_protocol={"protocol_id": "official"},
        preprocessing_sha256="d" * 64,
        policy_sha256="e" * 64,
    )
    natural = dict(rows(track="natural"), group=group("natural"), gallery=gallery())
    with pytest.raises(ValueError, match="controlled"):
        _freeze(context, bundle, [natural], [])


@pytest.mark.parametrize("primary_delta,expected", [(-0.08, "PASS-benchmark-pilot"), (0.1, "FAIL")])
def test_pilot_verdict_needs_pilot_gates_without_confirmatory_power(primary_delta, expected):
    from semantic_retrieval.evaluation.pilot_analysis import _hypotheses

    effects = []
    for hypothesis, names in (
        ("B-H1", ("absent_false_accept", "present_coverage")),
        ("B-H2", ("aurc", "top1_accuracy")),
    ):
        for comparator in ("pair_only", "strongest_scalar"):
            for index, metric in enumerate(names):
                value = primary_delta if index == 0 else 0.0
                effects.append(
                    dict(
                        hypothesis=hypothesis,
                        comparator=comparator,
                        metric=metric,
                        status="available",
                        point_delta=value,
                        ci_low=value - 0.005,
                        ci_high=value + 0.005,
                    )
                )
    gates = dict.fromkeys(("data", "four_gpu_qualification", "gradients", "stability", "invariance"), True)
    assert _hypotheses(effects, gates)[1] == expected
    gates["power"] = False
    assert _hypotheses(effects, gates)[1] == expected
    gates["stability"] = False
    assert _hypotheses(effects, gates)[1] == "inconclusive"


@pytest.mark.parametrize(
    "defect",
    [
        "zero_gt",
        "multiple_gt",
        "unknown_official",
        "underfilled",
        "unpaired",
        "same_event",
        "component_mismatch",
        "truth_mismatch",
    ],
)
def test_illegal_fixed_benchmark_panels_fail_closed(tmp_path, defect):
    data = rows()
    if defect == "zero_gt":
        data["relevance"][:] = 0
        data["official_relevance"][:] = 0
    elif defect == "multiple_gt":
        data["relevance"][:, 1] = 1
        data["official_relevance"][:, 1] = 1
    elif defect == "unknown_official":
        data["official_relevance"][:, 2] = -1
    elif defect == "underfilled":
        data["candidate_mask"][0, 1] = False
    elif defect == "unpaired":
        data["query_ids"] = np.array(["q", "r"])
    elif defect == "same_event":
        data["candidate_mask"][1] = data["candidate_mask"][0]
    elif defect == "component_mismatch":
        data["component_ids"] = np.array(["component", "other"])
    elif defect == "truth_mismatch":
        data["relevance"][1] = 0
        data["relevance"][1, 32] = 1
        data["official_relevance"] = data["relevance"].copy()
        data["candidate_mask"][1, 32] = False
        data["candidate_mask"][1, 0] = True
    with pytest.raises(ValueError):
        load(tmp_path, data)


def test_valid_fixed_controlled_pair_preserves_source_identity(tmp_path):
    result = load(tmp_path, rows())
    assert result["query_ids"].tolist() == ["q", "q"]
    assert result["candidate_mask"].sum(axis=1).tolist() == [32, 32]


def test_fixed_natural_cannot_duplicate_a_source_query(tmp_path):
    data = rows(track="natural")
    data["query_ids"] = np.array(["q", "q"])
    with pytest.raises(ValueError):
        load(tmp_path, data, group("natural"))


def test_small_natural_gallery_uses_minimum_of_k_and_gallery_size(tmp_path):
    data = rows(track="natural", size=2)
    data["candidate_mask"][:] = True
    assert load(tmp_path, data, group("natural"), size=2)["candidate_mask"].sum(axis=1).tolist() == [2, 2]


def test_static_expansion_can_recover_gt_in_both_controlled_copies(tmp_path):
    data = rows(size=65)
    data["initial_candidate_mask"] = data["candidate_mask"].copy()
    data["candidate_mask"][:, :64] = True
    descriptor = dict(group(), mechanism="static", K=64)
    result = load(tmp_path, data, descriptor, size=65)
    assert (result["candidate_mask"] & result["relevance"].astype(bool)).any(axis=1).tolist() == [True, True]
