from pathlib import Path

import numpy as np
import pytest


def test_tied_rc_is_order_invariant_and_right_endpoint_integrated():
    from semantic_retrieval.evaluation.pilot_analysis import risk_curve

    curve = risk_curve([0.9, 0.9, 0.2], [0, 1, 1])
    assert curve["coverage"].tolist() == [2 / 3, 1.0]
    assert curve["risk"].tolist() == [0.5, 2 / 3]
    assert curve["aurc"] == pytest.approx(5 / 9)
    assert risk_curve([0.9, 0.9, 0.2], [1, 0, 1])["aurc"] == curve["aurc"]


def test_official_alias_expansion_preserves_official_gt_identity():
    from semantic_retrieval.evaluation.pilot_analysis import official_ranking

    gallery = {"canonical_ids": ["a", "b"], "aliases": {"a": ["v0", "v1"], "b": ["v2"]}}
    result = official_ranking(np.array([[3.0, 2.0]]), np.array([[0, 1, 0]]), gallery)
    assert result["R1"]["value"] == 0
    assert result["MRR"]["value"] == 0.5
    assert result["mAP"]["value"] == 0.5
    assert result["MeanR"]["value"] == 2


def test_empty_returns_and_absent_rates_have_explicit_denominators():
    from semantic_retrieval.evaluation.pilot_analysis import policy_metrics

    truth = np.array([[1, 0], [1, 0]], bool)
    candidates = np.array([[1, 1], [0, 1]], bool)
    result = policy_metrics(np.zeros_like(truth), truth, candidates)
    assert result["set_micro_precision"] == {"numerator": 0, "denominator": 0, "value": None}
    assert result["present_coverage"]["denominator"] == 1
    assert result["absent_false_accept"]["denominator"] == 1
    assert result["macro_f1"]["value"] == 0
    assert result["clarify_rate"]["value"] == 0


def test_frozen_grid_cannot_select_abstain_everything_as_feasible():
    from semantic_retrieval.evaluation.pilot_analysis import select_operating_point

    scores = np.zeros((2, 2))
    truth = np.array([[1, 0], [1, 0]], bool)
    candidates = np.array([[1, 1], [0, 1]], bool)
    result = select_operating_point(scores, truth, candidates, presence_logits=np.zeros(2))
    assert result["status"] == "infeasible"
    assert result["operating_point"] is None


def identity(track="natural"):
    return dict(
        model_sha256="a" * 64,
        preprocessing_sha256="b" * 64,
        policy_sha256="c" * 64,
        label_protocol={"name": "official", "sha256": "d" * 64},
        relevance_space="benchmark_annotation",
        fit_gallery_version="cal",
        target_gallery_version="locked",
        fit_gallery_manifest_sha256="e" * 64,
        target_gallery_manifest_sha256="f" * 64,
        split_sha256="1" * 64,
        track=track,
        candidate_mechanism="fixed",
        k_weights={str(k): 1 / 6 for k in (4, 8, 16, 32, 64, 128)},
    )


def test_calibration_provenance_rejects_track_mix_and_unavailable_classes():
    from semantic_retrieval.evaluation.pilot_analysis import fit_pilot_calibration, map_threshold

    rows = []
    for k in (4, 8, 16, 32, 64, 128):
        rows.append(
            dict(
                K=k,
                query_ids=["q0", "q1"],
                canonical_ids=["a", "b"],
                pair_logits=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
                pair_labels=np.array([[0, 1], [0, 1]]),
                candidate_mask=np.ones((2, 2), bool),
                presence_logits=np.array([-1.0, 1.0]),
                presence_labels=np.array([0, 1]),
            )
        )
    state = fit_pilot_calibration(rows, identity(), split="val_calibration")
    assert state["pair"]["known_count"] == 4  # unique query-canonical, not 24 copies across K
    assert state["presence"]["status"] == "available"
    x = np.array([-5.0, -0.1, 0.1, 5.0])
    from scipy.special import expit

    t = map_threshold(0.6, state["pair"])
    assert np.array_equal(expit(x) >= 0.6, expit(x / state["pair"]["temperature"] + state["pair"]["bias"]) >= t)
    with pytest.raises(ValueError, match="track"):
        from semantic_retrieval.evaluation.pilot_analysis import calibrated_probabilities

        calibrated_probabilities(x, state, "pair", track="controlled", target_gallery_manifest_sha256="f" * 64)
    for row in rows:
        row["presence_labels"] = np.ones(2)
    assert fit_pilot_calibration(rows, identity(), split="val_calibration")["presence"]["status"] == "unavailable"
    with pytest.raises(ValueError, match="distinct"):
        wrong = identity()
        wrong["target_gallery_manifest_sha256"] = wrong["fit_gallery_manifest_sha256"]
        fit_pilot_calibration(rows, wrong, split="val_calibration")


def test_cluster_bootstrap_recomputes_each_seed_and_keeps_all_panel_copies_together():
    from semantic_retrieval.evaluation.pilot_analysis import cluster_bootstrap

    rows = [
        dict(
            component_ids=np.array(["a", "a", "b"]),
            query_ids=np.array(["q1", "q1", "q2"]),
            first=np.array([1.0, 1.0, 0.0]),
            second=np.array([0.0, 0.0, 0.0]),
        )
        for _ in range(3)
    ]
    result, samples = cluster_bootstrap(
        rows, lambda r, i: float(np.mean(r["first"][i]) - np.mean(r["second"][i])), samples=100, confidence=0.975
    )
    assert result["point_delta"] == pytest.approx(2 / 3)
    assert result["component_count"] == 2
    assert result["seed_values"] == pytest.approx([2 / 3] * 3)
    assert set(np.round(samples, 6)) <= {0.0, 0.666667, 1.0}
    assert result["confidence"] == 0.975


def test_binary_ap_is_tie_grouped_and_missing_class_has_null_auroc():
    from semantic_retrieval.evaluation.pilot_analysis import probability_metrics

    result = probability_metrics([0, 1], [0.5, 0.5])
    assert result["ap"] == 0.5
    assert probability_metrics([1, 1], [0.2, 0.8])["auroc"] is None
    assert probability_metrics([], [])["brier"] is None


def test_score_export_and_freeze_locked_pipeline_are_hash_bound(tmp_path):
    from types import SimpleNamespace

    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evaluation.pilot_analysis import file_sha256, run_pilot_analysis, write_pilot_rows

    context = SimpleNamespace(
        paths=SimpleNamespace(
            root=tmp_path, arrays=tmp_path / "arrays", plots=tmp_path / "plots", provenance=tmp_path / "provenance"
        ),
        manifest={"git_sha": "1" * 40},
        array_store=ArtifactArrayStore(tmp_path / "arrays"),
    )
    for d in ("plots", "provenance"):
        (tmp_path / d).mkdir()
    galleries = {}
    for tag, digest in [("dev", "2"), ("cal", "3"), ("locked", "4")]:
        galleries[tag] = dict(
            version=tag,
            manifest_sha256=digest * 64,
            canonical_ids=["a", "b"] + [f"z{i:03d}" for i in range(2, 129)],
            aliases={c: [f"v{i}"] for i, c in enumerate(["a", "b"] + [f"z{j:03d}" for j in range(2, 129)])},
        )
    truth = np.pad(np.array([[1, 0], [0, 1], [1, 0], [0, 1]]), ((0, 0), (0, 127)))
    score = np.pad(
        np.array([[2.0, -2.0], [-2.0, 2.0], [-2.0, -2.0], [-2.0, -2.0]]), ((0, 0), (0, 127)), constant_values=-100
    )

    def observations(track, k, tag):
        positions = np.arange(4) if track == "natural" else np.repeat(np.arange(4), 2)
        mask = np.zeros((len(positions), 129), bool)
        events = []
        for ri, qi in enumerate(positions):
            gt = int(np.flatnonzero(truth[qi])[0])
            present = qi < 2 if track == "natural" else ri % 2 == 0
            candidates = (
                ([gt] + [j for j in range(129) if j != gt])[:k] if present else [j for j in range(129) if j != gt][:k]
            )
            mask[ri, candidates] = True
            events.append(present)
        return dict(
            query_ids=[tag + str(i) for i in positions],
            component_ids=[tag + "c" + str(i) for i in positions],
            pair_logits=score[positions],
            relevance=truth[positions],
            official_relevance=truth[positions],
            candidate_mask=mask,
            presence_logits=np.where(events, 2.0, -2.0),
        )

    groups = []
    for split, gallery_tag in [("val_development", "dev"), ("val_calibration", "cal")]:
        for arm in ("A1", "A3"):
            for track in ("natural", "controlled"):
                for k in (4, 8, 16, 32, 64, 128) if split == "val_calibration" else (32,):
                    path = tmp_path / f"{split}-{arm}-{track}-{k}.npz"
                    ref = write_pilot_rows(path, **observations(track, k, gallery_tag))
                    groups.append(
                        dict(
                            arm=arm,
                            seed=0,
                            split=split,
                            track=track,
                            mechanism="fixed",
                            K=k,
                            gallery=gallery_tag,
                            model_sha256=("a" if arm == "A1" else "b") * 64,
                            split_sha256=("5" if split == "val_development" else "6") * 64,
                            arrays=ref,
                        )
                    )
    bundle = dict(
        schema_version=1,
        relevance_space="benchmark_annotation",
        label_protocol={"name": "official"},
        preprocessing_sha256="7" * 64,
        policy_sha256="8" * 64,
        galleries=galleries,
        groups=groups,
    )
    frozen = run_pilot_analysis(context, bundle, mode="freeze", bootstrap_samples=20)
    frozen_path = tmp_path / "provenance" / "pilot_frozen.json"
    assert frozen_path.is_file()
    assert frozen["strongest_scalar"]["B-H2"] in ("top_score", "margin", "negative_entropy", "energy")
    locked_groups = []
    for group in groups:
        if group["split"] == "val_development":
            ref = write_pilot_rows(
                tmp_path / f"locked-{group['arm']}-{group['track']}.npz", **observations(group["track"], 32, "lq")
            )
            locked_groups.append(dict(group, split="val_locked", gallery="locked", split_sha256="9" * 64, arrays=ref))
    locked_bundle = dict(
        bundle, groups=locked_groups, frozen={"path": str(frozen_path), "sha256": file_sha256(frozen_path)}
    )
    result = run_pilot_analysis(context, locked_bundle, mode="locked", bootstrap_samples=20)
    assert result["verdict"] == "inconclusive"  # missing three seeds/gates; never fake scientific PASS
    assert result["array_references"]
    assert (tmp_path / "plots" / "rc.png").is_file()
    assert result["groups"][0]["official_ranking"]["MRR"]["denominator"] == 4
    assert result["groups"][0]["full_gallery_aurc"] == pytest.approx(0.125)
    broken = dict(locked_bundle, policy_sha256="0" * 64)
    with pytest.raises(ValueError, match="frozen"):
        run_pilot_analysis(context, broken, mode="locked", bootstrap_samples=20)
    path = Path(groups[0]["arrays"]["path"])
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA"):
        run_pilot_analysis(context, bundle, mode="freeze", bootstrap_samples=20)


def test_connected_model_score_export_preserves_query_source_and_official_gt(model, inputs, tmp_path):
    from types import SimpleNamespace

    from semantic_retrieval.evaluation.pilot_analysis import export_model_scores
    from semantic_retrieval.types import QueryInputs

    # Existing two-query/three-video production fixture exercises actual matcher/presence consumers.
    n = inputs.input_ids.shape[0]
    m = inputs.video.shape[0]
    gallery = {
        "version": "dev",
        "manifest_sha256": "2" * 64,
        "canonical_ids": [f"c{i}" for i in range(m)],
        "aliases": {f"c{i}": [f"v{i}"] for i in range(m)},
    }
    relations = np.zeros((n, m), int)
    relations[:, 0] = 1
    context = SimpleNamespace(
        paths=SimpleNamespace(root=tmp_path, arrays=tmp_path / "arrays", provenance=tmp_path / "provenance"),
        manifest={"git_sha": "1" * 40},
    )
    context.paths.arrays.mkdir()
    context.paths.provenance.mkdir()
    panels = [
        dict(
            track="natural", K=4, query_ids=np.array([f"q{i}" for i in range(n)]), candidate_mask=np.ones((n, m), bool)
        )
    ]
    fragment = export_model_scores(
        context,
        model,
        QueryInputs(inputs.input_ids, inputs.text_mask),
        inputs.video,
        inputs.frame_mask,
        query_ids=[f"q{i}" for i in range(n)],
        component_ids=[f"source{i}" for i in range(n)],
        canonical_relevance=relations,
        official_relevance=relations,
        gallery=gallery,
        panels=panels,
        arm="A3",
        seed=0,
        split="val_development",
        split_sha256="3" * 64,
        label_protocol={"name": "official"},
        preprocessing_sha256="4" * 64,
        policy_sha256="5" * 64,
    )
    assert fragment["groups"][0]["K"] == 4
    with np.load(fragment["groups"][0]["arrays"]["path"]) as scores:
        assert scores["pair_logits"].shape == (n, m)
        assert scores["presence_logits"].shape == (n,)
        assert scores["full_presence_logits"].shape == (n,)
        assert scores["component_ids"].tolist() == [f"source{i}" for i in range(n)]
        assert np.array_equal(scores["official_relevance"], relations)
    assert (tmp_path / "provenance" / "pilot_score_fragment.json").is_file()


def test_calibration_dedup_rejects_conflicting_scores_and_locked_component_leak(tmp_path):
    from semantic_retrieval.evaluation.pilot_analysis import fit_pilot_calibration

    rows = []
    for k in (4, 8, 16, 32, 64, 128):
        rows.append(
            dict(
                K=k,
                query_ids=["q"],
                canonical_ids=["a", "b"],
                pair_logits=np.array([[1.0, -1.0]]),
                pair_labels=np.array([[1, 0]]),
                candidate_mask=np.ones((1, 2), bool),
                presence_logits=np.array([0.0]),
                presence_labels=np.array([1]),
            )
        )
    rows[-1]["pair_logits"][0, 0] = 1.1
    with pytest.raises(ValueError, match="relationship changed"):
        fit_pilot_calibration(rows, identity(), split="val_calibration")


def test_top1_policy_uses_canonical_identity_under_permutation():
    from semantic_retrieval.evaluation.pilot_analysis import apply_operating_point

    point = dict(pair_threshold=0.5, accept_threshold=0.0, accept_kind="pair_only")
    a = apply_operating_point(
        np.array([[1.0, 1.0]]), np.array([[1, 1]], bool), point, top1=True, canonical_ids=["b", "a"]
    )
    assert a.tolist() == [[False, True]]


def test_streamed_gallery_export_never_requires_full_raw_video_tensor(model, inputs, tmp_path):
    from types import SimpleNamespace

    from semantic_retrieval.evaluation.pilot_analysis import export_model_scores
    from semantic_retrieval.types import QueryInputs

    n, m = inputs.input_ids.shape[0], inputs.video.shape[0]
    gallery = {
        "version": "dev",
        "manifest_sha256": "2" * 64,
        "canonical_ids": [f"c{i}" for i in range(m)],
        "aliases": {f"c{i}": [f"v{i}"] for i in range(m)},
    }
    truth = np.zeros((n, m), int)
    truth[:, 0] = 1
    context = SimpleNamespace(
        paths=SimpleNamespace(root=tmp_path, arrays=tmp_path / "arrays", provenance=tmp_path / "provenance"),
        manifest={"git_sha": "1" * 40},
    )
    context.paths.arrays.mkdir()
    context.paths.provenance.mkdir()
    streamed = ((inputs.video[j : j + 1], inputs.frame_mask[j : j + 1]) for j in range(m))
    fragment = export_model_scores(
        context,
        model,
        QueryInputs(inputs.input_ids, inputs.text_mask),
        streamed,
        None,
        query_ids=[f"q{i}" for i in range(n)],
        component_ids=[f"c{i}" for i in range(n)],
        canonical_relevance=truth,
        official_relevance=truth,
        gallery=gallery,
        panels=[
            dict(track="natural", K=4, query_ids=[f"q{i}" for i in range(n)], candidate_mask=np.ones((n, m), bool))
        ],
        arm="A3",
        seed=0,
        split="val_development",
        split_sha256="3" * 64,
        label_protocol={"name": "official"},
        preprocessing_sha256="4" * 64,
        policy_sha256="5" * 64,
    )
    assert len(fragment["groups"]) == 1


def test_calibrated_action_comparison_preserves_raw_actions_when_sigmoid_saturates():
    from semantic_retrieval.evaluation.pilot_analysis import apply_calibrated_operating_point, apply_operating_point

    point = dict(pair_threshold=0.9, accept_threshold=0.9, accept_kind="presence")
    fitted = dict(status="available", temperature=0.01, bias=0.0)
    state = dict(pair=fitted, presence=fitted)
    scores = np.array([[2.0, 3.0]])
    candidates = np.ones_like(scores, bool)
    presence = np.array([3.0])
    expected = apply_operating_point(scores, candidates, point, presence_logits=presence)
    assert expected.tolist() == [[False, True]]
    assert np.array_equal(
        expected, apply_calibrated_operating_point(scores, candidates, point, state, presence_logits=presence)
    )


def test_three_seed_paired_effect_uses_source_clusters_and_recomputes_false_accept(tmp_path):
    from types import SimpleNamespace

    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evaluation.pilot_analysis import _effect

    context = SimpleNamespace(array_store=ArtifactArrayStore(tmp_path / "arrays"))
    truth = np.tile([1, 0], (8, 1)).astype(bool)
    candidates = np.tile([1, 1], (8, 1)).astype(bool)
    candidates[4:, 0] = False
    evaluated = []
    for seed in (0, 1, 2):
        for arm in ("A1", "A3"):
            answers = np.zeros_like(truth)
            answers[:4, 0] = True
            if arm == "A1":
                answers[4:, 1] = True
            row = dict(
                group=dict(arm=arm, seed=seed, track="controlled", mechanism="fixed", K=32),
                query_ids=np.array([f"q{i}" for i in range(8)]),
                component_ids=np.array([f"c{i}" for i in range(8)]),
                relevance=truth,
                candidate_mask=candidates,
            )
            evaluated.append(dict(row=row, answers=answers, scalar_answers={}))
    result = _effect(
        context,
        evaluated,
        "A3",
        "A1",
        track="controlled",
        metric="absent_false_accept",
        samples=200,
        confidence=0.975,
        references=[],
    )
    assert result["point_delta"] == -1
    assert result["ci_low"] == result["ci_high"] == -1
    assert result["component_count"] == 8
    assert result["seed_values"] == [-1.0, -1.0, -1.0]


def test_direct_c1_relation_table_expands_only_official_gt():
    from semantic_retrieval.evaluation.pilot_analysis import official_truth_from_relations

    y = official_truth_from_relations(["q"], {"q": ["v1"]}, ["v0", "v1", "v2"])
    assert y.tolist() == [[0, 1, 0]]
    with pytest.raises(ValueError, match="relation"):
        official_truth_from_relations(["q"], {"official_positives_by_query": {"q": ["v1"]}}, ["v1"])


def test_expansion_scores_only_visited_new_pairs_and_counts_real_calls():
    import torch

    from semantic_retrieval.evaluation.pilot_analysis import run_expanding_path
    from semantic_retrieval.types import PairScores

    calls = []

    def score(indices):
        calls.append(tuple(indices))
        logits = torch.tensor([-2.0, -2.0, 2.0])[indices]
        return PairScores(logits, logits[:, None], None)

    def presence(pairs):
        return pairs.logits.max()

    masks = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], bool)
    point = dict(pair_threshold=0.5, accept_threshold=0.5, accept_kind="presence")
    result = run_expanding_path(score, presence, ["a", "b", "c"], masks, (1, 2, 3), point)
    assert calls == [(0,), (1,), (2,)]
    assert result["attempted_k"] == [1, 2, 3]
    assert result["unique_pair_count"] == 3
    assert result["answer_indices"] == (2,)
    assert result["final_k"] == 3
    for aliases in (1, 2, 4, 8):
        for permutation in range(5):
            changed = run_expanding_path(
                score, presence, ["a", "b", "c"], masks, (1, 2, 3), point, alias_count=aliases, permutation=permutation
            )
            assert changed["answer_indices"] == result["answer_indices"]
            assert changed["attempted_k"] == result["attempted_k"]
            assert changed["unique_pair_count"] == 3


def test_connected_export_emits_real_expansion_and_invariance_arrays(model, inputs, tmp_path):
    from types import SimpleNamespace

    from semantic_retrieval.evaluation.pilot_analysis import export_model_scores
    from semantic_retrieval.types import QueryInputs

    context = SimpleNamespace(
        paths=SimpleNamespace(root=tmp_path, arrays=tmp_path / "arrays", provenance=tmp_path / "provenance"),
        manifest={"git_sha": "1" * 40},
    )
    context.paths.arrays.mkdir()
    context.paths.provenance.mkdir()
    ids = [f"c{i:03d}" for i in range(33)]
    qids = ["q0", "q1"]
    truth = np.zeros((2, 33), np.int8)
    truth[:, 32] = 1
    gallery = dict(version="locked", manifest_sha256="a" * 64, canonical_ids=ids, aliases={c: [c] for c in ids})
    orders = dict(
        query_ids=np.array(qids), canonical_ids=np.array(ids), ranks=np.tile(np.arange(33, dtype=np.int32), (2, 1))
    )
    point = dict(pair_threshold=1.0, accept_threshold=0.0, accept_kind="presence")
    frozen = dict(policies={"A3_seed0_controlled_fixed_K32": dict(operating_point=point)})
    fragment = export_model_scores(
        context,
        model,
        QueryInputs(inputs.input_ids, inputs.text_mask),
        inputs.video[[0] * 33],
        inputs.frame_mask[[0] * 33],
        query_ids=qids,
        component_ids=["s0", "s1"],
        canonical_relevance=truth,
        official_relevance=truth,
        gallery=gallery,
        panels=[],
        arm="A3",
        seed=0,
        split="val_locked",
        split_sha256="b" * 64,
        label_protocol={"name": "official"},
        preprocessing_sha256="c" * 64,
        policy_sha256="d" * 64,
        frozen=frozen,
        candidate_orders=orders,
    )
    systems = [g for g in fragment["groups"] if g["mechanism"] == "system"]
    assert len(systems) == 2
    assert len([g for g in fragment["groups"] if g["mechanism"] == "static"]) == 6
    for group in systems:
        with np.load(group["arrays"]["path"]) as arrays:
            n = len(arrays["query_ids"])
            assert arrays["score_error"].shape == (n, 4, 5)
            assert np.all(arrays["unique_pair_count"] == 33)
            assert np.all(arrays["attempted_k"] == [32, 64, 128])
            assert np.all(arrays["score_error"] <= 1e-6)
            assert np.all(arrays["answer_consistent"] == 1)
            assert np.all(arrays["attempted_k_consistent"] == 1)
            assert np.all(arrays["actual_returned_mask"] == 0)
            assert np.all(arrays["latency_ms"] > 0)
