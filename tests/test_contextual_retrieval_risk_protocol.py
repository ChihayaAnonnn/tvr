import copy

import numpy as np
import pytest

from prob_models.uncertainty_protocol import (
    choose_contextual_risk_checkpoint,
    contextual_risk_acceptance,
    contextual_risk_candidate_is_valid,
    contextual_risk_selection_tuple,
    file_sha256,
)


def _direction():
    return {
        "spearman_risk_vs_rank": 0.20,
        "failure_auroc": {"value": 0.70, "ci_low": 0.51, "ci_high": 0.80},
        "risk_coverage": {
            "accuracy_at_0.25": 0.90,
            "accuracy_at_0.50": 0.80,
            "accuracy_at_0.75": 0.70,
            "accuracy_at_1.00": 0.60,
        },
        "risk_stats": {"mean": 0.40, "std": 0.02},
        "aurc": 0.25,
    }


def _metrics():
    return {
        "t2v": _direction(),
        "v2t": _direction(),
        "paired_invariance": {"passed": True},
        "non_a3_mismatches": [],
    }


def _candidate(epoch=0, **updates):
    candidate = {
        "epoch": epoch,
        "valid": True,
        "finite": True,
        "runtime_abort": False,
        "rho": [0.30, 0.30],
        "auroc": [0.70, 0.70],
        "aurc": [0.30, 0.30],
        "evidence": {"risk": [0.1, 0.2]},
        "paired_digest": {
            "raw_wti_sha256": "raw",
            "a1_failure_risk_sha256": {"t2v": "tf", "v2t": "vf"},
            "a1_vacuity_sha256": {"t2v": "tv", "v2t": "vv"},
            "retrieval": {"t2v": {"R1": 1.0}, "v2t": {"R1": 2.0}},
        },
        "paired_invariance": {
            "passed": True,
            "checks": {
                "raw_wti": True,
                "a1_failure_risk": True,
                "a1_vacuity": True,
                "retrieval": True,
            },
        },
        "non_a3_mismatches": [],
    }
    candidate.update(updates)
    return candidate


def _with_artifacts(tmp_path, candidate):
    model = tmp_path / f"model.{candidate['epoch']}.pt"
    optimizer = tmp_path / f"optimizer.{candidate['epoch']}.pt"
    model.write_bytes(f"model-{candidate['epoch']}".encode())
    optimizer.write_bytes(f"optimizer-{candidate['epoch']}".encode())
    return {
        **candidate,
        "path": str(model.resolve()),
        "model_sha256": file_sha256(model),
        "optimizer_path": str(optimizer.resolve()),
        "optimizer_sha256": file_sha256(optimizer),
    }


def test_selection_tuple_is_registered_lexicographic_order():
    candidate = _candidate(
        epoch=3,
        rho=[0.30, 0.21],
        auroc=[0.60, 0.80],
        aurc=[0.10, 0.30],
    )

    assert contextual_risk_selection_tuple(candidate) == (
        0.21,
        0.70,
        -0.20,
        -3,
    )


def test_selector_excludes_invalid_candidates_and_uses_aurc_then_earlier_epoch(
    tmp_path,
):
    invalid = _with_artifacts(
        tmp_path, _candidate(0, valid=False, rho=[0.99, 0.99])
    )
    worse_aurc = _with_artifacts(tmp_path, _candidate(1, aurc=[0.4, 0.4]))
    later = _with_artifacts(tmp_path, _candidate(3))
    earlier = _with_artifacts(tmp_path, _candidate(2))

    assert choose_contextual_risk_checkpoint(
        [invalid, worse_aurc, later, earlier], output_dir=tmp_path
    ) == earlier


def test_selector_requires_artifact_verification():
    with pytest.raises(TypeError, match="output_dir"):
        choose_contextual_risk_checkpoint([_candidate()])


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda m: m["v2t"].update(spearman_risk_vs_rank=0.19), "rho"),
        (
            lambda m: m["v2t"]["failure_auroc"].update(ci_low=0.50),
            "auroc_ci",
        ),
        (
            lambda m: m["v2t"]["risk_coverage"].update(
                {"accuracy_at_0.75": 0.90}
            ),
            "selective",
        ),
        (lambda m: m["v2t"]["risk_stats"].update(std=0.019), "spread"),
        (lambda m: m["paired_invariance"].update(passed=False), "invariance"),
        (lambda m: m["non_a3_mismatches"].append("backbone"), "state"),
    ],
)
def test_acceptance_enforces_every_registered_threshold(mutation, failed_check):
    accepted = contextual_risk_acceptance(_metrics())
    rejected_metrics = copy.deepcopy(_metrics())
    mutation(rejected_metrics)

    rejected = contextual_risk_acceptance(rejected_metrics)

    assert accepted["passed"] is True
    assert accepted["allow_final_test"] is True
    assert all(accepted["checks"].values())
    assert rejected["passed"] is False
    assert rejected["allow_final_test"] is False
    assert rejected["checks"][failed_check] is False


@pytest.mark.parametrize(
    "updates",
    [
        {"finite": False},
        {"rho": None},
        {"evidence": {}},
        {"evidence": {"risk": [0.1, float("nan")]}},
        {"runtime_abort": True},
        {"non_a3_mismatches": ["backbone"]},
        {"paired_digest": {"raw_wti_sha256": "raw"}},
        {"paired_invariance": None},
        {"paired_invariance": {"passed": False, "checks": {}}},
    ],
)
def test_candidate_validity_requires_finite_complete_invariant_evidence(updates):
    assert contextual_risk_candidate_is_valid(_candidate(**updates)) is False


def test_candidate_validity_accepts_finite_numpy_evidence():
    candidate = _candidate(evidence={"risk": np.array([0.1, 0.2])})

    assert contextual_risk_candidate_is_valid(candidate) is True


def test_selector_rejects_candidates_without_intact_artifacts(tmp_path):
    model = tmp_path / "model.pt"
    optimizer = tmp_path / "optimizer.pt"
    model.write_bytes(b"model")
    optimizer.write_bytes(b"optimizer")
    candidate = _candidate(
        path=str(model.resolve()),
        model_sha256="tampered",
        optimizer_path=str(optimizer.resolve()),
        optimizer_sha256="tampered",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        choose_contextual_risk_checkpoint([candidate], output_dir=tmp_path)
