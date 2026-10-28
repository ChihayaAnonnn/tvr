import numpy as np
import pytest


def binary_calibration_metrics(*args, **kwargs):
    from semantic_retrieval.evaluation.binary import binary_calibration_metrics as implementation

    return implementation(*args, **kwargs)


def paired_bootstrap_delta(*args, **kwargs):
    from semantic_retrieval.evaluation.binary import paired_bootstrap_delta as implementation

    return implementation(*args, **kwargs)


def test_binary_calibration_metrics_match_hand_computation():
    labels = np.array([False, False, True, True])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    result = binary_calibration_metrics(labels, probabilities, bins=2)

    assert result["ece"] == pytest.approx(0.15)
    assert result["brier"] == pytest.approx(0.025)
    assert result["nll"] == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.8, 0.9])))


@pytest.mark.parametrize(
    ("labels", "probabilities", "message"),
    [
        ([0, 1], [0.2], "shape"),
        ([0, 1], [0.2, np.nan], "finite"),
        ([0, 1], [-0.1, 0.8], r"\[0, 1\]"),
    ],
)
def test_binary_calibration_metrics_reject_invalid_inputs(labels, probabilities, message):
    with pytest.raises(ValueError, match=message):
        binary_calibration_metrics(labels, probabilities)


@pytest.mark.parametrize("metric", ["auroc", "aurc"])
def test_paired_bootstrap_is_deterministic_and_paired(metric):
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    first_risk = np.array([0.1, 0.2, 0.4, 0.7, 0.8, 0.9])
    second_risk = np.array([0.2, 0.4, 0.6, 0.3, 0.5, 0.7])

    first = paired_bootstrap_delta(labels, first_risk, second_risk, metric, 10_000, 0)
    second = paired_bootstrap_delta(labels, first_risk, second_risk, metric, 10_000, 0)

    assert first == second
    assert first["metric"] == metric
    assert first["samples"] == 10_000
    assert first["seed"] == 0
    assert first["valid_replicates"] == 10_000
    if metric == "auroc":
        assert first["point_delta"] > 0
    else:
        assert first["point_delta"] < 0
    assert first["ci_low"] <= first["ci_high"]


def test_paired_bootstrap_rejects_invalid_inputs_and_constant_labels():
    labels = np.array([False, False, False])
    risks = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="both success and failure"):
        paired_bootstrap_delta(labels, risks, risks, "auroc", 10, 0)

    labels = np.array([False, True, True])
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_delta(labels, risks, np.array([0.1, np.nan, 0.3]), "auroc", 10, 0)
    with pytest.raises(ValueError, match="metric"):
        paired_bootstrap_delta(labels, risks, risks, "rho", 10, 0)
    with pytest.raises(ValueError, match="positive"):
        paired_bootstrap_delta(labels, risks, risks, "auroc", 0, 0)
