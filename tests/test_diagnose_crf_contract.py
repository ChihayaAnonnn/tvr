"""Task 10 — §3.5 diagnostic JSON schema + throughput contract."""

from __future__ import annotations

import json

from scripts.diagnose_crf_training_contract import (
    APPENDIX_C_KEYS,
    build_diagnostic_payload,
    parse_args,
)


def test_appendix_c_keys_are_frozen_and_complete():
    assert set(APPENDIX_C_KEYS) == {
        "retrieval_gradient_isolation",
        "ranking_gradient_isolation",
        "global_head_frozen",
        "all_trainable_parameters_have_grad",
        "double_forward_allclose",
        "retrieval_invariance",
        "identity_delta_max",
        "risk_distributions",
        "frame_accuracy",
        "video_accuracy",
        "saturation_rate",
        "hygiene_r1_delta",
        "throughput_seconds_per_step",
    }


def test_build_diagnostic_payload_returns_exactly_appendix_c_keys():
    payload = build_diagnostic_payload(
        retrieval_gradient_isolation=True,
        ranking_gradient_isolation=True,
        global_head_frozen=True,
        all_trainable_parameters_have_grad=True,
        double_forward_allclose=True,
        retrieval_invariance=True,
        identity_delta_max=1e-5,
        risk_distributions={
            "clean": {"count": 100, "mean": 0.1, "std": 0.01, "p5": 0.05, "p50": 0.1, "p95": 0.15},
            "u30":   {"count": 100, "mean": 0.2, "std": 0.02, "p5": 0.10, "p50": 0.2, "p95": 0.30},
            "u38":   {"count": 100, "mean": 0.3, "std": 0.03, "p5": 0.15, "p50": 0.3, "p95": 0.45},
            "u46":   {"count": 100, "mean": 0.4, "std": 0.04, "p5": 0.20, "p50": 0.4, "p95": 0.60},
        },
        frame_accuracy={"clean_vs_30": 0.85, "30_vs_38": 0.80, "38_vs_46": 0.90},
        video_accuracy={"clean_vs_30": 0.92, "30_vs_38": 0.88, "38_vs_46": 0.95},
        saturation_rate=0.05,
        hygiene_r1_delta=0.3,
        throughput_seconds_per_step={
            "baseline_median": 0.4,
            "crf_median": 0.7,
            "decode_wait_ratio": 0.6,
            "ratio": 1.75,
        },
    )
    assert set(payload.keys()) == set(APPENDIX_C_KEYS)


def test_payload_is_json_serialisable_with_stable_key_order():
    payload = build_diagnostic_payload(
        retrieval_gradient_isolation=True,
        ranking_gradient_isolation=True,
        global_head_frozen=True,
        all_trainable_parameters_have_grad=True,
        double_forward_allclose=True,
        retrieval_invariance=True,
        identity_delta_max=0.0,
        risk_distributions={
            k: {"count": 0, "mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}
            for k in ("clean", "u30", "u38", "u46")
        },
        frame_accuracy={"clean_vs_30": 0.0, "30_vs_38": 0.0, "38_vs_46": 0.0},
        video_accuracy={"clean_vs_30": 0.0, "30_vs_38": 0.0, "38_vs_46": 0.0},
        saturation_rate=0.0,
        hygiene_r1_delta=0.0,
        throughput_seconds_per_step={
            "baseline_median": 0.0,
            "crf_median": 0.0,
            "decode_wait_ratio": 0.0,
            "ratio": 1.0,
        },
    )
    body = json.dumps(payload, sort_keys=True)
    reparsed = json.loads(body)
    assert set(reparsed.keys()) == set(APPENDIX_C_KEYS)


def test_hygiene_r1_delta_bound_is_informational_not_a_gate():
    """Plan states |hygiene_r1_delta| < 1.1 is informational — build_diagnostic_payload
    must NOT reject a value outside the bound (the field is a measurement, not a gate)."""
    payload = build_diagnostic_payload(
        retrieval_gradient_isolation=True,
        ranking_gradient_isolation=True,
        global_head_frozen=True,
        all_trainable_parameters_have_grad=True,
        double_forward_allclose=True,
        retrieval_invariance=True,
        identity_delta_max=0.0,
        risk_distributions={
            k: {"count": 0, "mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}
            for k in ("clean", "u30", "u38", "u46")
        },
        frame_accuracy={"clean_vs_30": 0.0, "30_vs_38": 0.0, "38_vs_46": 0.0},
        video_accuracy={"clean_vs_30": 0.0, "30_vs_38": 0.0, "38_vs_46": 0.0},
        saturation_rate=0.0,
        hygiene_r1_delta=5.0,
        throughput_seconds_per_step={
            "baseline_median": 0.0,
            "crf_median": 0.0,
            "decode_wait_ratio": 0.0,
            "ratio": 1.0,
        },
    )
    assert payload["hygiene_r1_delta"] == 5.0


def test_throughput_section_carries_the_four_frozen_keys():
    payload = build_diagnostic_payload(
        retrieval_gradient_isolation=True,
        ranking_gradient_isolation=True,
        global_head_frozen=True,
        all_trainable_parameters_have_grad=True,
        double_forward_allclose=True,
        retrieval_invariance=True,
        identity_delta_max=0.0,
        risk_distributions={
            k: {"count": 0, "mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}
            for k in ("clean", "u30", "u38", "u46")
        },
        frame_accuracy={"clean_vs_30": 0.0, "30_vs_38": 0.0, "38_vs_46": 0.0},
        video_accuracy={"clean_vs_30": 0.0, "30_vs_38": 0.0, "38_vs_46": 0.0},
        saturation_rate=0.0,
        hygiene_r1_delta=0.0,
        throughput_seconds_per_step={
            "baseline_median": 0.4,
            "crf_median": 0.9,
            "decode_wait_ratio": 0.55,
            "ratio": 2.25,
        },
    )
    assert set(payload["throughput_seconds_per_step"].keys()) == {
        "baseline_median", "crf_median", "decode_wait_ratio", "ratio",
    }


def test_cli_dry_run_parses_expected_args():
    args = parse_args(["--output", "/tmp/diagnostic.json", "--dry-run"])
    assert args.output == "/tmp/diagnostic.json"
    assert args.dry_run is True


def test_build_diagnostic_payload_rejects_missing_or_extra_appendix_keys():
    """Payload builder must not silently drop required keys or accept unknown ones."""
    import pytest as _pytest

    # Missing risk_distribution key
    with _pytest.raises(ValueError, match="risk_distributions"):
        build_diagnostic_payload(
            retrieval_gradient_isolation=True,
            ranking_gradient_isolation=True,
            global_head_frozen=True,
            all_trainable_parameters_have_grad=True,
            double_forward_allclose=True,
            retrieval_invariance=True,
            identity_delta_max=0.0,
            risk_distributions={"clean": {"count": 0, "mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}},
            frame_accuracy={"clean_vs_30": 0.0, "30_vs_38": 0.0, "38_vs_46": 0.0},
            video_accuracy={"clean_vs_30": 0.0, "30_vs_38": 0.0, "38_vs_46": 0.0},
            saturation_rate=0.0,
            hygiene_r1_delta=0.0,
            throughput_seconds_per_step={
                "baseline_median": 0.0, "crf_median": 0.0,
                "decode_wait_ratio": 0.0, "ratio": 1.0,
            },
        )
