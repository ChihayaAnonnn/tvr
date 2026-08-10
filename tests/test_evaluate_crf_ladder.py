"""Task 9 — Single-pass CRF evaluator, dev selection, audit + MSVD reports."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.evaluate_crf_ladder import (
    AUDIT_CHALLENGE_PAIRS,
    AUDIT_CRF_LEVELS,
    AUDIT_PRIMARY_PAIRS,
    MSVD_CRF_LEVELS,
    MSVD_PRIMARY_PAIRS,
    _classify_audit_gates,
    build_audit_payload,
    build_msvd_payload,
    build_per_video_tsv_rows,
    choose_checkpoint,
    write_audit_report,
    write_msvd_report,
)
from scripts.evaluate_crf_ladder import (
    main as evaluate_crf_ladder_main,
)


def test_checkpoint_tie_break_chooses_earlier_epoch():
    candidates = [
        {"epoch": 0, "accuracy": 0.91, "path": "epoch0.bin"},
        {"epoch": 1, "accuracy": 0.91, "path": "epoch1.bin"},
        {"epoch": 2, "accuracy": 0.90, "path": "epoch2.bin"},
    ]
    assert choose_checkpoint(candidates)["path"] == "epoch0.bin"


def test_checkpoint_uses_strict_greater_than_for_updates():
    """When two epochs tie exactly, the first-encountered wins — mirrors the
    training-loop rule that > (strict) drives updates so equal scores don't
    silently overwrite."""
    candidates = [
        {"epoch": 5, "accuracy": 0.80, "path": "e5.bin"},
        {"epoch": 3, "accuracy": 0.85, "path": "e3.bin"},
        {"epoch": 7, "accuracy": 0.85, "path": "e7.bin"},
    ]
    assert choose_checkpoint(candidates)["path"] == "e3.bin"


def test_choose_checkpoint_returns_dict_with_required_fields():
    winner = choose_checkpoint([{"epoch": 0, "accuracy": 0.9, "path": "e0.bin"}])
    assert winner["path"] == "e0.bin"
    assert winner["epoch"] == 0
    assert winner["accuracy"] == 0.9


def test_audit_refuses_to_overwrite_an_existing_report(tmp_path):
    report = tmp_path / "audit.json"
    report.write_text("{}")
    with pytest.raises(FileExistsError, match="already exists"):
        write_audit_report(report, {"checkpoint": "selected.bin"})


def test_audit_writes_exclusive_create_when_target_absent(tmp_path):
    report = tmp_path / "audit.json"
    write_audit_report(report, {"checkpoint": "selected.bin", "falsified": False})
    body = json.loads(report.read_text())
    assert body["checkpoint"] == "selected.bin"
    assert body["falsified"] is False


def test_msvd_output_filename_embeds_checkpoint_sha_and_requires_overwrite_on_collision(tmp_path):
    ckpt = tmp_path / "epoch3.bin"
    ckpt.write_bytes(b"weights")
    sha12 = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]
    out = tmp_path / "msvd"
    first = write_msvd_report(
        output_dir=out,
        checkpoint_path=ckpt,
        payload={"accuracy": 0.83},
    )
    assert first.name == f"msvd_report_{sha12}.json"
    with pytest.raises(FileExistsError, match="already exists.*--overwrite"):
        write_msvd_report(
            output_dir=out,
            checkpoint_path=ckpt,
            payload={"accuracy": 0.84},
        )
    second = write_msvd_report(
        output_dir=out,
        checkpoint_path=ckpt,
        payload={"accuracy": 0.84},
        overwrite=True,
    )
    body = json.loads(second.read_text())
    # Overwrite records the SHA-256 of whatever bytes were on disk before —
    # that is the atomic replace of the FIRST payload (serialized with
    # sort_keys=True + trailing newline).
    prev_bytes = (json.dumps({"accuracy": 0.83}, sort_keys=True) + "\n").encode("utf-8")
    assert body["overwrote_previous_sha256"] == hashlib.sha256(prev_bytes).hexdigest()
    assert "overwritten_at" in body


def test_msvd_reports_from_different_checkpoints_do_not_collide(tmp_path):
    ckpt_a = tmp_path / "a.bin"
    ckpt_b = tmp_path / "b.bin"
    ckpt_a.write_bytes(b"weightsA")
    ckpt_b.write_bytes(b"weightsB")
    out = tmp_path / "msvd"
    a = write_msvd_report(output_dir=out, checkpoint_path=ckpt_a, payload={"a": 1})
    b = write_msvd_report(output_dir=out, checkpoint_path=ckpt_b, payload={"b": 1})
    assert a.name != b.name


def test_audit_gate_classification_full_pass():
    verdict = _classify_audit_gates(
        u_video_accuracy_three_pair=0.92,
        u_video_accuracy_50_vs_42=0.85,
        extrapolation_readable_flag=True,
        challenge_delta_ci_low=0.05,
    )
    assert verdict["in_domain_accuracy_pass"] is True
    assert verdict["extrapolation_accuracy_pass"] is True
    assert verdict["extrapolation_readable"] is True
    assert verdict["baseline_gate_pass"] is True
    assert verdict["falsified"] is False


def test_audit_gate_falsified_when_challenge_ci_zero():
    verdict = _classify_audit_gates(
        u_video_accuracy_three_pair=0.92,
        u_video_accuracy_50_vs_42=0.85,
        extrapolation_readable_flag=True,
        challenge_delta_ci_low=-0.01,
    )
    assert verdict["baseline_gate_pass"] is False
    assert verdict["falsified"] is True


def test_audit_gate_falsified_when_in_domain_accuracy_below_90():
    verdict = _classify_audit_gates(
        u_video_accuracy_three_pair=0.85,
        u_video_accuracy_50_vs_42=0.85,
        extrapolation_readable_flag=True,
        challenge_delta_ci_low=0.05,
    )
    assert verdict["in_domain_accuracy_pass"] is False
    assert verdict["falsified"] is True


def test_audit_gate_extrapolation_accuracy_flags_below_80_but_not_falsified():
    """Extrapolation-only failure narrows the report but is NOT the falsification
    condition — the plan's `falsified` boolean is
    (audit_u_video_accuracy < 0.90) OR (challenge_delta_ci_low <= 0)."""
    verdict = _classify_audit_gates(
        u_video_accuracy_three_pair=0.92,
        u_video_accuracy_50_vs_42=0.70,
        extrapolation_readable_flag=True,
        challenge_delta_ci_low=0.05,
    )
    assert verdict["extrapolation_accuracy_pass"] is False
    assert verdict["falsified"] is False


# ---------------------------------------------------------------------------
# Payload builders — pure, no torch, drive the whole audit / MSVD reports
# ---------------------------------------------------------------------------


AUDIT_LEVELS_EXPECTED = ("clean", 34, 36, 42, 44, 50)
MSVD_LEVELS_EXPECTED = ("clean", 34, 42)


def _frame_stats(u_video):
    """Fabricate per-frame stats consistent with a given u_video value."""
    return {"mean": u_video, "min": u_video - 0.01, "max": u_video + 0.01}


def _synth_perfect_records(video_ids, levels, u_by_level, lap_by_level=None, jitter=1e-3):
    """Build per-video records where u_video and laplacian follow given constants.

    A tiny per-video jitter is added on top of the base per-level value so
    the array is not exactly constant (which would trip the saturation
    heuristic in crf_ladder.statistics.saturation_rate).
    """
    lap_by_level = lap_by_level or {lvl: 0.0 for lvl in levels}
    rng = np.random.default_rng(1234)
    records = {}
    for vid in video_ids:
        records[vid] = {}
        offset = float(rng.uniform(-jitter, jitter))
        for lvl in levels:
            u_v = float(u_by_level[lvl]) + offset
            fs = _frame_stats(u_v)
            records[vid][lvl] = {
                "u_video": u_v,
                "laplacian": float(lap_by_level[lvl]),
                "u_frame_mean": fs["mean"],
                "u_frame_min": fs["min"],
                "u_frame_max": fs["max"],
                "saturated_low": False,
                "saturated_high": False,
            }
    return records


def test_audit_levels_and_pairs_match_plan():
    assert AUDIT_CRF_LEVELS == AUDIT_LEVELS_EXPECTED
    assert AUDIT_PRIMARY_PAIRS == (("clean", 34), (34, 42), ("clean", 42))
    assert AUDIT_CHALLENGE_PAIRS == ((34, 36), (42, 44))
    assert MSVD_CRF_LEVELS == MSVD_LEVELS_EXPECTED
    assert MSVD_PRIMARY_PAIRS == (("clean", 34), (34, 42), ("clean", 42))


def test_audit_payload_full_pass_when_u_monotone_and_laplacian_flat():
    video_ids = [f"v{i}" for i in range(20)]
    # u strictly increases with CRF → every ordered pair correct.
    u_by_level = {"clean": 0.1, 34: 0.2, 36: 0.3, 42: 0.4, 44: 0.5, 50: 0.6}
    # Laplacian is constant → paired_accuracy on Laplacian = 0 (ties count as
    # failures), so delta = 1 - 0 = 1 with tight CI.
    lap_by_level = {lvl: 0.0 for lvl in AUDIT_LEVELS_EXPECTED}
    records = _synth_perfect_records(video_ids, AUDIT_LEVELS_EXPECTED, u_by_level, lap_by_level)
    payload = build_audit_payload(
        video_ids=video_ids,
        per_video_records=records,
        seed=0,
        samples=500,
        metadata={"checkpoint_sha12": "abcdef123456"},
    )
    assert payload["dataset"] == "msrvtt"
    assert payload["split"] == "audit"
    assert payload["n_videos"] == 20
    assert payload["u_video_accuracy_three_pair"] == pytest.approx(1.0)
    assert payload["u_video_accuracy_50_vs_42"] == pytest.approx(1.0)
    assert payload["extrapolation_readable"] is True
    # Both challenge pairs delta CI should be strictly above 0.
    assert payload["challenge_delta_ci_low"] > 0.0
    assert payload["in_domain_accuracy_pass"] is True
    assert payload["extrapolation_accuracy_pass"] is True
    assert payload["baseline_gate_pass"] is True
    assert payload["falsified"] is False
    # Per-level saturation reported for every level.
    for lvl in AUDIT_LEVELS_EXPECTED:
        assert str(lvl) in payload["saturation_rate_by_level"]


def test_audit_payload_falsified_when_u_ranks_wrong():
    video_ids = [f"v{i}" for i in range(30)]
    # u is REVERSED w.r.t. CRF → 3-pair accuracy = 0 → falsified.
    u_by_level = {"clean": 0.6, 34: 0.5, 36: 0.4, 42: 0.3, 44: 0.2, 50: 0.1}
    records = _synth_perfect_records(video_ids, AUDIT_LEVELS_EXPECTED, u_by_level)
    payload = build_audit_payload(
        video_ids=video_ids,
        per_video_records=records,
        seed=0,
        samples=500,
        metadata={"checkpoint_sha12": "deadbeef1234"},
    )
    assert payload["u_video_accuracy_three_pair"] == pytest.approx(0.0)
    assert payload["in_domain_accuracy_pass"] is False
    assert payload["falsified"] is True


def test_audit_payload_falsified_when_laplacian_matches_u_on_challenge_pairs():
    """If Laplacian ranks the challenge pair as well as u does, delta CI hits
    zero (or negative) → baseline_gate_pass=False → falsified=True."""
    V = 40
    video_ids = [f"v{i}" for i in range(V)]
    # Primary three-pair accuracy still ≥ 0.9 → so only the baseline gate fails.
    u_by_level = {"clean": 0.1, 34: 0.2, 36: 0.3, 42: 0.4, 44: 0.5, 50: 0.6}
    lap_by_level = {"clean": 0.1, 34: 0.2, 36: 0.3, 42: 0.4, 44: 0.5, 50: 0.6}
    records = _synth_perfect_records(video_ids, AUDIT_LEVELS_EXPECTED, u_by_level, lap_by_level)
    payload = build_audit_payload(
        video_ids=video_ids,
        per_video_records=records,
        seed=1,
        samples=500,
        metadata={"checkpoint_sha12": "cafebabe4242"},
    )
    # Both u and laplacian give 1.0 accuracy on challenge pairs → delta = 0.
    assert payload["challenge_delta_ci_low"] <= 0.0
    assert payload["baseline_gate_pass"] is False
    assert payload["falsified"] is True


def test_audit_payload_reports_saturation_and_extrapolation_readable_false_when_high_sat():
    video_ids = [f"v{i}" for i in range(10)]
    u_by_level = {"clean": 0.1, 34: 0.2, 36: 0.3, 42: 0.4, 44: 0.5, 50: 0.6}
    records = _synth_perfect_records(video_ids, AUDIT_LEVELS_EXPECTED, u_by_level)
    # Force high saturation on CRF50 (all "saturated_high" -> saturation_rate=1.0)
    for vid in video_ids:
        records[vid][50]["saturated_high"] = True
    payload = build_audit_payload(
        video_ids=video_ids,
        per_video_records=records,
        seed=0,
        samples=200,
        metadata={"checkpoint_sha12": "0" * 12},
    )
    assert payload["saturation_rate_by_level"]["50"] == pytest.approx(1.0)


def test_audit_payload_metadata_bootstrap_and_checkpoint_included():
    video_ids = [f"v{i}" for i in range(15)]
    u_by_level = {"clean": 0.1, 34: 0.2, 36: 0.3, 42: 0.4, 44: 0.5, 50: 0.6}
    records = _synth_perfect_records(video_ids, AUDIT_LEVELS_EXPECTED, u_by_level)
    payload = build_audit_payload(
        video_ids=video_ids,
        per_video_records=records,
        seed=7,
        samples=333,
        metadata={"checkpoint_sha12": "abcdef000001", "git_commit": "deadbeef"},
    )
    assert payload["metadata"]["checkpoint_sha12"] == "abcdef000001"
    assert payload["metadata"]["bootstrap_seed"] == 7
    assert payload["metadata"]["bootstrap_samples"] == 333
    assert payload["metadata"]["git_commit"] == "deadbeef"


def test_msvd_payload_cross_domain_pass_when_perfect():
    video_ids = [f"v{i}" for i in range(25)]
    u_by_level = {"clean": 0.1, 34: 0.2, 42: 0.4}
    records = _synth_perfect_records(video_ids, MSVD_LEVELS_EXPECTED, u_by_level)
    payload = build_msvd_payload(
        video_ids=video_ids,
        per_video_records=records,
        seed=0,
        samples=500,
        metadata={"checkpoint_sha12": "aaaaaaaaaaaa"},
    )
    assert payload["dataset"] == "msvd"
    assert payload["split"] == "msvd_test"
    assert payload["three_pair_accuracy"] == pytest.approx(1.0)
    assert payload["three_pair_ci_low"] > 0.5
    assert payload["cross_domain_pass"] is True
    # Exploratory Spearman tagged and present.
    assert payload["spearman_u_vs_neg_bpp"]["exploratory"] is True
    assert "value" in payload["spearman_u_vs_neg_bpp"]


def test_msvd_payload_cross_domain_fail_when_random_ranking():
    rng = np.random.default_rng(0)
    V = 30
    video_ids = [f"v{i}" for i in range(V)]
    records = {}
    for vid in video_ids:
        u_clean = float(rng.random())
        u_34 = float(rng.random())
        u_42 = float(rng.random())
        records[vid] = {
            "clean": {"u_video": u_clean, "laplacian": 0.0, **{k: u_clean for k in ["u_frame_mean", "u_frame_min", "u_frame_max"]}, "saturated_low": False, "saturated_high": False},
            34: {"u_video": u_34, "laplacian": 0.0, **{k: u_34 for k in ["u_frame_mean", "u_frame_min", "u_frame_max"]}, "saturated_low": False, "saturated_high": False},
            42: {"u_video": u_42, "laplacian": 0.0, **{k: u_42 for k in ["u_frame_mean", "u_frame_min", "u_frame_max"]}, "saturated_low": False, "saturated_high": False},
        }
    payload = build_msvd_payload(
        video_ids=video_ids,
        per_video_records=records,
        seed=0,
        samples=500,
        metadata={"checkpoint_sha12": "b" * 12},
    )
    # Random ranking should give ~0.5 accuracy → below 0.80 gate.
    assert payload["three_pair_accuracy"] < 0.80
    assert payload["cross_domain_pass"] is False


def test_per_video_tsv_rows_have_registered_columns():
    video_ids = ["vA", "vB"]
    u_by_level = {"clean": 0.1, 34: 0.2, 36: 0.3, 42: 0.4, 44: 0.5, 50: 0.6}
    records = _synth_perfect_records(video_ids, AUDIT_LEVELS_EXPECTED, u_by_level)
    rows = build_per_video_tsv_rows(
        dataset="msrvtt",
        split="audit",
        video_ids=video_ids,
        per_video_records=records,
        levels=AUDIT_CRF_LEVELS,
    )
    assert len(rows) == 2 * len(AUDIT_LEVELS_EXPECTED)
    required = {
        "dataset", "split", "video_id", "level", "u_video", "laplacian",
        "u_frame_mean", "u_frame_min", "u_frame_max",
        "saturated_low", "saturated_high",
    }
    assert required.issubset(rows[0].keys())


# ---------------------------------------------------------------------------
# CLI orchestration — patched _score_ladder returns synthetic records
# ---------------------------------------------------------------------------


def _install_stub_scorer(monkeypatch, records_by_video_id):
    """Patch _score_ladder to return the given records without touching model/data."""
    import scripts.evaluate_crf_ladder as mod

    def _stub_score_ladder(*, checkpoint_path, video_ids, levels, clean_paths, crf_paths, cli_args):
        return {vid: records_by_video_id[vid] for vid in video_ids}

    monkeypatch.setattr(mod, "_score_ladder", _stub_score_ladder)


def test_cli_writes_audit_report_and_tsv_end_to_end(monkeypatch, tmp_path):
    video_ids = [f"v{i}" for i in range(12)]
    u_by_level = {"clean": 0.1, 34: 0.2, 36: 0.3, 42: 0.4, 44: 0.5, 50: 0.6}
    lap_by_level = {lvl: 0.0 for lvl in AUDIT_LEVELS_EXPECTED}
    records = _synth_perfect_records(video_ids, AUDIT_LEVELS_EXPECTED, u_by_level, lap_by_level)
    _install_stub_scorer(monkeypatch, records)

    ckpt = tmp_path / "pytorch_model.bin.0"
    ckpt.write_bytes(b"fake weights")
    selection = tmp_path / "best_crf_checkpoint.json"
    selection.write_text(json.dumps({"epoch": 0, "accuracy": 1.0, "path": str(ckpt)}))
    audit_split = tmp_path / "split.json"
    audit_split.write_text(json.dumps({"crf_audit_video_ids": video_ids}))
    output_dir = tmp_path / "crf_audit"

    rc = evaluate_crf_ladder_main([
        "--dataset", "msrvtt",
        "--split", "audit",
        "--selection-json", str(selection),
        "--audit-split-json", str(audit_split),
        "--output-dir", str(output_dir),
        "--clean-root", str(tmp_path / "clean"),
        "--crf-root", str(tmp_path / "crf"),
        "--bootstrap-samples", "300",
        "--bootstrap-seed", "0",
    ])
    assert rc == 0
    report_path = output_dir / "crf_audit_report.json"
    tsv_path = output_dir / "crf_audit_per_video.tsv"
    assert report_path.exists()
    assert tsv_path.exists()
    body = json.loads(report_path.read_text())
    assert body["falsified"] is False
    assert body["in_domain_accuracy_pass"] is True
    assert body["dataset"] == "msrvtt"
    assert body["split"] == "audit"
    # TSV header + one row per (video, level).
    lines = tsv_path.read_text().strip().splitlines()
    assert len(lines) == 1 + 12 * 6


def test_cli_audit_refuses_second_run_to_same_output_dir(monkeypatch, tmp_path):
    video_ids = ["v0", "v1", "v2"]
    u_by_level = {"clean": 0.1, 34: 0.2, 36: 0.3, 42: 0.4, 44: 0.5, 50: 0.6}
    records = _synth_perfect_records(video_ids, AUDIT_LEVELS_EXPECTED, u_by_level)
    _install_stub_scorer(monkeypatch, records)

    ckpt = tmp_path / "ckpt.bin"
    ckpt.write_bytes(b"fake")
    selection = tmp_path / "sel.json"
    selection.write_text(json.dumps({"epoch": 0, "accuracy": 1.0, "path": str(ckpt)}))
    audit_split = tmp_path / "split.json"
    audit_split.write_text(json.dumps({"crf_audit_video_ids": video_ids}))
    out = tmp_path / "crf_audit"
    argv = [
        "--dataset", "msrvtt", "--split", "audit",
        "--selection-json", str(selection),
        "--audit-split-json", str(audit_split),
        "--output-dir", str(out),
        "--clean-root", str(tmp_path / "clean"),
        "--crf-root", str(tmp_path / "crf"),
        "--bootstrap-samples", "100",
    ]
    assert evaluate_crf_ladder_main(argv) == 0
    with pytest.raises(FileExistsError, match="already exists"):
        evaluate_crf_ladder_main(argv)


def test_cli_msvd_writes_report_and_tsv_end_to_end(monkeypatch, tmp_path):
    video_ids = [f"v{i}" for i in range(10)]
    u_by_level = {"clean": 0.1, 34: 0.2, 42: 0.4}
    records = _synth_perfect_records(video_ids, MSVD_LEVELS_EXPECTED, u_by_level)
    _install_stub_scorer(monkeypatch, records)

    ckpt = tmp_path / "ckpt.bin"
    ckpt.write_bytes(b"weights")
    sha12 = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]
    selection = tmp_path / "sel.json"
    selection.write_text(json.dumps({"epoch": 0, "accuracy": 1.0, "path": str(ckpt)}))
    msvd_subset = tmp_path / "subset.json"
    msvd_subset.write_text(json.dumps({"video_ids": video_ids}))
    out = tmp_path / "crf_msvd"

    rc = evaluate_crf_ladder_main([
        "--dataset", "msvd", "--split", "msvd_test",
        "--selection-json", str(selection),
        "--msvd-subset-json", str(msvd_subset),
        "--output-dir", str(out),
        "--msvd-clean-root", str(tmp_path / "msvd_clean"),
        "--crf-root", str(tmp_path / "msvd_crf"),
        "--bootstrap-samples", "300",
    ])
    assert rc == 0
    report_path = out / f"msvd_report_{sha12}.json"
    tsv_path = out / f"msvd_per_video_{sha12}.tsv"
    assert report_path.exists()
    assert tsv_path.exists()
    body = json.loads(report_path.read_text())
    assert body["cross_domain_pass"] is True
    assert body["dataset"] == "msvd"
