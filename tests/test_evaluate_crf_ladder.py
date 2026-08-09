"""Task 9 — Single-pass CRF evaluator, dev selection, audit + MSVD reports."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.evaluate_crf_ladder import (
    _classify_audit_gates,
    choose_checkpoint,
    write_audit_report,
    write_msvd_report,
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
