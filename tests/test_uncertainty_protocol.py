import json

import numpy as np
import pytest
import torch

from prob_models.uncertainty_protocol import (
    append_candidate_atomic,
    compare_paired_invariance,
    file_sha256,
    non_sidecar_state_dict_mismatches,
    validate_candidate_artifacts,
    validate_resume_lineage,
    validate_selected_checkpoint,
)


def _candidate(tmp_path, epoch):
    model = tmp_path / f"model.{epoch}.pt"
    optimizer = tmp_path / f"optimizer.{epoch}.pt"
    model.write_bytes(f"model-{epoch}".encode())
    optimizer.write_bytes(f"optimizer-{epoch}".encode())
    return {
        "epoch": epoch,
        "path": str(model.resolve()),
        "model_sha256": file_sha256(model),
        "optimizer_path": str(optimizer.resolve()),
        "optimizer_sha256": file_sha256(optimizer),
        "non_a3_mismatches": [],
    }


def test_candidate_artifacts_reject_outside_path_and_hash_mismatch(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    outside = _candidate(tmp_path, 0)

    with pytest.raises(ValueError, match="outside the output directory"):
        validate_candidate_artifacts(
            [outside], output_dir=output, method_name="A3"
        )

    inside = _candidate(output, 1)
    inside["model_sha256"] = "wrong"
    with pytest.raises(ValueError, match="A3 candidate model.*SHA-256"):
        validate_candidate_artifacts(
            [inside], output_dir=output, method_name="A3"
        )


def test_candidate_artifacts_reject_duplicate_epochs_and_paths(tmp_path):
    first = _candidate(tmp_path, 0)
    duplicate_epoch = {**_candidate(tmp_path, 1), "epoch": 0}
    with pytest.raises(ValueError, match="epochs must be unique"):
        validate_candidate_artifacts(
            [first, duplicate_epoch], output_dir=tmp_path, method_name="A3"
        )

    duplicate_path = {**_candidate(tmp_path, 2), "path": first["path"]}
    duplicate_path["model_sha256"] = first["model_sha256"]
    with pytest.raises(ValueError, match="artifact paths must be unique"):
        validate_candidate_artifacts(
            [first, duplicate_path], output_dir=tmp_path, method_name="A3"
        )

    cross_role = {
        **_candidate(tmp_path, 3),
        "path": first["optimizer_path"],
        "model_sha256": first["optimizer_sha256"],
    }
    with pytest.raises(ValueError, match="artifact paths must be unique"):
        validate_candidate_artifacts(
            [first, cross_role], output_dir=tmp_path, method_name="A3"
        )


def test_append_candidate_is_atomic_and_strictly_append_only(tmp_path, monkeypatch):
    path = tmp_path / "candidates.json"
    first = _candidate(tmp_path, 1)
    second = _candidate(tmp_path, 2)
    replacements = []

    import prob_models.uncertainty_protocol as protocol

    real_replace = protocol.os.replace

    def recording_replace(source, destination):
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(protocol.os, "replace", recording_replace)

    assert append_candidate_atomic(path, first) == [first]
    assert append_candidate_atomic(path, second) == [first, second]
    assert json.loads(path.read_text()) == [first, second]
    assert len(replacements) == 2
    assert all(str(destination) == str(path) for _, destination in replacements)
    assert all(str(source) != str(path) for source, _ in replacements)

    original = path.read_bytes()
    for duplicate in (
        {**_candidate(tmp_path, 3), "epoch": 2},
        {**_candidate(tmp_path, 4), "path": first["path"]},
        {**_candidate(tmp_path, 5), "optimizer_path": first["optimizer_path"]},
        {**_candidate(tmp_path, 6), "path": first["optimizer_path"]},
        {
            **_candidate(tmp_path, 7),
            "optimizer_path": first["path"],
        },
    ):
        with pytest.raises(ValueError, match="append-only|unique"):
            append_candidate_atomic(path, duplicate)
        assert path.read_bytes() == original

    with pytest.raises(ValueError, match="append-only"):
        append_candidate_atomic(path, {**_candidate(tmp_path, 8), "epoch": 0})
    assert path.read_bytes() == original


def test_append_candidate_serializes_finite_array_evidence(tmp_path):
    path = tmp_path / "candidates.json"
    candidate = {
        **_candidate(tmp_path, 0),
        "evidence": {
            "numpy": np.array([0.1, 0.2], dtype=np.float32),
            "tensor": torch.tensor([0.3, 0.4]),
        },
    }

    appended = append_candidate_atomic(path, candidate)

    assert appended[0]["evidence"]["numpy"] == pytest.approx([0.1, 0.2])
    assert appended[0]["evidence"]["tensor"] == pytest.approx([0.3, 0.4])
    assert json.loads(path.read_text()) == appended


def test_append_candidate_refuses_to_overwrite_non_list_artifact(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text('{"registered": true}')

    with pytest.raises(ValueError, match="candidate list"):
        append_candidate_atomic(path, _candidate(tmp_path, 0))

    assert path.read_text() == '{"registered": true}'


def test_resume_lineage_rejects_non_highest_epoch(tmp_path):
    older = _candidate(tmp_path, 1)
    latest = _candidate(tmp_path, 2)

    with pytest.raises(ValueError, match="highest-epoch"):
        validate_resume_lineage(
            [older, latest],
            output_dir=tmp_path,
            model_checkpoint=older["path"],
            optimizer_checkpoint=older["optimizer_path"],
            optimizer_payload={"epoch": 1},
            sidecar_prefix="contextual_risk_sidecar.",
            method_name="A3",
        )


def test_non_sidecar_audit_allows_only_registered_prefixes(tmp_path):
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    torch.save(
        {
            "backbone": torch.tensor([1.0], dtype=torch.float16),
            "retrieval_risk_sidecar.w": torch.tensor([2.0]),
        },
        base,
    )
    torch.save(
        {
            "backbone": torch.tensor([1.0], dtype=torch.float32),
            "retrieval_risk_sidecar.w": torch.tensor([2.0]),
            "contextual_risk_sidecar.w": torch.tensor([3.0]),
        },
        candidate,
    )

    assert non_sidecar_state_dict_mismatches(
        base, candidate, ("contextual_risk_sidecar.",)
    ) == []

    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    payload["backbone"] = torch.tensor([9.0])
    torch.save(payload, candidate)
    assert non_sidecar_state_dict_mismatches(
        base, candidate, ("contextual_risk_sidecar.",)
    ) == ["backbone"]


def test_paired_invariance_reports_missing_and_mismatched_digests():
    baseline = {"raw_wti_sha256": "raw", "retrieval": {"R1": 1.0}}
    candidate = {"raw_wti_sha256": "changed"}

    comparison = compare_paired_invariance(
        baseline,
        candidate,
        ("raw_wti_sha256", "retrieval"),
    )

    assert comparison == {
        "passed": False,
        "checks": {"raw_wti": False, "retrieval": False},
    }


def test_csd_paired_invariance_preserves_missing_digest_key_error():
    from prob_models.csd_logvar_protocol import compare_paired_invariance as compare_csd

    with pytest.raises(KeyError, match="a1_failure_risk_sha256"):
        compare_csd({"raw_wti_sha256": "raw"}, {"raw_wti_sha256": "raw"})


def test_selected_checkpoint_revalidates_record_and_file(tmp_path):
    candidate = _candidate(tmp_path, 1)
    selection = {"winner": dict(candidate), "candidates": [candidate]}
    assert validate_selected_checkpoint(
        selection, output_dir=tmp_path, method_name="A3"
    ) == candidate

    selection["winner"]["model_sha256"] = "changed"
    with pytest.raises(ValueError, match="selected A3 model hash"):
        validate_selected_checkpoint(
            selection, output_dir=tmp_path, method_name="A3"
        )
