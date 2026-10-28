"""A successor cannot conceal an unapproved or changed failed attempt."""

import json
from types import SimpleNamespace

import pytest

from experiment_artifacts import append_registry_records, load_registry, registry_record_sha256
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.runtime import gpu_qualification
from test_runtime_artifacts import commit_reservation, git, reservation


@pytest.fixture
def reviewed_attempt(tmp_path):
    repo, rt = reservation.__wrapped__(tmp_path, SimpleNamespace(param=dict(dataset="synthetic")))
    commit_reservation(repo, rt)
    old = json.loads((rt.run_dir / "manifest.json").read_text())
    # Preserve the real reservation, plus an intentionally invalid committed status.
    status = rt.run_dir / "status.json"
    status.write_text(json.dumps(dict(state="validating", exit_code=1)))
    evidence = repo / "experiments/fixture-study/forensic.json"
    locks = {".lock", "status.json.lock", "provenance/audit.jsonl.lock"}
    files = {p.relative_to(rt.run_dir).as_posix(): file_sha256(p)
             for p in rt.run_dir.rglob("*")
             if p.is_file() and p.relative_to(rt.run_dir).as_posix() not in locks}
    evidence.write_text(json.dumps(dict(run_id=rt.run_id, artifact_path=str(rt.run_dir),
        source_sha=old["git_sha"], manifest_sha256=rt.manifest_sha256,
        observed_worker_exit_code=1, archive_status_valid=False, file_sha256=files)))
    report = repo / "experiments/fixture-study/manual-seal.json"
    report.write_text(json.dumps(dict(run_id=rt.run_id, manifest_sha256=rt.manifest_sha256,
        software_only=True, approved=True, observed_worker_exit_code=1,
        runtime_status_preserved=True, forensic_path=evidence.relative_to(repo).as_posix(),
        forensic_sha256=file_sha256(evidence))))
    reserved = load_registry(rt.registry_path)[0]
    correction = dict(reserved, event="correction", state="failed",
                      supersedes_sha256=registry_record_sha256(reserved))
    append_registry_records(rt.registry_path, [correction])
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "manually seal failed attempt without rewriting runtime evidence")
    successor = dict(old, run_id="successor", parent_run_ids=["completed-prep", rt.run_id],
        selection_lineage=[dict(kind="software_qualification_retry", parent_run_id=rt.run_id,
            approved=True, reason="User approved a new attempt after reviewing saturation fix",
            report_path=report.relative_to(repo).as_posix(), report_sha256=file_sha256(report))])
    return repo, rt, successor, report, evidence


def test_approved_successor_preserves_original_status_and_audit(reviewed_attempt):
    repo, rt, manifest, _, _ = reviewed_attempt
    original = {p: p.read_bytes() for p in (rt.run_dir / "status.json", rt.run_dir / "provenance/audit.jsonl")}
    gpu_qualification.validate_qualification_parents(repo, manifest)
    assert all(p.read_bytes() == value for p, value in original.items())


@pytest.mark.parametrize("defect", ["approval", "uncommitted", "report", "evidence", "original", "extra", "duplicate", "registry"])
def test_successor_rejects_missing_authority_or_changed_evidence(reviewed_attempt, defect):
    repo, rt, manifest, report, evidence = reviewed_attempt
    lineage = manifest["selection_lineage"][0]
    if defect == "approval":
        lineage["approved"] = False
    elif defect == "uncommitted":
        git(repo, "reset", "--soft", "HEAD~1")
    elif defect == "report":
        report.write_text(report.read_text() + " ")
    elif defect == "evidence":
        evidence.write_text(evidence.read_text() + " ")
    elif defect == "original":
        (rt.run_dir / "command.sh").write_text("changed")
    elif defect == "extra":
        manifest["parent_run_ids"].append("unreviewed")
    elif defect == "duplicate":
        manifest["parent_run_ids"][1] = "completed-prep"
    else:
        records = load_registry(rt.registry_path)
        rt.registry_path.write_text(json.dumps(records[0]) + "\n")
        git(repo, "add", ".")
        git(repo, "commit", "-qm", "missing terminal correction")
    with pytest.raises(ValueError):
        gpu_qualification.validate_qualification_parents(repo, manifest)
