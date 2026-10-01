"""Development release is independent of later locked effect/invariance gates."""

import json
import sys
from types import SimpleNamespace

import pytest

from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evaluation import pilot_analysis as analysis
from semantic_retrieval.runtime.artifacts import RunContext

REQUIRED = ("data", "four_gpu_qualification", "gradients", "stability")


@pytest.mark.parametrize("failed", REQUIRED + ("missing",))
def test_failed_authenticated_development_cannot_release_freeze(tmp_path, monkeypatch, failed):
    gates = dict.fromkeys(REQUIRED, True)
    if failed == "missing":
        gates.pop("stability")
    else:
        gates[failed] = False
    gate_file = tmp_path / "gates.json"
    gate_file.write_text(
        json.dumps(dict(relevance_space="benchmark_annotation", target_gallery_manifest_sha256="a" * 64))
    )
    bundle = dict(
        groups=[],
        galleries={"locked": {"manifest_sha256": "a" * 64}},
        qualification=dict(path=str(gate_file), sha256=file_sha256(gate_file)),
    )
    context = RunContext(paths=None, manifest={}, array_store=None, config=None, repo_root=tmp_path)
    monkeypatch.setattr(analysis, "_validate_bundle", lambda _: None)
    from semantic_retrieval.evaluation import pilot_archive

    monkeypatch.setattr(pilot_archive, "authenticate_score_fragments", lambda *args: None)
    monkeypatch.setitem(
        sys.modules,
        "semantic_retrieval.runtime.pilot_workflow",
        SimpleNamespace(validate_analysis_qualification=lambda *args: gates),
    )
    monkeypatch.setattr(analysis, "_freeze", lambda *args, **kwargs: pytest.fail("failed development reached freeze"))
    with pytest.raises(ValueError, match="development.*gate"):
        analysis.run_pilot_analysis(context, bundle, mode="freeze")


def test_frozen_artifact_persists_development_gates_without_requiring_locked_invariance(tmp_path):
    from test_pilot_review_math import gallery, group, rows

    from experiment_artifacts import ArtifactArrayStore

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
    gates = dict.fromkeys(REQUIRED, True)
    gates.update(invariance=False, power=False)
    frozen = analysis._freeze(context, bundle, [controlled], [], qualification_gates=gates)
    assert frozen["qualification_gates"] == dict.fromkeys(REQUIRED, True)
    assert frozen["development_gate_passed"] is True
    assert (
        json.loads((provenance / "pilot_frozen.json").read_text())["qualification_gates"]
        == frozen["qualification_gates"]
    )
