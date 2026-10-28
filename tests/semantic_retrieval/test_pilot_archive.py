"""Composition checks after the separately tested lifecycle/phase authentication."""

import json
from types import SimpleNamespace

import pytest

from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evaluation import pilot_archive as archive


@pytest.fixture
def archived_fragment(tmp_path, monkeypatch):
    refs = {}

    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        ref = dict(path=str(path), sha256=file_sha256(path))
        refs[str(path)] = ref["sha256"]
        return ref

    label = {"protocol_id": "official"}
    binding = dict(
        phase="P1b",
        run_id="training",
        summary=write("train-summary.json", {}),
        config=write("train-config.json", {}),
        checkpoint=write("training/checkpoint.pt", {}),
        metadata=write("training/checkpoint.json", dict(config_sha256="config", model_sha256="model")),
        model_sha256="model",
        source_sha="train-source",
        manifest_sha256="manifest",
    )
    metadata = json.loads(open(binding["metadata"]["path"]).read())
    metadata["sha256"] = binding["checkpoint"]["sha256"]
    binding["metadata"] = write("training/checkpoint.json", metadata)
    group = dict(
        arm="A3",
        seed=0,
        model_sha256="model",
        split="val_locked",
        split_sha256="split",
        track="natural",
        mechanism="fixed",
        K=32,
        arrays=write("scores/scores.npz", {}),
    )
    fragment = dict(
        source_sha="export-source",
        label_protocol=label,
        relevance_space="benchmark_annotation",
        preprocessing_sha256="preprocessing",
        policy_sha256="policy",
        galleries={},
        groups=[group],
        producer_identity=dict(run_id="scores", manifest_sha256="manifest", training_parent=binding),
    )
    fragment_ref = write("scores/fragment.json", fragment)
    summary = write("score-summary.json", {})
    data = dict(split="val_locked", episode_sha256="split", label_protocol=label)
    parent_config = dict(model={"pilot_arm": "A3"}, optimizer={"seed": 0})
    parent = dict(
        run_dir=str(tmp_path / "training"),
        manifest=dict(run_id="training", git_sha="train-source", config=parent_config),
    )
    score = dict(
        run_dir=str(tmp_path / "scores"),
        manifest=dict(
            run_id="scores",
            git_sha="export-source",
            parent_run_ids=["training"],
            config=dict(
                model={"pilot_arm": "A3"},
                optimizer={"seed": 0},
                evaluation=dict(data=data, deployment=dict(checkpoint_sha256=binding["checkpoint"]["sha256"])),
            ),
        ),
    )
    context = SimpleNamespace(repo_root=tmp_path)
    bundle = dict(
        fragment_references=[fragment_ref],
        score_phase_summaries={"V0S": summary},
        groups=[group],
        label_protocol=label,
        relevance_space="benchmark_annotation",
        preprocessing_sha256="preprocessing",
        policy_sha256="policy",
        galleries={},
    )
    # Phase and output primitives already validate registry/lifecycle/inventory. This fixture
    # isolates cross-run identity composition, retaining real reserved-input hashes.
    monkeypatch.setattr(archive, "context_input_references", lambda _: refs)
    monkeypatch.setattr(archive, "_phase", lambda repo, ref, phase: {"runs": [score if phase == "V0S" else parent]})
    monkeypatch.setattr(archive, "_output", lambda run, ref: archive._reserved(context, ref))
    monkeypatch.setattr(archive, "manifest_sha256", lambda _: "manifest")
    cfg = SimpleNamespace(
        runtime=SimpleNamespace(eval_split="train", source_sha="train-source", run_id="training"),
        data=SimpleNamespace(label_protocol=label),
    )
    monkeypatch.setattr(archive, "load_system_config", lambda _: cfg)
    monkeypatch.setattr(archive, "effective_manifest_config", lambda *a, **kw: parent_config)
    monkeypatch.setattr(archive, "config_sha256", lambda _: "config")
    return context, bundle, parent, score, fragment, write


def test_archive_accepts_same_training_identity_across_export_source_changes(archived_fragment):
    context, bundle, *_ = archived_fragment
    archive.authenticate_score_fragments(context, bundle)


@pytest.mark.parametrize("field,value", [("pilot_arm", "A2"), ("seed", 1)])
def test_archive_rejects_training_arm_or_seed_relabel(archived_fragment, field, value):
    context, bundle, parent, *_ = archived_fragment
    section = "model" if field == "pilot_arm" else "optimizer"
    parent["manifest"]["config"][section][field] = value
    with pytest.raises(ValueError, match="relabeled"):
        archive.authenticate_score_fragments(context, bundle)


def test_archive_rejects_forged_producer_source(archived_fragment):
    context, bundle, _, _, fragment, write = archived_fragment
    fragment["source_sha"] = "different-source"
    bundle["fragment_references"] = [write("scores/fragment.json", fragment)]
    with pytest.raises(ValueError, match="producing run/source"):
        archive.authenticate_score_fragments(context, bundle)


def test_archive_rejects_unarchived_or_substituted_analysis_groups(archived_fragment):
    context, bundle, *_ = archived_fragment
    bundle["groups"] = [dict(bundle["groups"][0], model_sha256="other-model")]
    with pytest.raises(ValueError, match="exact archived"):
        archive.authenticate_score_fragments(context, bundle)
    with pytest.raises(ValueError, match="authenticated score producer"):
        archive.authenticate_score_fragments(context, dict(groups=[]))


def test_archive_rejects_substituted_global_protocol(archived_fragment):
    context, bundle, *_ = archived_fragment
    bundle["label_protocol"] = {"protocol_id": "substituted"}
    with pytest.raises(ValueError, match="global"):
        archive.authenticate_score_fragments(context, bundle)


def test_reserved_accepts_relative_control_summary_without_rewriting_git_reference(tmp_path, monkeypatch):
    summary = tmp_path / "experiments/semantic-answerability-pilot-v1/summaries/C0S.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}")
    reference = dict(path=summary.relative_to(tmp_path).as_posix(), sha256=file_sha256(summary), commit="a" * 40)
    original = dict(reference)
    context = SimpleNamespace(repo_root=tmp_path)
    monkeypatch.setattr(archive, "context_input_references", lambda _: {str(summary): reference["sha256"]})
    assert archive._reserved(context, reference) == summary
    assert reference == original
    with pytest.raises(ValueError, match="canonical"):
        archive._reserved(context, dict(reference, path="../outside-summary.json"))
