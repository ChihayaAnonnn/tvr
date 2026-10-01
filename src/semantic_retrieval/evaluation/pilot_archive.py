"""Archive authentication at pilot scoring and analysis boundaries."""

import json
from pathlib import Path

from experiment_artifacts import load_validated_run_lifecycle, make_run_paths, manifest_sha256
from semantic_retrieval.config import effective_manifest_config, load_system_config
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.runtime.checkpoint import config_sha256
from semantic_retrieval.runtime.input_bindings import context_input_references


def _phase(repository, reference, phase):
    from semantic_retrieval.runtime.pilot_protocol import verify_phase_evidence

    return verify_phase_evidence(repository, reference, expected_phase=phase)


def _reserved(context, reference):
    path = Path(reference["path"])
    if not path.is_absolute():
        # Git summary identity stays relative; only reserved-file lookup is absolute.
        if (
            len(path.parts) < 3
            or path.parts[0] != "experiments"
            or ".." in path.parts
            or path.as_posix() != reference["path"]
        ):
            raise ValueError("archive reference path is not canonical CONTROL-relative")
        path = Path(context.repo_root) / path
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("archive reference path is not canonical")
    if (
        context_input_references(context).get(str(path)) != reference["sha256"]
        or file_sha256(path) != reference["sha256"]
    ):
        raise ValueError("archive reference must be a hash-bound reserved input")
    return path


def _output(run, reference):
    root = Path(run["run_dir"])
    path = Path(reference["path"])
    if not path.is_relative_to(root) or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("producer output must belong to its authenticated run")
    paths = make_run_paths(root.parents[3], run["manifest"]["study"], run["manifest"]["run_id"])
    manifest, status, _ = load_validated_run_lifecycle(paths)
    if (
        manifest != run["manifest"]
        or status["state"] != "completed"
        or status["output_hashes"].get(path.relative_to(root).as_posix()) != reference["sha256"]
        or file_sha256(path) != reference["sha256"]
    ):
        raise ValueError("producer output differs from committed completed evidence")
    return path


def authenticate_training_parent(context, binding, cfg):
    arm, seed = cfg.model.pilot_arm, cfg.train.seed
    phase = ("P1a" if seed == 0 else "P2a") if arm in ("A0", "P1") else ("P1b" if seed == 0 else "P2b")
    if binding.get("phase") != phase:
        raise ValueError("training parent phase does not match arm and seed")
    _reserved(context, binding["summary"])
    evidence = _phase(context.repo_root, binding["summary"], phase)
    matches = [r for r in evidence["runs"] if r["manifest"]["run_id"] == binding["run_id"]]
    if len(matches) != 1 or binding["run_id"] not in context.manifest["parent_run_ids"]:
        raise ValueError("completed training parent must be an explicit reserved parent")
    run = matches[0]
    parent = load_system_config(_reserved(context, binding["config"]))
    if (
        parent.runtime.run_id != binding["run_id"]
        or parent.runtime.source_sha != run["manifest"]["git_sha"]
        or effective_manifest_config(parent, eval_split=parent.runtime.eval_split) != run["manifest"]["config"]
    ):
        raise ValueError("training configuration differs from authenticated producing manifest")
    if (
        parent.model != cfg.model
        or parent.train.seed != seed
        or parent.train.pilot is None
        or parent.train.pilot.arm != arm
        or parent.data is None
        or parent.data.split != "train"
        or parent.data.relevance_space != "benchmark_annotation"
        or dict(parent.data.label_protocol) != dict(cfg.data.label_protocol)
    ):
        raise ValueError("training parent arm/seed/model/label identity mismatch")
    checkpoint = _output(run, binding["checkpoint"])
    metadata = _output(run, binding["metadata"])
    _reserved(context, binding["checkpoint"])
    _reserved(context, binding["metadata"])
    if (
        checkpoint != cfg.deployment.checkpoint_path
        or binding["checkpoint"]["sha256"] != cfg.deployment.checkpoint_sha256
        or metadata != checkpoint.with_suffix(".json")
    ):
        raise ValueError("deployed checkpoint is not the authenticated training output")
    meta = json.loads(metadata.read_text())
    expected = config_sha256(parent)
    if (
        meta["sha256"] != binding["checkpoint"]["sha256"]
        or meta["config_sha256"] != expected
        or cfg.deployment.checkpoint_config_sha256 != expected
    ):
        raise ValueError("checkpoint producing config identity mismatch")
    return dict(
        binding,
        model_sha256=meta["model_sha256"],
        source_sha=run["manifest"]["git_sha"],
        manifest_sha256=manifest_sha256(run["manifest"]),
    )


def authenticate_score_fragments(context, bundle):
    references = bundle.get("fragment_references", [])
    summaries = bundle.get("score_phase_summaries", {})
    if not references or not summaries:
        raise ValueError("scientific analysis requires authenticated score producer fragments")
    runs = []
    for phase, reference in summaries.items():
        if phase not in ("C0S", "V0S"):
            raise ValueError("unknown score producer phase")
        _reserved(context, reference)
        runs.extend(_phase(context.repo_root, reference, phase)["runs"])
    collected = []
    for reference in references:
        path = _reserved(context, reference)
        owners = [r for r in runs if path.is_relative_to(Path(r["run_dir"]))]
        if len(owners) != 1:
            raise ValueError("score fragment has no unique authenticated producer")
        run = owners[0]
        _output(run, reference)
        fragment = json.loads(path.read_text())
        identity = ("relevance_space", "label_protocol", "preprocessing_sha256", "policy_sha256")
        if any(fragment[key] != bundle[key] for key in identity) or any(
            bundle["galleries"].get(name) != gallery for name, gallery in fragment["galleries"].items()
        ):
            raise ValueError("analysis global protocol/gallery differs from authenticated producer")
        producer = fragment.get("producer_identity")
        if (
            not isinstance(producer, dict)
            or producer.get("run_id") != run["manifest"]["run_id"]
            or producer.get("manifest_sha256") != manifest_sha256(run["manifest"])
            or fragment["source_sha"] != run["manifest"]["git_sha"]
        ):
            raise ValueError("score fragment producing run/source identity mismatch")
        config = run["manifest"]["config"]
        arm = config["model"]["pilot_arm"]
        seed = config["optimizer"]["seed"]
        data = config["evaluation"]["data"]
        binding = producer["training_parent"]
        expected_phase = ("P1a" if seed == 0 else "P2a") if arm in ("A0", "P1") else ("P1b" if seed == 0 else "P2b")
        if binding.get("phase") != expected_phase:
            raise ValueError("score fragment training phase differs from arm and seed")
        for field in ("summary", "config", "checkpoint", "metadata"):
            _reserved(context, binding[field])
        training = _phase(context.repo_root, binding["summary"], binding["phase"])
        parents = [r for r in training["runs"] if r["manifest"]["run_id"] == binding["run_id"]]
        if len(parents) != 1 or binding["run_id"] not in run["manifest"]["parent_run_ids"]:
            raise ValueError("score fragment training parent is not archived")
        parent = parents[0]
        if (
            parent["manifest"]["config"]["model"] != config["model"]
            or parent["manifest"]["config"]["optimizer"]["seed"] != seed
        ):
            raise ValueError("score producer relabeled a different training arm/seed")
        parent_config = load_system_config(Path(binding["config"]["path"]))
        if (
            effective_manifest_config(parent_config, eval_split=parent_config.runtime.eval_split)
            != parent["manifest"]["config"]
            or parent_config.runtime.source_sha != parent["manifest"]["git_sha"]
            or parent_config.runtime.run_id != binding["run_id"]
            or binding.get("source_sha") != parent["manifest"]["git_sha"]
            or binding.get("manifest_sha256") != manifest_sha256(parent["manifest"])
            or parent_config.data is None
            or parent_config.data.label_protocol != data["label_protocol"]
        ):
            raise ValueError("fragment parent config/source/label identity mismatch")
        _output(parent, binding["checkpoint"])
        _output(parent, binding["metadata"])
        meta = json.loads(Path(binding["metadata"]["path"]).read_text())
        if (
            meta["config_sha256"] != config_sha256(parent_config)
            or meta["sha256"] != binding["checkpoint"]["sha256"]
            or meta["model_sha256"] != binding["model_sha256"]
            or config["evaluation"]["deployment"]["checkpoint_sha256"] != binding["checkpoint"]["sha256"]
        ):
            raise ValueError("score producer checkpoint/model identity mismatch")
        for group in fragment["groups"]:
            if (
                group["arm"] != arm
                or group["seed"] != seed
                or group["model_sha256"] != meta["model_sha256"]
                or group["split"] != data["split"]
                or group["split_sha256"] != data["episode_sha256"]
                or fragment["label_protocol"] != data["label_protocol"]
            ):
                raise ValueError("fragment group differs from producing model/data protocol")
            _reserved(context, group["arrays"])
            _output(run, group["arrays"])
            collected.append(group)
    if sorted(collected, key=lambda g: (g["arm"], g["seed"], g["track"], g["mechanism"], g["K"])) != sorted(
        bundle["groups"], key=lambda g: (g["arm"], g["seed"], g["track"], g["mechanism"], g["K"])
    ):
        raise ValueError("analysis groups are not the exact archived producer fragments")
