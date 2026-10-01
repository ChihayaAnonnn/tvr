"""Archive-to-config handoff for the pilot's training and score-export batches."""

from dataclasses import asdict, replace
from pathlib import Path

from experiment_artifacts import strict_json_loads
from semantic_retrieval.config import (
    DataConfig,
    DeploymentConfig,
    ModelConfig,
    PilotTrainConfig,
    PolicyConfig,
    RuntimeConfig,
    SystemConfig,
    TrainConfig,
    effective_manifest_config,
    load_system_config,
)
from semantic_retrieval.models.clip import ClipConfig
from semantic_retrieval.runtime.checkpoint import config_sha256
from semantic_retrieval.runtime.input_bindings import write_input_indices
from semantic_retrieval.runtime.pilot import pilot_trainable_prefixes
from semantic_retrieval.runtime.pilot_candidates import KS, file_reference, write_json
from semantic_retrieval.runtime.pilot_evidence import (
    build_analysis_qualification,
)
from semantic_retrieval.runtime.pilot_evidence import (
    training_report_gates as training_report_gates,
)
from semantic_retrieval.runtime.pilot_evidence import (
    validate_analysis_qualification as validate_analysis_qualification,
)
from semantic_retrieval.runtime.pilot_protocol import (
    CONTROL,
    phase_contract,
    validate_frozen_prerequisite,
    validate_q0_report,
    verify_phase_evidence,
)


def _deduplicate(references):
    result = {}
    for ref in references:
        value = dict(path=str(ref["path"]), sha256=ref["sha256"])
        if value["path"] in result and result[value["path"]] != value:
            raise ValueError("workflow input has conflicting hashes")
        result[value["path"]] = value
    return list(result.values())


def _write_control_document(path, payload):
    write_json(path, payload)
    return file_reference(path)


def compact_input_references(references, *, index_path=None):
    """Only raw-video inventories enter indices; small scientific inputs stay direct."""
    refs = _deduplicate(references)
    if len(refs) <= 900:
        return refs
    if index_path is None:
        raise ValueError("large raw-video inventory requires a tracked input-index path")
    path = Path(index_path)
    if not path.is_absolute() or not path.name.endswith(".input-index.json"):
        raise ValueError("input-index path must be canonical and use the authenticated suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_input_indices(path.parent, path.name.removesuffix(".input-index.json"), refs)


def _placeholder_runtime(repository):
    repo = Path(repository).resolve()
    return RuntimeConfig(
        repo / CONTROL / "study.yaml",
        repo / CONTROL / "registry.jsonl",
        repo / "pending-pilot-run",
        "pending",
        "0" * 64,
        "0" * 40,
        "train",
    )


def training_configuration(arm, *, seed, candidates, parent=None, repo_root="/pending-pilot"):
    if arm not in {"A0", "P1", "A1", "A2", "A3", "A4"} or seed not in (0, 1, 2):
        raise ValueError("training configuration requires a registered arm/seed")
    if arm in {"A1", "A2", "A3", "A4"} and parent is None:
        raise ValueError("branch configuration requires a completed P1 parent")
    refs = candidates["splits"]["train"]
    payload = strict_json_loads(Path(refs["episodes"]["path"]).read_text())
    clip = ClipConfig(**candidates["clip"])
    model = ModelConfig(
        clip,
        Path(candidates["clip_weights"]["path"]),
        candidates["clip_weights"]["sha256"],
        clip.embed_dim,
        0,
        8,
        8,
        256,
        pilot_arm=arm,
    )
    parent_fields = (
        {}
        if parent is None
        else dict(
            parent_config_path=Path(parent["config"]["path"]),
            parent_config_sha256=parent["config"]["sha256"],
            parent_checkpoint_path=Path(parent["checkpoint"]["path"]),
            parent_checkpoint_sha256=parent["checkpoint"]["sha256"],
            parent_metadata_sha256=parent["metadata"]["sha256"],
            parent_run_id=parent["run_id"],
        )
    )
    sampler_seed = (
        seed if parent is None else load_system_config(Path(parent["config"]["path"])).train.pilot.sampler_seed
    )
    pilot = PilotTrainConfig(arm, sampler_seed, Path(refs["orders"]["path"]), refs["orders"]["sha256"], **parent_fields)
    train = TrainConfig(
        1e-4,
        0.01,
        seed,
        "constant",
        240 if arm == "A0" else 120,
        pilot_trainable_prefixes(arm),
        dict(coarse=1.0, match=1.0, pair=1.0, exist=float(arm in {"A2", "A3", "A4"}), align=0.0, group=0.0),
        20,
        pilot,
    )
    data = DataConfig(
        Path(refs["episodes"]["path"]),
        refs["episodes"]["sha256"],
        payload["gallery_version"],
        "train",
        "benchmark_annotation",
        payload["label_protocol"],
    )
    return SystemConfig(_placeholder_runtime(repo_root), model, train, PolicyConfig(0.5, 0.5, KS, 128), data)


def _archive_reference(run, relative):
    path = run["paths"].root / relative
    expected = run["status"]["output_hashes"].get(relative)
    if expected is None:
        raise ValueError("completed prerequisite lacks required workflow output: " + relative)
    ref = file_reference(path)
    if ref["sha256"] != expected:
        raise ValueError("workflow output differs from immutable terminal archive")
    return ref


def _candidate_result(evidence):
    if len(evidence["runs"]) != 1:
        raise ValueError("candidate handoff requires exactly one producer")
    ref = _archive_reference(evidence["runs"][0], "provenance/pilot-candidates.json")
    return strict_json_loads(Path(ref["path"]).read_text())


def _checkpoint_inputs(run):
    manifest = run["manifest"]
    arm = manifest["variant"]
    final = 240 if arm == "A0" else 120
    name = f"checkpoints/iteration-{final:08d}-step-{final:08d}"
    return dict(
        run_id=manifest["run_id"],
        config=_archive_reference(run, "provenance/system-config.json"),
        checkpoint=_archive_reference(run, name + ".pt"),
        metadata=_archive_reference(run, name + ".json"),
    )


def _data_references(episode_reference, *, gallery_reference=None):
    payload = strict_json_loads(Path(episode_reference["path"]).read_text())
    direct = [episode_reference] + list(payload["label_references"].values())
    if gallery_reference is None:
        videos = [
            dict(path=path, sha256=digest)
            for row in payload["episodes"]
            for path, digest in zip(row["video_paths"], row["content_sha256"])
        ]
    else:
        direct.append(gallery_reference)
        gallery = strict_json_loads(Path(gallery_reference["path"]).read_text())
        videos = [dict(path=row["video_path"], sha256=row["content_sha256"]) for row in gallery["videos"]]
    return _deduplicate(direct), _deduplicate(videos)


def _run_template(config, *, variant, seed, environment, parents, dataset_inputs, checkpoint_inputs, command):
    return dict(
        dataset="msrvtt",
        stage="baseline",
        variant=variant,
        seed=seed,
        environment=environment,
        input_checkpoints=_deduplicate(checkpoint_inputs),
        dataset_splits=_deduplicate(dataset_inputs),
        config=effective_manifest_config(config, eval_split=config.data.split),
        parent_run_ids=sorted(set(parents)),
        selection_lineage=[],
        validation_authorized=True,
        test_authorized=False,
        expected_scalar_outputs=[],
        array_references=[],
        command=command,
    )


def score_panel_document(reference, *, arm, seed, parent, candidate_orders, frozen=None):
    if file_reference(reference["path"]) != reference:
        raise ValueError("candidate panels changed before score handoff")
    document = strict_json_loads(Path(reference["path"]).read_text())
    if document.get("arm") != arm or document.get("seed") != seed:
        raise ValueError("candidate panel arm/seed differs from completed training parent")
    document["training_parent"] = parent
    document["candidate_orders"] = candidate_orders
    if frozen is not None:
        document["frozen"] = frozen
    return document


def workflow_templates(
    repo_root, phase_id, *, evidence, environment, python_executable, index_directory, gpu_uuids=None, clip_weights=None
):
    """Build a complete reserved batch from authenticated outputs, not hand-copied configs."""
    repo = Path(repo_root).resolve()
    contract = phase_contract(phase_id)
    if phase_id in {"Q0P", "Q0", "E0", "V0E"}:
        return producer_templates(
            repo,
            phase_id,
            evidence=evidence,
            environment=environment,
            python_executable=python_executable,
            directory=index_directory,
            gpu_uuids=gpu_uuids,
            clip_weights=clip_weights,
        )
    if phase_id in {"C0", "V0"}:
        return analysis_templates(
            repo,
            phase_id,
            evidence=evidence,
            environment=environment,
            python_executable=python_executable,
            directory=index_directory,
        )
    training = phase_id in {"P1a", "P1b", "P2a", "P2b"}
    if not training and phase_id not in {"C0S", "V0S"}:
        raise ValueError("workflow templates support registered training and score-export phases")
    required = set(contract["requires"])
    if not training:
        required |= {"P1a", "P1b", "P2a", "P2b"}
    authenticated = {name: verify_phase_evidence(repo, evidence.get(name), name) for name in sorted(required)}
    q0 = authenticated["Q0"]["runs"][0]
    report = strict_json_loads((q0["paths"].provenance / "pilot-qualification.json").read_text())
    validate_q0_report(report, source_sha=q0["manifest"]["git_sha"])
    uuids = [r["gpu_uuid"] for r in sorted(report["rank_records"], key=lambda r: r["rank"])]
    launch = [
        "env",
        "CUDA_VISIBLE_DEVICES=" + ",".join(uuids),
        f"PYTHONPATH={repo / 'src'}:{repo}",
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
    ]
    candidates = _candidate_result(authenticated["E0"])
    weights = candidates["clip_weights"]
    parents = [r["manifest"]["run_id"] for e in authenticated.values() for r in e["runs"]]
    outputs, final_parents, score_indices = [], {}, {}
    for name in ("P1a", "P1b", "P2a", "P2b"):
        for run in authenticated.get(name, {}).get("runs", []):
            final_parents[run["manifest"]["variant"], run["manifest"]["seed"]] = dict(
                _checkpoint_inputs(run), phase=name, summary=evidence[name]
            )
    locked = _candidate_result(authenticated["V0E"]) if phase_id == "V0S" else None
    frozen_ref = validate_frozen_prerequisite(repo, authenticated["C0"]) if phase_id == "V0S" else None
    if training:
        refs = candidates["splits"]["train"]
        direct, videos = _data_references(refs["episodes"])
        direct.append(refs["orders"])
        # One shared immutable inventory per whole batch; never index configs/checkpoints.
        indexed = compact_input_references(
            videos, index_path=Path(index_directory).resolve() / f"{phase_id}-videos.input-index.json"
        )
    for variant, seed in contract["runs"]:
        if training:
            parent = final_parents.get(("P1", seed)) if variant in {"A1", "A2", "A3", "A4"} else None
            cfg = training_configuration(variant, seed=seed, candidates=candidates, parent=parent, repo_root=repo)
            inputs = direct + indexed
            checkpoints = [weights]
            if parent:
                inputs = inputs + [parent["config"]]
                checkpoints += [parent["checkpoint"], parent["metadata"]]
            command = [
                str(python_executable),
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                "scripts/train_semantic_retrieval.py",
                "--config",
                "{artifact_run_dir}/provenance/system-config.json",
            ]
        else:
            arm, split = variant.split("-", 1)
            parent = final_parents[arm, seed]
            parent_cfg = load_system_config(Path(parent["config"]["path"]))
            refs = (locked if split == "val_locked" else candidates)["splits"][split]
            payload = strict_json_loads(Path(refs["episodes"]["path"]).read_text())
            data = DataConfig(
                Path(refs["episodes"]["path"]),
                refs["episodes"]["sha256"],
                payload["gallery_version"],
                split,
                "benchmark_annotation",
                payload["label_protocol"],
            )
            deployment = DeploymentConfig(
                Path(parent["checkpoint"]["path"]),
                parent["checkpoint"]["sha256"],
                config_sha256(parent_cfg),
                Path(refs["gallery"]["path"]),
                refs["gallery"]["sha256"],
                None,
                None,
                candidates["splits"]["val_calibration"]["episodes"]["sha256"],
                "calibrate",
            )
            cfg = replace(parent_cfg, runtime=_placeholder_runtime(repo), data=data, deployment=deployment, resume=None)
            base_panel = next(p["panels"] for p in refs["panels"] if p["arm"] == arm and p["seed"] == seed)
            panel_dir = Path(index_directory).resolve()
            if not panel_dir.is_relative_to(repo / CONTROL / "configs"):
                raise ValueError("score handoff must be tracked pilot control metadata")
            panel_dir.mkdir(parents=True, exist_ok=True)
            panel = _write_control_document(
                panel_dir / f"{arm}-seed{seed}-{split}-panels.json",
                score_panel_document(
                    base_panel, arm=arm, seed=seed, parent=parent, candidate_orders=refs["ranks"], frozen=frozen_ref
                ),
            )
            metadata, videos = _data_references(refs["episodes"], gallery_reference=refs["gallery"])
            # Evaluated500/1000 galleries are shared across all arm/seed workers.
            if split not in score_indices:
                score_indices[split] = compact_input_references(
                    videos, index_path=Path(index_directory).resolve() / f"{phase_id}-{split}-videos.input-index.json"
                )
            inputs = (
                metadata
                + score_indices[split]
                + [
                    panel,
                    refs["ranks"],
                    parent["config"],
                    dict(path=str(repo / parent["summary"]["path"]), sha256=parent["summary"]["sha256"]),
                ]
                + refs["array_inputs"]
            )
            if frozen_ref:
                inputs.append(frozen_ref)
                frozen = strict_json_loads(Path(frozen_ref["path"]).read_text())
                checkpoint_metadata = strict_json_loads(Path(parent["metadata"]["path"]).read_text())
                if frozen["model_identities"][f"{arm}_seed{seed}"] != checkpoint_metadata["model_sha256"]:
                    raise ValueError("locked checkpoint differs from C0 frozen model identity")
            checkpoints = [weights, parent["checkpoint"], parent["metadata"]]
            command = [
                str(python_executable),
                "scripts/export_semantic_pilot_scores.py",
                "--config",
                "{artifact_run_dir}/provenance/system-config.json",
                "--panels",
                panel["path"],
                "--panels-sha256",
                panel["sha256"],
            ]
        env = dict(environment, distributed_topology="torchrun-4" if training else "single-process")
        outputs.append(
            _run_template(
                cfg,
                variant=variant,
                seed=seed,
                environment=env,
                parents=parents,
                dataset_inputs=inputs,
                checkpoint_inputs=checkpoints,
                command=launch + command,
            )
        )
    return outputs


def producer_templates(
    repo_root, phase_id, *, evidence, environment, python_executable, directory, gpu_uuids=None, clip_weights=None
):
    from semantic_retrieval.runtime.pilot_candidates import candidate_manifest_config
    from semantic_retrieval.runtime.pilot_protocol import Q0_PROTOCOL

    repo = Path(repo_root).resolve()
    contract = phase_contract(phase_id)
    required = set(contract["requires"]) | ({"E0"} if phase_id == "V0E" else set())
    if not required.issubset(evidence):
        raise ValueError("producer factory is missing archived prerequisite evidence")
    authenticated = {name: verify_phase_evidence(repo, evidence[name], name) for name in sorted(required)}
    parents = sorted({r["manifest"]["run_id"] for e in authenticated.values() for r in e["runs"]})
    launch = [
        "env",
        f"PYTHONPATH={repo / 'src'}:{repo}",
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
    ]
    env = dict(environment, precision="fp32", distributed_topology="single-process")
    checkpoints, inputs = [], []
    if phase_id == "Q0P":
        env.update(gpu_count=0, gpu_model="none")
        launch.append("CUDA_VISIBLE_DEVICES=")
        config = dict(
            model={},
            loss={},
            optimizer={},
            scheduler={},
            evaluation=dict(eval_split="val", qualification_scenario="pilot_q0_prep_v2"),
        )
        command = [
            str(python_executable),
            "scripts/qualify_semantic_pilot.py",
            "--mode",
            "prepare-production",
            "--run-dir",
            "{artifact_run_dir}",
        ]
    elif phase_id == "Q0":
        if (
            not isinstance(gpu_uuids, list)
            or len(gpu_uuids) != 4
            or len(set(gpu_uuids)) != 4
            or any(not isinstance(x, str) or not x.startswith("GPU-") for x in gpu_uuids)
        ):
            raise ValueError("Q0 factory needs the four physical GPU UUIDs in registered order")
        prep = authenticated["Q0P"]["runs"][0]
        inventory = strict_json_loads(
            Path(_archive_reference(prep, "provenance/pilot-input-inventory.json")["path"]).read_text()
        )
        if (
            inventory.get("protocol") != Q0_PROTOCOL
            or inventory.get("software_only") is not True
            or inventory.get("random_weights") is not True
        ):
            raise ValueError("Q0 factory requires its new synthetic v2 archive")
        for ref in inventory["inputs"]:
            relative = Path(ref["path"]).relative_to(prep["paths"].root).as_posix()
            if _archive_reference(prep, relative) != ref:
                raise ValueError("Q0 inventory differs from its archived producer")
        cfg = load_system_config(Path(_archive_reference(prep, "provenance/pilot-system-template.json")["path"]))
        if cfg.model.clip.fixture:
            raise ValueError("Q0 factory refuses a CPU capacity fixture")
        weights = dict(path=str(cfg.model.clip_weights), sha256=cfg.model.clip_sha256)
        checkpoints = [weights]
        inputs = [r for r in inventory["inputs"] if r != weights]
        config = effective_manifest_config(cfg, eval_split="train")
        env.update(gpu_count=4, distributed_topology="torchrun-4")
        launch.append("CUDA_VISIBLE_DEVICES=" + ",".join(gpu_uuids))
        command = [
            str(python_executable),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "scripts/qualify_semantic_pilot.py",
            "--mode",
            "four-gpu",
            "--run-dir",
            "{artifact_run_dir}",
            "--config",
            "{artifact_run_dir}/provenance/system-config.json",
        ]
    else:
        q0 = authenticated["Q0"]["runs"][0]
        report = strict_json_loads(
            Path(_archive_reference(q0, "provenance/pilot-qualification.json")["path"]).read_text()
        )
        validate_q0_report(report, source_sha=q0["manifest"]["git_sha"])
        launch.append(
            "CUDA_VISIBLE_DEVICES="
            + ",".join(r["gpu_uuid"] for r in sorted(report["rank_records"], key=lambda r: r["rank"]))
        )
        env.update(gpu_count=4)
        if phase_id == "V0E":
            candidates = _candidate_result(authenticated["E0"])
            clip, weights = candidates["clip"], candidates["clip_weights"]
            frozen = validate_frozen_prerequisite(repo, authenticated["C0"])
            inputs.append(frozen)
        else:
            if clip_weights is None or file_reference(clip_weights["path"]) != clip_weights:
                raise ValueError("E0 factory needs explicitly hash-bound scientific pretrained CLIP weights")
            clip, weights = asdict(ClipConfig(512, 224, 768, 12, 32, 77, 49408, 512, 8, 12)), clip_weights
        checkpoints = [weights]
        d1 = authenticated["D1"]["runs"][0]
        result = strict_json_loads(
            Path(_archive_reference(d1, "provenance/preparation-result.json")["path"]).read_text()
        )
        splits = ("val_locked",) if phase_id == "V0E" else ("train", "val_development", "val_calibration")
        directory = Path(directory).resolve()
        if not directory.is_relative_to(repo / CONTROL / "configs"):
            raise ValueError("candidate handoff must be tracked pilot control metadata")
        directory.mkdir(parents=True, exist_ok=True)
        body = dict(schema_version=1, clip=clip, weights=weights, splits={sp: result["manifests"][sp] for sp in splits})
        videos = []
        for sp, refs in body["splits"].items():
            for ref in refs.values():
                relative = Path(ref["path"]).relative_to(d1["paths"].root).as_posix()
                if _archive_reference(d1, relative) != ref:
                    raise ValueError("candidate input differs from D1 archive")
            direct, raw = _data_references(refs["episodes"], gallery_reference=refs["gallery"])
            inputs += direct
            videos += raw
        inputs += compact_input_references(videos, index_path=directory / f"{phase_id}-videos.input-index.json")
        body_ref = _write_control_document(directory / "candidate-config.json", body)
        inputs.append(body_ref)
        config = candidate_manifest_config(body, phase_id=phase_id)
        command = [
            str(python_executable),
            "scripts/prepare_semantic_pilot_candidates.py",
            "--run-dir",
            "{artifact_run_dir}",
            "--config",
            body_ref["path"],
        ]
    return [
        dict(
            dataset=contract["dataset"],
            stage="baseline",
            variant=contract["runs"][0][0],
            seed=0,
            environment=env,
            input_checkpoints=_deduplicate(checkpoints),
            dataset_splits=_deduplicate(inputs),
            config=config,
            parent_run_ids=parents,
            selection_lineage=[],
            validation_authorized=True,
            test_authorized=False,
            expected_scalar_outputs=[],
            array_references=[],
            command=launch + command,
        )
    ]


def analysis_templates(repo_root, phase_id, *, evidence, environment, python_executable, directory):
    """Materialize the complete CPU analysis batch from authenticated scientific outputs."""
    from semantic_retrieval.evaluation.pilot_analysis import build_pilot_bundle
    from semantic_retrieval.runtime.pilot_evidence import BASE_PHASES

    repo = Path(repo_root).resolve()
    required = set(BASE_PHASES) | ({"C0", "V0E", "V0S"} if phase_id == "V0" else set())
    chain = {name: evidence[name] for name in required if name in evidence}
    document, measured_inputs, authenticated = build_analysis_qualification(repo, chain)
    directory = Path(directory).resolve()
    if not directory.is_relative_to(repo / CONTROL / "configs"):
        raise ValueError("analysis handoff must be immutable tracked pilot control metadata")
    directory.mkdir(parents=True, exist_ok=True)
    fragments, inputs, checkpoints = [], list(measured_inputs), []
    score_phases = ("V0S",) if phase_id == "V0" else ("C0S",)
    for name in score_phases:
        if name not in authenticated:
            continue
        for run in authenticated[name]["runs"]:
            ref = _archive_reference(run, "provenance/pilot_score_fragment.json")
            fragment = strict_json_loads(Path(ref["path"]).read_text())
            fragments.append(ref)
            inputs.append(ref)
            for group in fragment["groups"]:
                reference = group["arrays"]
                relative = Path(reference["path"]).relative_to(run["paths"].root).as_posix()
                actual = _archive_reference(run, relative)
                if any(reference[k] != actual[k] for k in ("path", "sha256")):
                    raise ValueError("score array differs from its terminal producer inventory")
                inputs.append(actual)
    d1 = authenticated["D1"]["runs"][0]
    preparation = strict_json_loads(
        Path(_archive_reference(d1, "provenance/preparation-result.json")["path"]).read_text()
    )
    reference = preparation["manifests"]["val_locked"]["gallery"]
    gallery = strict_json_loads(Path(reference["path"]).read_text())
    aliases = {}
    for row in gallery["videos"]:
        aliases.setdefault(row["canonical_id"], []).append(row["alias_id"])
    locked_gallery = dict(
        version=gallery["gallery_version"],
        manifest_sha256=reference["sha256"],
        canonical_ids=sorted(aliases),
        aliases={k: sorted(v) for k, v in sorted(aliases.items())},
    )
    for name in ("P1a", "P1b", "P2a", "P2b"):
        for run in authenticated[name]["runs"]:
            parent = _checkpoint_inputs(run)
            inputs.append(parent["config"])
            checkpoints += [parent["checkpoint"], parent["metadata"]]
    frozen = validate_frozen_prerequisite(repo, authenticated["C0"]) if phase_id == "V0" else None
    if frozen:
        inputs.append(frozen)
    gate_ref = _write_control_document(directory / "qualification.json", document)
    bundle = build_pilot_bundle(fragments, locked_gallery=locked_gallery, frozen=frozen, qualification=gate_ref)
    bundle["score_phase_summaries"] = {name: chain[name] for name in score_phases}
    bundle_ref = _write_control_document(directory / "bundle.json", bundle)
    config_ref = _write_control_document(
        directory / "analysis-config.json",
        dict(bundle=bundle_ref, mode="locked" if phase_id == "V0" else "freeze", bootstrap_samples=10000),
    )
    split = "val_locked" if phase_id == "V0" else "val_development"
    env = dict(environment, gpu_count=0, gpu_model="none", precision="fp32", distributed_topology="single-process")
    command = [
        "env",
        "CUDA_VISIBLE_DEVICES=",
        f"PYTHONPATH={repo / 'src'}:{repo}",
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        str(python_executable),
        "scripts/analyze_semantic_pilot.py",
        "--run-dir",
        "{artifact_run_dir}",
        "--config",
        config_ref["path"],
    ]
    return [
        dict(
            dataset="msrvtt",
            stage="baseline",
            variant=phase_contract(phase_id)["runs"][0][0],
            seed=0,
            environment=env,
            input_checkpoints=_deduplicate(checkpoints),
            dataset_splits=_deduplicate(inputs + [gate_ref, bundle_ref, config_ref]),
            config=dict(
                model={},
                loss={},
                optimizer={},
                scheduler={},
                evaluation=dict(eval_split=split, qualification_scenario="pilot_analysis"),
            ),
            parent_run_ids=sorted({r["manifest"]["run_id"] for e in authenticated.values() for r in e["runs"]}),
            selection_lineage=[],
            validation_authorized=True,
            test_authorized=False,
            expected_scalar_outputs=[],
            array_references=[],
            command=command,
        )
    ]
