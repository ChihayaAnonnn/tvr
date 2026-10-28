import copy
from types import SimpleNamespace

import pytest
from test_pilot_protocol import pilot_repo


def archive_batch(repo, phase, producer):
    """CPU-only test archival: use actual manager/lifecycle, never execute a GPU worker."""
    import hashlib
    from pathlib import Path
    from experiment_artifacts import (
        append_registry_records,
        artifact_json_bytes,
        hash_run_outputs,
        make_run_paths,
        reserve_run_directory,
        transition_run_status,
    )
    from scripts import manage_experiment_run as manager
    from semantic_retrieval.runtime.pilot_protocol import CONTROL, PILOT, phase_contract
    from test_pilot_protocol import template
    from test_runtime_artifacts import git

    contract = phase_contract(phase)
    specs = []
    for variant, seed in contract["runs"]:
        spec = template(variant, seed, 4 if phase == "Q0" else 0)
        spec["dataset"] = contract["dataset"]
        if phase in ("C0", "V0"):
            spec["config"]["evaluation"] = dict(
                eval_split=contract["eval_split"], qualification_scenario="pilot_analysis"
            )
        specs.append(spec)
    path = Path(CONTROL) / "phases" / f"{phase}.json"
    (repo / path).parent.mkdir(exist_ok=True)
    (repo / path).write_bytes(artifact_json_bytes(dict(phase=phase, study=PILOT, runs=specs)))
    git(repo, "add", str(path))
    git(repo, "commit", "-qm", "test phase")
    source = git(repo, "rev-parse", "HEAD")
    runs = []
    for spec in specs:
        manifest = manager._build_manifest(
            spec,
            study=PILOT,
            git_sha=source,
            artifact_root=repo / "artifacts",
            registry=Path(CONTROL) / "registry.jsonl",
            common_root=repo,
        )
        paths = make_run_paths(repo / "artifacts", PILOT, manifest["run_id"])
        reserve_run_directory(paths, manifest)
        paths.command.write_bytes(
            manager._render_command(spec["command"], run_id=manifest["run_id"], run_dir=paths.root)
        )
        append_registry_records(repo / CONTROL / "registry.jsonl", [manager._registry_record(manifest, "reserved")])
        runs.append((manifest, paths))
    git(repo, "add", f"{CONTROL}/registry.jsonl")
    git(repo, "commit", "-qm", "test reserved batch")
    for manifest, paths in runs:
        transition_run_status(paths, "reserved", "running")
        producer(manifest, paths)
        transition_run_status(paths, "running", "validating")
        transition_run_status(paths, "validating", "completed", exit_code=0, output_hashes=hash_run_outputs(paths))
    summary = Path(CONTROL) / "summaries" / f"{phase}.json"
    manager.finalize_phase(
        SimpleNamespace(
            repo_root=repo,
            phase=path,
            registry=Path(CONTROL) / "registry.jsonl",
            artifact_root=repo / "artifacts",
            summary=summary,
        )
    )
    git(repo, "add", str(summary), f"{CONTROL}/registry.jsonl")
    git(repo, "commit", "-qm", "test terminal batch")
    return dict(
        path=summary.as_posix(),
        sha256=hashlib.sha256((repo / summary).read_bytes()).hexdigest(),
        commit=git(repo, "rev-parse", "HEAD"),
    )


@pytest.fixture
def analysis_chain(pilot_repo):
    import json
    from semantic_retrieval.runtime.pilot_candidates import file_reference, write_json
    from semantic_retrieval.runtime.pilot_protocol import Q0_CHECKS, Q0_PROTOCOL
    from semantic_retrieval.data.pilot_catalog import TARGETS

    protocol = {"protocol_id": "cpu-test-only"}

    def d1(manifest, paths):
        splits = {
            sp: dict(
                official_video_count=TARGETS["train_min"] if sp == "train" else TARGETS[sp],
                canonical_video_count=129,
                query_count=TARGETS["train_min"] if sp == "train" else TARGETS[sp],
                component_count=1000,
            )
            for sp in ("train", "val_development", "val_calibration", "val_locked")
        }
        summary = dict(ready=True, relevance_space="benchmark_annotation", label_protocol=protocol, splits=splits)
        manifests = {}
        raw = paths.provenance / "raw.avi"
        raw.write_bytes(b"CPU metadata fixture")
        relation = paths.provenance / "relation.json"
        write_json(relation, {"q": ["official"]})
        for sp in splits:
            gallery_path = paths.provenance / (sp + "-gallery.json")
            write_json(
                gallery_path,
                dict(
                    gallery_version="locked-test" if sp == "val_locked" else sp,
                    videos=[
                        dict(
                            canonical_id="v",
                            alias_id="official",
                            video_path=str(raw),
                            content_sha256=file_reference(raw)["sha256"],
                        )
                    ],
                ),
            )
            episode = paths.provenance / (sp + "-episodes.json")
            write_json(episode, dict(label_references={"relations": file_reference(relation)}, episodes=[]))
            manifests[sp] = dict(gallery=file_reference(gallery_path), episodes=file_reference(episode))
        write_json(paths.provenance / "input-summary.json", summary)
        write_json(paths.provenance / "preparation-result.json", dict(manifests=manifests))

    evidence = {"D1": archive_batch(pilot_repo, "D1", d1)}

    def q0(manifest, paths):
        write_json(
            paths.provenance / "pilot-qualification.json",
            dict(
                schema_version=1,
                protocol=Q0_PROTOCOL,
                source_sha=manifest["git_sha"],
                world_size=4,
                device="cuda",
                precision="fp32",
                tf32=False,
                synthetic_only=True,
                physical_indices=[0, 2, 3, 4],
                checks={k: True for k in Q0_CHECKS},
                rank_records=[dict(rank=r, gpu_uuid=f"GPU-CPU-TEST-{r}") for r in range(4)],
                seconds_per_update=0.1,
                resume_parameter_max_abs_delta=0.0,
                resume_optimizer_max_abs_delta=0.0,
            ),
        )

    evidence["Q0"] = archive_batch(pilot_repo, "Q0", q0)
    evidence["E0"] = archive_batch(pilot_repo, "E0", lambda m, p: None)
    reports = {(r["arm"], r["seed"]): r for r in training_reports()}

    def training(manifest, paths):
        report = reports[manifest["variant"], manifest["seed"]]
        for name, rows in (("losses", report["losses"]), ("gradients", report["gradients"])):
            (paths.logs / f"pilot-{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        steps = 240 if manifest["variant"] == "A0" else 120
        name = f"iteration-{steps:08d}-step-{steps:08d}"
        (paths.checkpoints / (name + ".pt")).write_bytes(b"CPU metadata test; never executed")
        write_json(paths.checkpoints / (name + ".json"), {})
        write_json(paths.provenance / "system-config.json", {})

    for phase in ("P1a", "P1b", "P2a", "P2b"):
        evidence[phase] = archive_batch(pilot_repo, phase, training)

    def scores(manifest, paths):
        arm, split = manifest["variant"].split("-", 1)
        array = paths.arrays / "fixture.npz"
        import numpy as np

        np.savez(array, pair_logits=np.zeros((1, 1)))
        group = dict(
            arm=arm,
            seed=manifest["seed"],
            split=split,
            track="natural",
            mechanism="fixed",
            K=32,
            gallery=split,
            model_sha256="a" * 64,
            split_sha256="b" * 64,
            arrays=dict(file_reference(array), format="npz"),
        )
        gallery = dict(version=split, manifest_sha256="c" * 64, canonical_ids=["v"], aliases={"v": ["official"]})
        write_json(
            paths.provenance / "pilot_score_fragment.json",
            dict(
                schema_version=1,
                relevance_space="benchmark_annotation",
                label_protocol=protocol,
                preprocessing_sha256="d" * 64,
                policy_sha256="e" * 64,
                groups=[group],
                galleries={split: gallery},
                source_sha=manifest["git_sha"],
                evaluation_gpu_hours=0.0,
                inference_wall_seconds=0.1,
            ),
        )

    evidence["C0S"] = archive_batch(pilot_repo, "C0S", scores)
    return pilot_repo, evidence


def test_analysis_factory_reserves_complete_archive_handoff_and_recomputed_gates(analysis_chain):
    import json
    from semantic_retrieval.runtime.pilot_workflow import workflow_templates, validate_analysis_qualification
    from semantic_retrieval.runtime.pilot_protocol import CONTROL
    from test_runtime_artifacts import git

    repo, evidence = analysis_chain
    templates = workflow_templates(
        repo,
        "C0",
        evidence=evidence,
        environment={},
        python_executable="/python",
        index_directory=repo / CONTROL / "configs" / "C0-inputs",
    )
    assert len(templates) == 1
    run = templates[0]
    refs = {r["path"]: r["sha256"] for r in run["dataset_splits"] + run["input_checkpoints"]}
    config = json.loads((repo / CONTROL / "configs" / "C0-inputs" / "analysis-config.json").read_text())
    assert config["mode"] == "freeze" and config["bootstrap_samples"] == 10000
    bundle = json.loads(__import__("pathlib").Path(config["bundle"]["path"]).read_text())
    assert len(bundle["fragment_references"]) == 30 and bundle["score_phase_summaries"] == {"C0S": evidence["C0S"]}
    assert all(refs[g["arrays"]["path"]] == g["arrays"]["sha256"] for g in bundle["groups"])
    document = json.loads(__import__("pathlib").Path(bundle["qualification"]["path"]).read_text())
    assert document["gates"]["stability"] and document["gates"]["gradients"]
    assert document["gates"]["invariance"] is False
    manifest = dict(
        git_sha=git(repo, "rev-parse", "HEAD"),
        dataset_splits=run["dataset_splits"],
        input_checkpoints=run["input_checkpoints"],
        selection_lineage=[dict(kind="pilot_phase", protocol_version=1, phase_id="C0", evidence=evidence)],
    )
    context = SimpleNamespace(repo_root=repo, manifest=manifest, input_references=None)
    assert validate_analysis_qualification(context, document) == document["gates"]
    forged = copy.deepcopy(document)
    forged["gates"]["invariance"] = True
    with pytest.raises(ValueError, match="recomputation"):
        validate_analysis_qualification(context, forged)
    context.input_references = None
    manifest["dataset_splits"] = [r for r in manifest["dataset_splits"] if not r["path"].endswith("pilot-losses.jsonl")]
    with pytest.raises(ValueError, match="unreserved"):
        validate_analysis_qualification(context, document)


def test_locked_analysis_factory_binds_completed_freeze_and_both_score_batches(analysis_chain):
    import json
    from pathlib import Path
    import numpy as np
    from semantic_retrieval.evaluation import pilot_analysis
    from semantic_retrieval.runtime.pilot_candidates import write_json, file_reference
    from semantic_retrieval.runtime.pilot_evidence import build_analysis_qualification
    from semantic_retrieval.runtime.pilot_workflow import workflow_templates, validate_analysis_qualification
    from semantic_retrieval.runtime.pilot_protocol import CONTROL
    from test_runtime_artifacts import git

    repo, evidence = analysis_chain
    doc, _, authenticated = build_analysis_qualification(repo, evidence)
    d1 = authenticated["D1"]["runs"][0]
    preparation = json.loads((d1["paths"].provenance / "preparation-result.json").read_text())
    target = dict(
        version="locked-test",
        manifest_sha256=preparation["manifests"]["val_locked"]["gallery"]["sha256"],
        canonical_ids=["v"],
        aliases={"v": ["official"]},
    )

    def freeze(manifest, paths):
        write_json(
            paths.provenance / "pilot_frozen.json",
            dict(
                schema_version=1,
                relevance_space="benchmark_annotation",
                source_sha=manifest["git_sha"],
                analysis_sha256=pilot_analysis.file_sha256(pilot_analysis.__file__),
                label_protocol=doc["label_protocol"],
                target_gallery=target,
                model_identities={f"{a}_seed{s}": "a" * 64 for a in ("A0", "A1", "A2", "A3", "A4") for s in (0, 1, 2)},
                policies={"test": {}},
                scalar_policies={"test": {}},
                calibration={"test": {}},
                strongest_scalar={"test": {}},
                qualification_gates={k: True for k in ("data", "four_gpu_qualification", "gradients", "stability")},
                development_gate_passed=True,
            ),
        )

    evidence["C0"] = archive_batch(repo, "C0", freeze)
    evidence["V0E"] = archive_batch(repo, "V0E", lambda m, p: None)

    def scores(manifest, paths):
        array = paths.arrays / "fixture.npz"
        np.savez(array, pair_logits=np.zeros((1, 1)))
        group = dict(
            arm=manifest["variant"].split("-")[0],
            seed=manifest["seed"],
            split="val_locked",
            track="natural",
            mechanism="fixed",
            K=32,
            gallery="locked-test",
            model_sha256="a" * 64,
            split_sha256="b" * 64,
            arrays=dict(file_reference(array), format="npz"),
        )
        write_json(
            paths.provenance / "pilot_score_fragment.json",
            dict(
                relevance_space="benchmark_annotation",
                label_protocol=doc["label_protocol"],
                preprocessing_sha256="d" * 64,
                policy_sha256="e" * 64,
                galleries={"locked-test": target},
                groups=[group],
                inference_wall_seconds=0.1,
                evaluation_gpu_hours=0.0,
            ),
        )

    evidence["V0S"] = archive_batch(repo, "V0S", scores)
    run = workflow_templates(
        repo,
        "V0",
        evidence=evidence,
        environment={},
        python_executable="/python",
        index_directory=repo / CONTROL / "configs" / "V0-inputs",
    )[0]
    config = json.loads((repo / CONTROL / "configs" / "V0-inputs" / "analysis-config.json").read_text())
    bundle = json.loads(Path(config["bundle"]["path"]).read_text())
    assert config["mode"] == "locked"
    assert len(bundle["fragment_references"]) == 15
    assert bundle["score_phase_summaries"] == {"V0S": evidence["V0S"]}
    assert all(g["split"] == "val_locked" for g in bundle["groups"])
    assert bundle["frozen"] in run["dataset_splits"]
    gate = json.loads(Path(bundle["qualification"]["path"]).read_text())
    context = SimpleNamespace(
        repo_root=repo,
        input_references=None,
        manifest=dict(
            git_sha=git(repo, "rev-parse", "HEAD"),
            dataset_splits=run["dataset_splits"],
            input_checkpoints=run["input_checkpoints"],
            selection_lineage=[dict(kind="pilot_phase", protocol_version=1, phase_id="V0", evidence=evidence)],
        ),
    )
    assert validate_analysis_qualification(context, gate)["invariance"] is False


def test_q0_producer_factory_consumes_exact_archived_v2_inventory(analysis_chain):
    from dataclasses import asdict
    from semantic_retrieval.config import config_payload
    from semantic_retrieval.models.clip import ClipConfig
    from semantic_retrieval.runtime.pilot_candidates import write_json, file_reference
    from semantic_retrieval.runtime.pilot_protocol import CONTROL, Q0_PROTOCOL
    from semantic_retrieval.runtime.pilot_workflow import workflow_templates, training_configuration

    repo, evidence = analysis_chain
    captured = {}

    def prepare(manifest, paths):
        episode = paths.provenance / "episodes.json"
        orders = paths.provenance / "orders.json"
        weights = paths.provenance / "clip.npz"
        weights.write_bytes(b"capacity metadata test, never loaded")
        protocol = dict(
            protocol_id="software-only",
            **{
                k: "a" * 64
                for k in (
                    "annotation_sha256",
                    "relation_sha256",
                    "id_map_sha256",
                    "source_report_sha256",
                    "allowlist_sha256",
                    "source_annotation_sha256",
                )
            },
        )
        write_json(episode, dict(gallery_version="synthetic-v2", label_protocol=protocol))
        write_json(orders, {"q": ["v"]})
        candidates = dict(
            clip=asdict(ClipConfig(512, 224, 768, 12, 32, 77, 49408, 512, 8, 12)),
            clip_weights=file_reference(weights),
            splits={"train": dict(episodes=file_reference(episode), orders=file_reference(orders))},
        )
        cfg = training_configuration("P1", seed=0, candidates=candidates, repo_root=repo)
        template = paths.provenance / "pilot-system-template.json"
        write_json(template, config_payload(cfg))
        refs = [file_reference(p) for p in (weights, episode, orders, template)]
        captured["refs"] = refs
        write_json(
            paths.provenance / "pilot-input-inventory.json",
            dict(protocol=Q0_PROTOCOL, software_only=True, random_weights=True, inputs=refs),
        )

    evidence["Q0P"] = archive_batch(repo, "Q0P", prepare)
    run = workflow_templates(
        repo,
        "Q0",
        evidence=evidence,
        environment={"gpu_model": "A800"},
        python_executable="/python",
        index_directory=repo / CONTROL / "configs" / "Q0-inputs",
        gpu_uuids=[f"GPU-TEST-{i}" for i in range(4)],
    )[0]
    assert sorted(run["input_checkpoints"] + run["dataset_splits"], key=lambda r: r["path"]) == sorted(
        captured["refs"], key=lambda r: r["path"]
    )
    assert run["environment"]["gpu_count"] == 4 and run["environment"]["distributed_topology"] == "torchrun-4"
    assert run["config"]["model"]["pilot_arm"] == "P1"
    assert "--nproc_per_node=4" in run["command"]


def test_e0_factory_binds_d1_gallery_and_explicit_scientific_weights(analysis_chain):
    import json
    from semantic_retrieval.runtime.pilot_candidates import file_reference, candidate_manifest_config
    from semantic_retrieval.runtime.pilot_protocol import CONTROL
    from semantic_retrieval.runtime.pilot_workflow import workflow_templates

    repo, evidence = analysis_chain
    weights = repo / "pretrained.npz"
    weights.write_bytes(b"controller metadata fixture, never loaded")
    run = workflow_templates(
        repo,
        "E0",
        evidence=evidence,
        environment={"gpu_model": "A800"},
        python_executable="/python",
        index_directory=repo / CONTROL / "configs" / "E0-inputs",
        clip_weights=file_reference(weights),
    )[0]
    body_path = repo / CONTROL / "configs" / "E0-inputs" / "candidate-config.json"
    body = json.loads(body_path.read_text())
    assert set(body["splits"]) == {"train", "val_development", "val_calibration"}
    assert body["weights"] == file_reference(weights) and run["input_checkpoints"] == [file_reference(weights)]
    assert file_reference(body_path) in run["dataset_splits"]
    assert run["config"] == candidate_manifest_config(body, phase_id="E0")
    assert run["environment"]["gpu_count"] == 4


def test_c3_consumer_uses_real_committed_c4_summary_and_output_contract(analysis_chain):
    import json
    from pathlib import Path
    from semantic_retrieval.evaluation.pilot_archive import _reserved, _phase, _output, authenticate_score_fragments
    from semantic_retrieval.runtime.pilot_protocol import CONTROL
    from semantic_retrieval.runtime.pilot_workflow import workflow_templates

    repo, evidence = analysis_chain
    run = workflow_templates(
        repo,
        "C0",
        evidence=evidence,
        environment={},
        python_executable="/python",
        index_directory=repo / CONTROL / "configs" / "C3-boundary",
    )[0]
    context = SimpleNamespace(
        repo_root=repo,
        input_references=None,
        manifest=dict(dataset_splits=run["dataset_splits"], input_checkpoints=run["input_checkpoints"]),
    )
    summary = evidence["C0S"]
    assert _reserved(context, summary) == repo / summary["path"]
    archived = _phase(repo, summary, "C0S")
    producer = archived["runs"][0]
    assert producer["run_dir"] == str(producer["paths"].root)
    configuration = json.loads((repo / CONTROL / "configs" / "C3-boundary" / "analysis-config.json").read_text())
    bundle = json.loads(Path(configuration["bundle"]["path"]).read_text())
    reference = bundle["fragment_references"][0]
    assert _output(producer, reference) == Path(reference["path"])
    # Metadata-only software fixtures must still reject at model producer identity.
    with pytest.raises(ValueError, match="producing run/source identity"):
        authenticate_score_fragments(context, bundle)


@pytest.mark.parametrize("defect", ["missing", "gradient_false", "stability_false", "unreleased", "extra"])
def test_archived_freeze_cannot_unlock_failed_or_missing_development_gates(pilot_repo, defect):
    from semantic_retrieval.evaluation import pilot_analysis
    from semantic_retrieval.runtime.pilot_candidates import write_json
    from semantic_retrieval.runtime.pilot_protocol import validate_frozen_prerequisite, verify_phase_evidence

    def freeze(manifest, paths):
        gates = {k: True for k in ("data", "four_gpu_qualification", "gradients", "stability")}
        if defect == "gradient_false":
            gates["gradients"] = False
        if defect == "stability_false":
            gates["stability"] = False
        if defect == "extra":
            gates["invariance"] = True
        payload = dict(
            schema_version=1,
            relevance_space="benchmark_annotation",
            source_sha=manifest["git_sha"],
            analysis_sha256=pilot_analysis.file_sha256(pilot_analysis.__file__),
            label_protocol={"protocol_id": "fixture"},
            target_gallery={"version": "fixture"},
            model_identities={
                f"{arm}_seed{seed}": "a" * 64 for arm in ("A0", "A1", "A2", "A3", "A4") for seed in (0, 1, 2)
            },
            policies={"fixture": {}},
            scalar_policies={"fixture": {}},
            calibration={"fixture": {}},
            strongest_scalar={"fixture": {}},
            qualification_gates=gates,
            development_gate_passed=defect != "unreleased",
        )
        if defect == "missing":
            payload.pop("qualification_gates")
        write_json(paths.provenance / "pilot_frozen.json", payload)

    reference = archive_batch(pilot_repo, "C0", freeze)
    evidence = verify_phase_evidence(pilot_repo, reference, "C0")
    with pytest.raises(ValueError, match="development gates"):
        validate_frozen_prerequisite(pilot_repo, evidence)


def training_reports():
    from semantic_retrieval.runtime.pilot import PilotSampler
    from semantic_retrieval.objectives.retrieval import LOSS_NAMES

    reports = []
    for seed in (0, 1, 2):
        for arm in ("A0", "P1", "A1", "A2", "A3", "A4"):
            steps = 240 if arm == "A0" else 120
            sampler = PilotSampler([f"q{i}" for i in range(32)], seed=seed)
            losses = []
            for iteration in range(1, steps + 1):
                _, k, panel = sampler.next_batch()
                losses.append(
                    dict(
                        iteration=iteration,
                        step=iteration,
                        stepped=1,
                        k=k,
                        panel=panel,
                        **{name: 0.1 for name in (*LOSS_NAMES, "total")},
                        forward_seconds=0.1,
                        backward_seconds=0.1,
                        optimizer_seconds=0.1,
                    )
                )
            gradients = [
                dict(
                    iteration=i,
                    matching_gradient_norm=1.0,
                    presence_gradient_norm=1.0 if arm == "A3" else 0.0,
                    gradient_cosine=0.2 if arm == "A3" else None,
                )
                for i in range(20, steps + 1, 20)
            ]
            reports.append(dict(arm=arm, seed=seed, losses=losses, gradients=gradients))
    return reports


def test_training_gates_require_actual_complete_steps_and_detach_measurements():
    from semantic_retrieval.runtime.pilot_workflow import training_report_gates

    reports = training_reports()
    result = training_report_gates(reports)
    assert result["stability"]["passed"] and result["gradients"]["passed"]
    assert result["stability"]["run_count"] == 18
    bad = copy.deepcopy(reports)
    bad[0]["losses"].pop()
    assert not training_report_gates(bad)["stability"]["passed"]
    bad = copy.deepcopy(reports)
    next(r for r in bad if r["arm"] == "A2")["gradients"][0]["presence_gradient_norm"] = 1.0
    assert not training_report_gates(bad)["gradients"]["passed"]


@pytest.mark.parametrize("defect", ["missing_arm", "nan", "no_step", "duplicate_probe", "missing_probe"])
def test_missing_or_invalid_training_measurements_cannot_supply_pass(defect):
    from semantic_retrieval.runtime.pilot_workflow import training_report_gates

    reports = training_reports()
    if defect == "missing_arm":
        reports.pop()
    if defect == "nan":
        reports[0]["losses"][0]["total"] = float("nan")
    if defect == "no_step":
        reports[0]["losses"][0]["stepped"] = 0
    if defect == "duplicate_probe":
        reports[3]["gradients"][-1] = reports[3]["gradients"][0]
    if defect == "missing_probe":
        reports[3]["gradients"].clear()
    result = training_report_gates(reports)
    assert not (result["stability"]["passed"] and result["gradients"]["passed"])


def test_bare_analysis_gate_booleans_are_rejected_before_use(tmp_path):
    from semantic_retrieval.runtime.pilot_workflow import validate_analysis_qualification

    context = SimpleNamespace(repo_root=tmp_path, manifest={"git_sha": "a" * 40})
    with pytest.raises(ValueError, match="archive-derived"):
        validate_analysis_qualification(
            context,
            dict(
                relevance_space="benchmark_annotation",
                target_gallery_manifest_sha256="b" * 64,
                gates={key: True for key in ("data", "four_gpu_qualification", "gradients", "stability", "invariance")},
            ),
        )


def test_invariance_requires_all_alias_permutation_measurements(tmp_path):
    import numpy as np
    from semantic_retrieval.runtime.pilot_candidates import file_reference, write_json
    from semantic_retrieval.runtime.pilot_evidence import archived_invariance_gate
    from semantic_retrieval.runtime.pilot_protocol import phase_contract

    runs = []
    for variant, seed in phase_contract("V0S")["runs"]:
        root = tmp_path / f"{variant}-{seed}"
        root.mkdir()
        groups = []
        hashes = {}
        for track in ("natural", "controlled"):
            path = root / f"{track}.npz"
            shape = (2, 4, 5)
            np.savez(
                path,
                score_error=np.zeros(shape),
                answer_consistent=np.ones(shape),
                action_consistent=np.ones(shape),
                attempted_k_consistent=np.ones(shape),
                alias_count=np.broadcast_to(np.array([1, 2, 4, 8])[None, :, None], shape),
                permutation=np.broadcast_to(np.arange(5)[None, None, :], shape),
            )
            ref = file_reference(path)
            hashes[path.name] = ref["sha256"]
            groups.append(dict(arm=variant.split("-")[0], seed=seed, track=track, mechanism="system", arrays=ref))
        fragment = root / "provenance/pilot_score_fragment.json"
        fragment.parent.mkdir()
        write_json(fragment, dict(groups=groups))
        hashes["provenance/pilot_score_fragment.json"] = file_reference(fragment)["sha256"]
        runs.append(dict(paths=SimpleNamespace(root=root), status=dict(output_hashes=hashes)))
    result, refs = archived_invariance_gate(dict(runs=runs))
    assert result["passed"] and result["group_count"] == 30
    # Hash-bound incomplete probe coverage cannot be promoted by a group flag.
    path = tmp_path / "A0-val_locked-0" / "natural.npz"
    with np.load(path) as rows:
        altered = dict(rows)
    altered["permutation"] = np.zeros_like(altered["permutation"])
    np.savez(path, **altered)
    ref = file_reference(path)
    run = runs[0]
    run["status"]["output_hashes"][path.name] = ref["sha256"]
    fragment = run["paths"].root / "provenance/pilot_score_fragment.json"
    import json

    document = json.loads(fragment.read_text())
    document["groups"][0]["arrays"] = ref
    fragment.write_text(json.dumps(document))
    run["status"]["output_hashes"]["provenance/pilot_score_fragment.json"] = file_reference(fragment)["sha256"]
    assert not archived_invariance_gate(dict(runs=runs))[0]["passed"]
