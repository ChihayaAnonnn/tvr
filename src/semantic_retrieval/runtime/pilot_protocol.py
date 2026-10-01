"""Archived phase dependencies for the separately authorized benchmark pilot."""

import copy
import hashlib
import math
import re
import subprocess
from pathlib import Path

from experiment_artifacts import (
    hash_run_outputs, load_validated_run_lifecycle, make_run_paths, manifest_sha256,
    parse_registry_text, strict_json_loads, validate_artifact_json_payload,
)

PILOT = "semantic-answerability-pilot-v1"
CONTROL = f"experiments/{PILOT}"
Q0_PROTOCOL = f"{PILOT}:c1-c2-c3-q0-v2"
PHYSICAL_INDICES = (0, 2, 3, 4)
Q0_CHECKS = (
    "label_v2", "dynamic_panels", "unique_rank_queries", "global_known_counts",
    "a2_detach", "a3_shared", "a4_score_only", "unknown_no_step",
    "sampler_checkpoint", "rng_restore", "optimizer_restore", "hash_tamper",
    "rank_parameter_consistency", "calibration_tracks", "cross_gallery_identity",
    "cancellation_guard",
)
_TRAIN = {
    "P1a": (("A0", 0), ("P1", 0)),
    "P1b": tuple((a, 0) for a in ("A1", "A2", "A3", "A4")),
    "P2a": tuple((a, s) for s in (1, 2) for a in ("A0", "P1")),
    "P2b": tuple((a, s) for s in (1, 2) for a in ("A1", "A2", "A3", "A4")),
}
_REQUIRES = {
    "D1": (), "Q0P": (), "Q0": ("D1", "Q0P"), "E0": ("D1", "Q0"),
    "P1a": ("D1", "Q0", "E0"), "P1b": ("D1", "Q0", "E0", "P1a"),
    "P2a": ("D1", "Q0", "E0", "P1a", "P1b"),
    "P2b": ("D1", "Q0", "E0", "P1a", "P1b", "P2a"),
    "C0S": ("D1", "Q0", "E0", "P1a", "P1b", "P2a", "P2b"),
    "C0": ("P1a", "P1b", "P2a", "P2b", "C0S"),
    "V0E": ("D1", "Q0", "C0"),
    "V0S": ("D1", "Q0", "E0", "C0", "V0E"), "V0": ("C0", "V0S"),
}


def phase_contract(phase_id):
    if phase_id not in _REQUIRES:
        raise ValueError("unknown pilot phase")
    if phase_id in _TRAIN:
        runs, category, hours, dataset, split = _TRAIN[phase_id], "training", 8.0, "msrvtt", "train"
    elif phase_id in {"C0S", "V0S"}:
        splits = ("val_development", "val_calibration") if phase_id == "C0S" else ("val_locked",)
        runs = tuple((f"{a}-{sp}", s) for s in (0, 1, 2) for a in ("A0", "A1", "A2", "A3", "A4") for sp in splits)
        category, hours, dataset, split = "evaluation", (8.0 if phase_id == "C0S" else 10.0) / len(runs), "msrvtt", None
    else:
        variant = {"D1": "benchmark_compile", "Q0P": "pilot-q0-prep", "Q0": "pilot-q0-v2",
                   "E0": "fixed-clip-candidates", "V0E": "fixed-clip-locked-candidates",
                   "C0": "pilot-analysis-freeze", "V0": "pilot-analysis-locked"}[phase_id]
        runs, dataset = ((variant, 0),), "synthetic" if phase_id in {"Q0P", "Q0"} else "msrvtt"
        category = "qualification" if phase_id == "Q0" else "evaluation"
        hours = 8.0 if phase_id == "Q0" else (4.0 if phase_id == "E0" else (2.0 if phase_id == "V0E" else 0.0))
        split = {"D1": "val", "Q0P": "val", "Q0": "train", "E0": "val_development",
                 "C0": "val_development", "V0E": "val_locked", "V0": "val_locked"}[phase_id]
    return dict(phase_id=phase_id, requires=_REQUIRES[phase_id], runs=runs, category=category,
                run_gpu_hours=hours, dataset=dataset, eval_split=split)


def _git_bytes(repo, *args):
    try:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        raise ValueError("pilot evidence must be committed in the current Git history") from exc


def committed_bytes(repo_root, reference, *, prefix=CONTROL):
    """Read exact, tracked, unchanged bytes at the referenced ancestor commit."""
    repo = Path(repo_root).resolve()
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "commit"}:
        raise ValueError("pilot evidence reference requires path, sha256 and commit")
    path = Path(reference["path"])
    if (path.is_absolute() or ".." in path.parts or not path.is_relative_to(prefix)
            or not re.fullmatch("[0-9a-f]{64}", str(reference["sha256"]))
            or not re.fullmatch("[0-9a-f]{40}", str(reference["commit"]))):
        raise ValueError("pilot evidence namespace or digest is invalid")
    disk = repo / path
    if disk.resolve() != disk or disk.is_symlink() or not disk.is_file():
        raise ValueError("pilot evidence path is not a canonical tracked file")
    _git_bytes(repo, "merge-base", "--is-ancestor", reference["commit"], "HEAD")
    old = _git_bytes(repo, "show", f"{reference['commit']}:{path.as_posix()}")
    current = _git_bytes(repo, "show", f"HEAD:{path.as_posix()}")
    if old != current or old != disk.read_bytes() or hashlib.sha256(old).hexdigest() != reference["sha256"]:
        raise ValueError("pilot committed evidence digest or working bytes differ")
    return old


def verify_phase_evidence(repo_root, evidence_ref, expected_phase):
    from scripts.manage_experiment_run import _registry_record

    repo = Path(repo_root).resolve()
    raw = committed_bytes(repo, evidence_ref, prefix=f"{CONTROL}/summaries")
    summary = strict_json_loads(raw.decode("utf-8"))
    validate_artifact_json_payload(summary)
    if (not isinstance(summary, dict) or summary.get("study") != PILOT
            or not isinstance(summary.get("phase"), str)
            or not (summary["phase"] == expected_phase or summary["phase"].startswith(expected_phase + "-"))
            or not isinstance(summary.get("runs"), list) or not summary["runs"]
            or summary.get("terminal_counts") != {"completed": len(summary["runs"])}):
        raise ValueError("pilot predecessor phase must be the complete successful batch")
    registry = parse_registry_text(_git_bytes(repo, "show", f"HEAD:{CONTROL}/registry.jsonl").decode("utf-8"))
    old_registry = parse_registry_text(_git_bytes(repo, "show", f"{evidence_ref['commit']}:{CONTROL}/registry.jsonl").decode("utf-8"))
    if (repo / CONTROL / "registry.jsonl").read_bytes() != _git_bytes(repo, "show", f"HEAD:{CONTROL}/registry.jsonl"):
        raise ValueError("predecessor terminal registry must be committed")
    results, seen = [], set()
    for row in summary["runs"]:
        if not isinstance(row, dict) or row.get("state") != "completed" or row.get("run_id") in seen:
            raise ValueError("pilot predecessor batch has invalid or duplicate runs")
        seen.add(row["run_id"])
        entries = [r for r in registry if r["run_id"] == row["run_id"] and r["state"] == "completed"]
        if len(entries) != 1 or entries[0] not in old_registry:
            raise ValueError("predecessor terminal registry was not committed with its summary")
        root = Path(entries[0]["artifact_path"])
        paths = make_run_paths(root.parents[3], PILOT, row["run_id"])
        if paths.root != root:
            raise ValueError("predecessor belongs to a foreign artifact namespace")
        manifest, status, _ = load_validated_run_lifecycle(paths)
        if (manifest["study"] != PILOT or manifest["test_authorized"] is not False
                or status["state"] != "completed" or status["exit_code"] != 0
                or manifest_sha256(manifest) != row.get("manifest_sha256")
                or status["output_hashes"] != row.get("output_hashes")
                or hash_run_outputs(paths) != status["output_hashes"]):
            raise ValueError("pilot predecessor identity or output inventory differs")
        selected = entries[0]["selected_validation"]
        expected = _registry_record(manifest, "completed", summary_metrics=row.get("summary_metrics"), selected_validation=selected)
        if entries[0] != expected:
            raise ValueError("pilot predecessor terminal record differs from its archive")
        results.append(dict(manifest=manifest, status=status, paths=paths))
    contract = phase_contract(expected_phase)
    if ([(r['manifest']['variant'], r['manifest']['seed']) for r in results] != list(contract['runs'])
            or any(r['manifest']['dataset'] != contract['dataset'] or r['manifest']['stage'] != 'baseline' for r in results)):
        raise ValueError('pilot predecessor is not the complete registered phase batch')
    return dict(phase=summary["phase"], summary=summary, runs=results)


def validate_frozen_prerequisite(repo_root, evidence):
    """Require C0's actual immutable scientific freeze before any locked worker."""
    from semantic_retrieval.data.episodes import file_sha256
    from semantic_retrieval.evaluation import pilot_analysis
    if len(evidence['runs']) != 1:
        raise ValueError('locked access requires one completed C0 frozen analysis')
    run = evidence['runs'][0]
    path = run['paths'].provenance/'pilot_frozen.json'
    digest = run['status']['output_hashes'].get('provenance/pilot_frozen.json')
    if not path.is_file() or digest != file_sha256(path):
        raise ValueError('locked access requires the hash-bound C0 frozen artifact')
    frozen = strict_json_loads(path.read_text())
    evaluation = run['manifest']['config']['evaluation']
    expected_models = {f'{arm}_seed{seed}' for seed in (0,1,2) for arm in ('A0','A1','A2','A3','A4')}
    if (run['manifest']['variant'] != 'pilot-analysis-freeze'
            or evaluation != dict(eval_split='val_development',qualification_scenario='pilot_analysis')
            or frozen.get('schema_version') != 1 or frozen.get('relevance_space') != 'benchmark_annotation'
            or frozen.get('source_sha') != run['manifest']['git_sha']
            or frozen.get('analysis_sha256') != file_sha256(pilot_analysis.__file__)
            or set(frozen.get('model_identities',{})) != expected_models
            or not isinstance(frozen.get('label_protocol'),dict) or not frozen['label_protocol']
            or not isinstance(frozen.get('target_gallery'),dict)
            or any(not isinstance(frozen.get(key),dict) or not frozen[key] for key in ('policies','scalar_policies','calibration','strongest_scalar'))):
        raise ValueError('C0 frozen model, source, target or policy identity is incomplete')
    return dict(path=str(path),sha256=digest)


def validate_q0_report(report, *, source_sha):
    if (not isinstance(report, dict) or report.get("schema_version") != 1
            or report.get("protocol") != Q0_PROTOCOL or report.get("source_sha") != source_sha
            or report.get("world_size") != 4 or report.get("device") != "cuda"
            or report.get("precision") != "fp32" or report.get("tf32") is not False
            or report.get("synthetic_only") is not True
            or report.get("physical_indices") != list(PHYSICAL_INDICES)
            or not isinstance(report.get("checks"), dict)
            or any(report["checks"].get(k) is not True for k in Q0_CHECKS)):
        raise ValueError("new pilot four-GPU qualification evidence is incomplete")
    ranks = report.get("rank_records")
    if (not isinstance(ranks, list) or len(ranks) != 4
            or {r.get("rank") for r in ranks if isinstance(r, dict)} != {0, 1, 2, 3}
            or len({r.get("gpu_uuid") for r in ranks if isinstance(r, dict)}) != 4
            or any(not isinstance(r.get("gpu_uuid"), str) or not r["gpu_uuid"].startswith("GPU-") for r in ranks)):
        raise ValueError("qualification must record four unique physical devices and ranks")
    seconds = report.get("seconds_per_update")
    if (type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds <= 0
            or seconds * 240 > 2 * 3600):
        raise ValueError("qualified throughput cannot satisfy the registered per-run budget")
    for field in ("resume_parameter_max_abs_delta", "resume_optimizer_max_abs_delta"):
        value = report.get(field)
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1e-6:
            raise ValueError("qualification restore deltas exceed the FP32 tolerance")
    return report


def _marker(manifest):
    markers = [e for e in manifest.get("selection_lineage", []) if isinstance(e, dict) and e.get("kind") == "pilot_phase"]
    if len(markers) != 1:
        raise ValueError("pilot run needs exactly one archived phase identity")
    marker = markers[0]
    if marker.get("protocol_version") != 1 or not isinstance(marker.get("evidence"), dict):
        raise ValueError("pilot phase protocol or evidence is invalid")
    return marker


def _same_q0_source(repo, q0_source, source):
    changed = _git_bytes(repo, "diff", "--name-only", f"{q0_source}..{source}", "--", "src/semantic_retrieval", "scripts", ":(top,glob)*.py", "experiments/contextual-risk-v2/schemas").decode().splitlines()
    if changed:
        raise ValueError("execution source changed after the new pilot qualification: " + ", ".join(changed))


def validate_pilot_gpu_qualification(context):
    marker = _marker(context.manifest)
    reference = marker["evidence"].get("Q0")
    evidence = verify_phase_evidence(context.repo_root, reference, "Q0")
    if len(evidence["runs"]) != 1:
        raise ValueError("pilot Q0 must have one complete four-rank qualification run")
    run = evidence["runs"][0]
    manifest = run["manifest"]
    if (manifest["dataset"] != "synthetic" or manifest["variant"] != "pilot-q0-v2"
            or manifest["environment"]["gpu_count"] != 4):
        raise ValueError("old or scientific evidence cannot replace new synthetic Q0")
    report = strict_json_loads((run["paths"].provenance / "pilot-qualification.json").read_text())
    validate_q0_report(report, source_sha=manifest["git_sha"])
    _same_q0_source(context.repo_root, manifest["git_sha"], context.manifest["git_sha"])
    if manifest["run_id"] not in context.manifest["parent_run_ids"]:
        raise ValueError("pilot scientific worker must declare its qualified Q0 parent")
    if hasattr(context,'config'):
        import os
        import torch
        from scripts.qualify_semantic_retrieval import local_device_uuid
        uuids = [r['gpu_uuid'] for r in sorted(report['rank_records'],key=lambda r:r['rank'])]
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 4
                or os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') != uuids
                or [local_device_uuid(i) for i in range(4)] != uuids):
            raise ValueError('worker physical GPU UUID topology differs from qualified Q0')
    return report


def build_phase(phase_id, *, phase_path, scheduled_timestamp_utc, run_templates, evidence):
    contract = phase_contract(phase_id)
    if (not isinstance(run_templates, list)
            or [(r.get("variant"), r.get("seed")) for r in run_templates] != list(contract["runs"])):
        raise ValueError("pilot phase batch must contain every registered arm/seed in order")
    if set(contract["requires"]) - set(evidence):
        raise ValueError("pilot phase is missing prerequisite evidence")
    path = Path(phase_path)
    if path.is_absolute() or ".." in path.parts or not path.is_relative_to(f"{CONTROL}/phases"):
        raise ValueError("pilot phase path must be inside the registered control plane")
    if not re.fullmatch(r"\d{8}T\d{6}Z", scheduled_timestamp_utc):
        raise ValueError("pilot run identity requires a scheduled UTC timestamp")
    runs = []
    for raw in run_templates:
        run = copy.deepcopy(raw)
        if run.get("test_authorized") is not False or run.get("dataset") != contract["dataset"]:
            raise ValueError("pilot phase test or dataset identity differs")
        run.pop("timestamp_utc", None)
        run["scheduled_timestamp_utc"] = scheduled_timestamp_utc
        run["selection_lineage"] = [e for e in run.get("selection_lineage", []) if e.get("kind") != "pilot_phase"]
        run["selection_lineage"].append(dict(kind="pilot_phase", protocol_version=1, phase_id=phase_id,
            phase_path=path.as_posix(), evidence=copy.deepcopy(evidence), gpu_hours=contract["run_gpu_hours"]))
        runs.append(run)
    return dict(phase=f"{phase_id}-{scheduled_timestamp_utc}", study=PILOT, runs=runs)


def assert_phase_authorized(repo_root, manifest, operation):
    """Validate whole committed batch and all prerequisites before archive mutations."""
    from types import SimpleNamespace
    from scripts.manage_experiment_run import _build_manifest

    repo = Path(repo_root).resolve()
    marker = _marker(manifest)
    contract = phase_contract(marker["phase_id"])
    phase_path = Path(marker["phase_path"])
    if phase_path.is_absolute() or ".." in phase_path.parts or not phase_path.is_relative_to(f"{CONTROL}/phases"):
        raise ValueError("pilot phase must be a tracked control-plane file")
    raw = _git_bytes(repo, "show", f"{manifest['git_sha']}:{phase_path.as_posix()}")
    if raw != (repo / phase_path).read_bytes() or raw != _git_bytes(repo, "show", f"HEAD:{phase_path.as_posix()}"):
        raise ValueError("pilot phase must match its committed source exactly")
    phase = strict_json_loads(raw.decode())
    runs = phase.get("runs", [])
    if (phase.get("study") != PILOT or [(r.get("variant"), r.get("seed")) for r in runs] != list(contract["runs"])
            or any(r.get("dataset") != contract["dataset"] or r.get("test_authorized") is not False for r in runs)):
        raise ValueError("pilot scientific batch must contain all registered arm/seed identities")
    expected = [_build_manifest(r, study=PILOT, git_sha=manifest["git_sha"], artifact_root=Path(manifest["artifact_path"]).parents[3],
                registry=Path(CONTROL) / "registry.jsonl", common_root=Path(_git_bytes(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").decode().strip()).parent) for r in runs]
    if manifest not in expected or any(_marker(m) != marker for m in expected):
        raise ValueError("pilot manifest differs from its complete committed phase")
    if marker.get("gpu_hours") != contract["run_gpu_hours"]:
        raise ValueError("pilot run exceeds its allocated GPU budget")
    env = manifest["environment"]
    gpu = contract["run_gpu_hours"] > 0
    if (env.get("precision") != "fp32" or type(env.get("gpu_count")) is not int
            or env["gpu_count"] != (4 if gpu else 0)
            or (gpu and "A800" not in env.get("gpu_model", ""))):
        raise ValueError("pilot environment differs from frozen FP32/four-A800 topology")
    split = manifest["config"]["evaluation"].get("eval_split")
    wanted = contract["eval_split"]
    if wanted is None:
        wanted = manifest["variant"].split("-", 1)[1]
    if split != wanted or split == "test":
        raise ValueError("pilot phase cannot access an unfrozen or sealed evaluation split")
    evidence = {}
    for required in contract["requires"]:
        evidence[required] = verify_phase_evidence(repo, marker["evidence"].get(required), required)
        if any(r["manifest"]["run_id"] not in manifest["parent_run_ids"] for r in evidence[required]["runs"]):
            raise ValueError("pilot prerequisite run must appear in parent_run_ids")
    if "D1" in evidence:
        from semantic_retrieval.data.pilot_catalog import TARGETS
        ready = [r for r in evidence["D1"]["runs"] if r["manifest"]["variant"] == "benchmark_compile"]
        summaries = [strict_json_loads((r['paths'].provenance/'input-summary.json').read_text()) for r in ready]
        if not summaries or any(s.get('ready') is not True or s.get('relevance_space') != 'benchmark_annotation'
                or not s.get('label_protocol') or set(s.get('splits',{})) != {'train','val_development','val_calibration','val_locked'}
                or any(s['splits'][sp].get('official_video_count',0) < (TARGETS['train_min'] if sp=='train' else TARGETS[sp])
                    or s['splits'][sp].get('canonical_video_count',0) < TARGETS['canonical_min']
                    or s['splits'][sp].get('query_count',0) != s['splits'][sp].get('official_video_count')
                    for sp in s['splits'])
                or s['splits']['val_locked'].get('component_count',0) < TARGETS['locked_components_min'] for s in summaries):
            raise ValueError("pilot D1 benchmark data compilation is not ready")
    if marker['phase_id'] in {'V0E','V0S','V0'}:
        frozen = validate_frozen_prerequisite(repo,evidence['C0'])
        from semantic_retrieval.runtime.input_bindings import manifest_input_references
        inputs = manifest_input_references(repo,manifest)
        if inputs.get(frozen['path']) != frozen['sha256']:
            raise ValueError('locked worker must reserve its exact C0 frozen artifact')
    if gpu and marker["phase_id"] != "Q0":
        validate_pilot_gpu_qualification(SimpleNamespace(manifest=manifest, repo_root=repo))
    if marker["phase_id"] in _TRAIN:
        if manifest["config"]["model"].get("pilot_arm") != manifest["variant"]:
            raise ValueError("training arm identity differs from reserved model mode")
        pilot = manifest["config"]["evaluation"].get("pilot")
        if pilot is None:
            pilot = manifest["config"]["optimizer"].get("pilot")
        # C2's effective serializer places the bound pilot config in evaluation.
        if not isinstance(pilot, dict) or pilot.get("arm") != manifest["variant"]:
            raise ValueError("scientific training requires the new dynamic sampler protocol")
        if manifest['config']['optimizer'].get('seed') != manifest['seed']:
            raise ValueError('training seed differs from reserved arm/seed identity')
    limits = dict(training=144.0, qualification=8.0, evaluation=24.0)
    totals = {key: 0.0 for key in limits}
    records = parse_registry_text(_git_bytes(repo, "show", f"HEAD:{CONTROL}/registry.jsonl").decode())
    batch_ids = {m["run_id"] for m in expected}
    for record in records:
        if record["event"] != "reserved" or record["run_id"] in batch_ids:
            continue
        root = Path(record["artifact_path"])
        old, _, _ = load_validated_run_lifecycle(make_run_paths(root.parents[3], PILOT, record["run_id"]))
        markers = [e for e in old["selection_lineage"] if isinstance(e, dict) and e.get("kind") == "pilot_phase"]
        if not markers:
            continue
        previous = _marker(old)
        if previous["phase_id"] == marker["phase_id"]:
            raise ValueError("pilot phase is already reserved; automatic retry is forbidden")
        old_contract = phase_contract(previous["phase_id"])
        totals[old_contract["category"]] += old_contract["run_gpu_hours"]
    totals[contract["category"]] += contract["run_gpu_hours"] * len(expected)
    if any(totals[k] > limit + 1e-9 for k, limit in limits.items()) or sum(totals.values()) > 176 + 1e-9:
        raise ValueError("pilot reservation exceeds the registered GPU-hour budget")
