"""Recompute pilot analysis gates from immutable, committed worker archives."""

import math
from collections import Counter
from pathlib import Path

import numpy as np

from experiment_artifacts import strict_json_loads
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.data.pilot_catalog import TARGETS
from semantic_retrieval.objectives.retrieval import LOSS_NAMES
from semantic_retrieval.runtime.input_bindings import context_input_references
from semantic_retrieval.runtime.pilot_protocol import (
    _marker,
    _same_q0_source,
    validate_q0_report,
    verify_phase_evidence,
)

PRODUCER = "semantic-pilot-archive-gates-v1"
TRAIN_PHASES = ("P1a", "P1b", "P2a", "P2b")
BASE_PHASES = ("D1", "Q0", "E0", *TRAIN_PHASES, "C0S")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def training_report_gates(reports):
    """No archive claim is made here: the caller authenticates every input log."""
    expected = {(arm, seed) for seed in (0, 1, 2) for arm in ("A0", "P1", "A1", "A2", "A3", "A4")}
    identities = [(r["arm"], r["seed"]) for r in reports]
    complete = len(identities) == len(expected) and set(identities) == expected
    stable = gradient_ok = complete
    step_count = probe_count = 0
    for report in reports:
        arm = report["arm"]
        steps = 240 if arm == "A0" else 120
        losses = report.get("losses", [])
        probes = report.get("gradients", [])
        good = [r.get("iteration") for r in losses] == list(range(1, steps + 1))
        numeric = (*LOSS_NAMES, "total", "forward_seconds", "backward_seconds", "optimizer_seconds")
        good &= all(
            r.get("stepped") == 1
            and r.get("step") == r.get("iteration")
            and all(_finite(r.get(k)) and r[k] >= 0 for k in numeric)
            for r in losses
        )
        for offset in range(0, steps, 120):
            good &= Counter((r.get("k"), r.get("panel")) for r in losses[offset : offset + 120]) == Counter(
                {(k, panel): 15 for k in (4, 8, 16, 32) for panel in ("present", "absent")}
            )
        stable &= good
        step_count += len(losses)
        if arm in ("A2", "A3", "A4"):
            good = [r.get("iteration") for r in probes] == list(range(20, steps + 1, 20))
            for row in probes:
                a, b, cosine = (
                    row.get(k) for k in ("matching_gradient_norm", "presence_gradient_norm", "gradient_cosine")
                )
                good &= _finite(a) and a > 0 and _finite(b) and b >= 0
                if arm == "A3":
                    good &= _finite(b) and b > 0 and _finite(cosine) and -1 <= cosine <= 1
                else:
                    good &= b == 0 and cosine is None
            gradient_ok &= good
            probe_count += len(probes)
    return dict(
        stability=dict(
            status="available" if complete else "incomplete",
            passed=bool(stable),
            run_count=len(reports),
            update_count=step_count,
        ),
        gradients=dict(
            status="available" if complete else "incomplete",
            passed=bool(gradient_ok),
            probe_count=probe_count,
            tolerance=0.0,
        ),
    )


def archived_reference(run, relative):
    path = run["paths"].root / relative
    digest = run["status"]["output_hashes"].get(relative)
    if digest is None or not path.is_file() or file_sha256(path) != digest:
        raise ValueError("analysis evidence is not an immutable archived output: " + relative)
    return dict(path=str(path), sha256=digest)


def _json(reference):
    return strict_json_loads(Path(reference["path"]).read_text())


def _jsonl(reference):
    return [strict_json_loads(line) for line in Path(reference["path"]).read_text().splitlines() if line]


def _data_gate(summary):
    splits = summary.get("splits", {})
    return (
        summary.get("ready") is True
        and summary.get("relevance_space") == "benchmark_annotation"
        and bool(summary.get("label_protocol"))
        and set(splits) == {"train", "val_development", "val_calibration", "val_locked"}
        and all(
            splits[sp].get("official_video_count", 0) >= (TARGETS["train_min"] if sp == "train" else TARGETS[sp])
            and splits[sp].get("canonical_video_count", 0) >= TARGETS["canonical_min"]
            and splits[sp].get("query_count", 0) == splits[sp].get("official_video_count")
            for sp in splits
        )
        and splits["val_locked"].get("component_count", 0) >= TARGETS["locked_components_min"]
    )


def build_analysis_qualification(repo_root, evidence):
    """Return a recomputable document and every measured input that must be reserved."""
    from semantic_retrieval.evaluation import pilot_analysis

    repo = Path(repo_root).resolve()
    phases = BASE_PHASES + (("C0", "V0E", "V0S") if "V0S" in evidence else ())
    if set(evidence) != set(phases):
        raise ValueError("analysis requires the exact complete archive-derived evidence chain")
    authenticated = {name: verify_phase_evidence(repo, evidence[name], name) for name in phases}
    inputs = [dict(path=str(repo / r["path"]), sha256=r["sha256"]) for r in evidence.values()]
    d1 = authenticated["D1"]["runs"][0]
    summary_ref = archived_reference(d1, "provenance/input-summary.json")
    result_ref = archived_reference(d1, "provenance/preparation-result.json")
    summary, result = _json(summary_ref), _json(result_ref)
    gallery_ref = result["manifests"]["val_locked"]["gallery"]
    # Linked references must themselves belong to D1's terminal output inventory.
    relative = Path(gallery_ref["path"]).relative_to(d1["paths"].root).as_posix()
    if archived_reference(d1, relative) != gallery_ref:
        raise ValueError("D1 target gallery is disconnected from its archived inventory")
    q0 = authenticated["Q0"]["runs"][0]
    q0_ref = archived_reference(q0, "provenance/pilot-qualification.json")
    q0_report = validate_q0_report(_json(q0_ref), source_sha=q0["manifest"]["git_sha"])
    inputs += [summary_ref, result_ref, gallery_ref, q0_ref]
    reports = []
    for phase in TRAIN_PHASES:
        for run in authenticated[phase]["runs"]:
            manifest = run["manifest"]
            _same_q0_source(repo, q0["manifest"]["git_sha"], manifest["git_sha"])
            losses = archived_reference(run, "logs/pilot-losses.jsonl")
            gradients = archived_reference(run, "logs/pilot-gradients.jsonl")
            inputs += [losses, gradients]
            reports.append(
                dict(arm=manifest["variant"], seed=manifest["seed"], losses=_jsonl(losses), gradients=_jsonl(gradients))
            )
    measurements = training_report_gates(reports)
    invariance = dict(status="unavailable", passed=False, group_count=0)
    if "V0S" in authenticated:
        invariance, probe_refs = archived_invariance_gate(authenticated["V0S"])
        inputs += probe_refs
    measurements["invariance"] = invariance
    document = dict(
        schema_version=1,
        producer=PRODUCER,
        relevance_space="benchmark_annotation",
        qualified_source_sha=q0["manifest"]["git_sha"],
        analysis_sha256=file_sha256(pilot_analysis.__file__),
        label_protocol=summary["label_protocol"],
        target_gallery_manifest_sha256=gallery_ref["sha256"],
        evidence={name: evidence[name] for name in phases},
        gates=dict(
            data=bool(_data_gate(summary)),
            four_gpu_qualification=bool(q0_report),
            gradients=measurements["gradients"]["passed"],
            stability=measurements["stability"]["passed"],
            invariance=invariance["passed"],
        ),
        measurements=measurements,
    )
    return document, inputs, authenticated


def archived_invariance_gate(evidence):
    """Coverage is verified in addition to numerical invariance; absent probes cannot pass."""
    refs, measured = [], []
    for run in evidence["runs"]:
        fragment_ref = archived_reference(run, "provenance/pilot_score_fragment.json")
        fragment = _json(fragment_ref)
        refs.append(fragment_ref)
        for group in fragment["groups"]:
            if group["mechanism"] != "system":
                continue
            reference = group["arrays"]
            relative = Path(reference["path"]).relative_to(run["paths"].root).as_posix()
            actual = archived_reference(run, relative)
            if any(reference[k] != actual[k] for k in ("path", "sha256")):
                raise ValueError("invariance arrays differ from completed worker archive")
            refs.append(actual)
            with np.load(actual["path"], allow_pickle=False) as rows:
                keys = ("score_error", "answer_consistent", "action_consistent", "attempted_k_consistent")
                if not all(k in rows for k in keys):
                    continue
                score = rows["score_error"]
                covered = (
                    score.ndim == 3
                    and score.shape[1:] == (4, 5)
                    and "alias_count" in rows
                    and "permutation" in rows
                    and rows["alias_count"].shape == score.shape
                    and rows["permutation"].shape == score.shape
                    and np.array_equal(
                        rows["alias_count"], np.broadcast_to(np.array([1, 2, 4, 8])[None, :, None], score.shape)
                    )
                    and np.array_equal(rows["permutation"], np.broadcast_to(np.arange(5)[None, None, :], score.shape))
                )
                passed = (
                    covered
                    and score.size > 0
                    and np.isfinite(score).all()
                    and np.all(score >= 0)
                    and float(score.max()) <= 1e-6
                    and all(rows[k].shape == score.shape and np.all(rows[k] == 1) for k in keys[1:])
                )
                measured.append((group["arm"], group["seed"], group["track"], bool(passed)))
    expected = {
        (arm, seed, track)
        for arm in ("A0", "A1", "A2", "A3", "A4")
        for seed in (0, 1, 2)
        for track in ("natural", "controlled")
    }
    complete = len(measured) == len(expected) and {m[:3] for m in measured} == expected
    return dict(
        status="available" if complete else "unavailable",
        passed=bool(complete and all(m[3] for m in measured)),
        group_count=len(measured),
    ), refs


def validate_analysis_qualification(context, document):
    """Reject operator booleans, foreign chains, changed source and unreserved evidence."""
    fields = {
        "schema_version",
        "producer",
        "relevance_space",
        "qualified_source_sha",
        "analysis_sha256",
        "label_protocol",
        "target_gallery_manifest_sha256",
        "evidence",
        "gates",
        "measurements",
    }
    if (
        not isinstance(document, dict)
        or set(document) != fields
        or document.get("schema_version") != 1
        or document.get("producer") != PRODUCER
    ):
        raise ValueError("analysis qualification requires archive-derived evidence, not caller gate booleans")
    marker = _marker(context.manifest)
    phase = marker["phase_id"]
    if phase not in ("C0", "V0") or ("V0S" in document["evidence"]) != (phase == "V0"):
        raise ValueError("analysis gate evidence differs from the reserved phase")
    if any(marker["evidence"].get(name) != ref for name, ref in document["evidence"].items()):
        raise ValueError("analysis gate evidence differs from reserved predecessor summaries")
    computed, inputs, _ = build_analysis_qualification(context.repo_root, document["evidence"])
    _same_q0_source(context.repo_root, computed["qualified_source_sha"], context.manifest["git_sha"])
    if computed != document:
        raise ValueError("analysis gates or measurements differ from immutable archive recomputation")
    reserved = context_input_references(context)
    if any(reserved.get(r["path"]) != r["sha256"] for r in inputs):
        raise ValueError("analysis qualification has unreserved measured inputs")
    return computed["gates"]
