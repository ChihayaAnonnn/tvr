#!/usr/bin/env python3
"""Reserved CPU preparation of separately bound training-source benchmark assets."""

import argparse
import csv
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import yaml

from semantic_retrieval.config import RuntimeConfig
from semantic_retrieval.data.benchmark_labels import load_benchmark_labels
from semantic_retrieval.data.episodes import EpisodePreprocessing, audit_episode_splits, file_sha256, read_episodes
from semantic_retrieval.data.msrvtt_snapshot import (
    caption_relation_digest,
    inspect_training_video,
    training_annotations,
)
from semantic_retrieval.data.pilot_catalog import _safe_path, assign_pilot_splits, identity_digest, load_pilot_catalog
from semantic_retrieval.data.pilot_compile import compile_pilot_manifests, summarize_pilot_inputs
from semantic_retrieval.runtime.artifacts import complete_run, fail_run, prepare_run, publish_metrics, start_run
from semantic_retrieval.runtime.distributed import configure_fp32
from semantic_retrieval.runtime.input_bindings import context_input_references

PREPARATION_KEYS = {
    "mode",
    "protocol_id",
    "relevance_space",
    "allowed_root",
    "catalog",
    "annotation",
    "source_confirmation",
    "allowlist_csv",
    "source_annotation",
    "video_directory",
    "candidate_orders",
}


def write_input(path, value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    with path.open("xb") as stream:
        stream.write(data)
    return dict(path=str(path), sha256=file_sha256(path))


def _bound(context, ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("input reference must have exact path/SHA")
    refs = context_input_references(context)
    if refs.get(ref["path"]) != ref["sha256"]:
        raise ValueError("preparation input is not reserved")
    path = Path(ref["path"])
    if not path.is_absolute() or path.resolve() != path or path.is_symlink() or file_sha256(path) != ref["sha256"]:
        raise ValueError("preparation input path/hash mismatch")
    return path


def validate_source_confirmation(context, preparation):
    ref = preparation["source_confirmation"]
    audit_path = _bound(context, ref)
    study = yaml.safe_load(context.config.study_path.read_text())
    anchor = study.get("source_confirmation")
    if not isinstance(anchor, dict) or set(anchor) != {"path", "sha256", "commit"} or anchor["sha256"] != ref["sha256"]:
        raise ValueError("source confirmation must match registered study trust anchor")
    common = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=context.repo_root, text=True
        ).strip()
    ).parent
    accepted = {context.repo_root / anchor["path"], common / anchor["path"]}
    if audit_path not in accepted:
        raise ValueError("source confirmation path differs from registered anchor")
    for commit in (anchor["commit"], "HEAD"):
        blob = subprocess.check_output(["git", "show", f"{commit}:{anchor['path']}"], cwd=context.repo_root)
        if hashlib.sha256(blob).hexdigest() != ref["sha256"]:
            raise ValueError("source confirmation differs from Git-tracked proof")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", anchor["commit"], "HEAD"],
        cwd=context.repo_root,
        check=True,
        capture_output=True,
    )
    if preparation["protocol_id"] != study.get("protocol_id"):
        raise ValueError("preparation protocol differs from registered study")
    return audit_path


def export_snapshot(context, preparation):
    root = Path(preparation["allowed_root"])
    audit_path = validate_source_confirmation(context, preparation)
    audit = json.loads(audit_path.read_text())
    if (
        audit["status"]["local_annotation_matches_published_captions"]
        != "VERIFIED_EXACT_FOR_V2_TRAINING_ALLOWLIST_ONLY"
    ):
        raise ValueError("training caption identity has not been verified")
    csv_path = _bound(context, preparation["allowlist_csv"])
    raw_path = _bound(context, preparation["source_annotation"])
    approved = {(row["path"], row["sha256"]) for row in audit["local_assets"]}
    if any(
        (ref["path"], ref["sha256"]) not in approved
        for ref in (preparation["allowlist_csv"], preparation["source_annotation"])
    ):
        raise ValueError("source files differ from reviewed source identity")
    if raw_path.name != "MSRVTT_v2.json" or csv_path.name != "MSRVTT_train.9k.csv":
        raise ValueError("snapshot export is limited to verified v2/9k pairing")
    _safe_path(csv_path, base=root, allowed_root=root)
    _safe_path(raw_path, base=root, allowed_root=root)
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["video_id"]:
            raise ValueError("training allowlist must be ID-only")
        ids = [r["video_id"] for r in reader]
    expected = audit["training_caption_comparison"]["local_results"]["MSRVTT_v2.json"]
    if len(ids) != len(set(ids)) or len(ids) != expected["training_video_count"]:
        raise ValueError("training allowlist count/uniqueness mismatch")
    videos = {
        i: _safe_path(Path(preparation["video_directory"]) / f"{i}.mp4", base=root, allowed_root=root) for i in ids
    }
    reserved = context_input_references(context)
    if any(str(p) not in reserved for p in videos.values()):
        raise ValueError("every training video must be a reserved input")
    # Only here, after identity and membership checks, decode selected captions.
    rows, privacy = training_annotations(raw_path.read_bytes(), set(ids))
    if (
        len(rows) != expected["selected_records"]
        or caption_relation_digest(rows) != expected["relation_sha256"]
        or {r["official_video_id"] for r in rows} != set(ids)
    ):
        raise ValueError("exported training relationships differ from reviewed audit")
    directory = context.paths.provenance / "train-only"
    directory.mkdir()
    allow = write_input(
        directory / "allowlist.json",
        dict(schema_version=1, origin_split="source_train", official_video_ids=sorted(ids)),
    )
    records = write_input(
        directory / "training-records.json",
        dict(
            schema_version=1,
            origin_split="source_train",
            sentences=[
                dict(video_id=r["official_video_id"], sen_id=r["caption_id"], caption=r["query_text"]) for r in rows
            ],
        ),
    )
    report = write_input(
        directory / "source-report.json",
        dict(
            schema_version=1,
            origin_split="source_train",
            protocol_id=preparation["protocol_id"],
            source_verified=True,
            training_records=records,
            allowlist_sha256=allow["sha256"],
            source_confirmation=preparation["source_confirmation"],
            source_confirmation_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=context.repo_root, text=True
            ).strip(),
            source_annotation=preparation["source_annotation"],
            allowlist_source=preparation["allowlist_csv"],
            caption_relation_sha256=expected["relation_sha256"],
            privacy=privacy,
        ),
    )
    metadata = dict(
        schema_version=1,
        origin_split="source_train",
        protocol_id=preparation["protocol_id"],
        source_annotation_sha256=preparation["source_annotation"]["sha256"],
        source_report_sha256=report["sha256"],
        allowlist_sha256=allow["sha256"],
    )
    full = write_input(directory / "training-captions.json", {**metadata, "annotations": rows})
    selected = {}
    for row in rows:
        value = row["caption_id"]
        key = (0, int(value)) if value.isdecimal() else (1, value)
        old = selected.get(row["official_video_id"])
        if old is None or key < old[0]:
            selected[row["official_video_id"]] = (key, row)
    pilot_rows = [selected[i][1] for i in sorted(ids)]
    annotation = write_input(directory / "pilot-annotations.json", {**metadata, "annotations": pilot_rows})
    catalog = []

    def inspect_id(i):
        path = videos[i]
        return inspect_training_video(path, expected_sha256=reserved[str(path)])

    with ThreadPoolExecutor(max_workers=8) as executor:
        frame_counts = dict(zip(sorted(ids), executor.map(inspect_id, sorted(ids))))
    for i in sorted(ids):
        path = videos[i]
        count = frame_counts[i]
        digest = reserved[str(path)]
        canonical = f"sha256:{digest}"
        common = dict(
            official_video_id=i,
            canonical_id=canonical,
            source_ids=[f"msrvtt:{i}"],
            caption_group_ids=[f"msrvtt:{i}"],
            origin_split="source_train",
        )
        catalog.append(
            dict(asset_id=f"video:{i}", kind="video", path=str(path), sha256=digest, frame_count=count, **common)
        )
        catalog.append(
            dict(
                asset_id=selected[i][1]["query_id"],
                kind="query",
                path=annotation["path"],
                sha256=annotation["sha256"],
                frame_count=None,
                **common,
            )
        )
    catalog_ref = write_input(
        directory / "catalog.json",
        dict(
            schema_version=1,
            origin_split="source_train",
            protocol_id=preparation["protocol_id"],
            allowlist=allow,
            source_report=report,
            entries=catalog,
        ),
    )
    return dict(
        official_video_count=len(ids),
        caption_count=len(rows),
        query_count=len(pilot_rows),
        decoded_excluded_caption_count=0,
        inputs=dict(
            catalog=catalog_ref,
            annotation=annotation,
            allowlist=allow,
            source_report=report,
            full_snapshot=full,
            training_records=records,
        ),
    )


def prepare_assignment(catalog, labels, *, report_path):
    try:
        assignment = assign_pilot_splits(catalog)
    except ValueError as exc:
        videos = [e for e in catalog if e.kind == "video"]
        write_input(
            report_path,
            dict(
                schema_version=1,
                ready=False,
                stop_reasons=[str(exc)],
                relevance_space="benchmark_annotation",
                semantic_review_performed=False,
                source_counts=dict(
                    official_video_count=len({e.official_video_id for e in videos}),
                    canonical_video_count=len({e.canonical_id for e in videos}),
                    query_count=sum(e.kind == "query" for e in catalog),
                ),
            ),
        )
        raise
    summary = summarize_pilot_inputs(catalog=catalog, assignment=assignment, labels=labels)
    write_input(report_path, summary)
    return assignment, summary


def execute_preparation(runtime, preparation, *, repo_root):
    configure_fp32()
    if (
        not isinstance(preparation, dict)
        or set(preparation) != PREPARATION_KEYS
        or preparation["mode"] not in {"snapshot", "inventory", "compile"}
        or preparation["relevance_space"] != "benchmark_annotation"
        or not preparation["protocol_id"]
    ):
        raise ValueError("invalid preparation mode/label scope")
    context = prepare_run(runtime, repo_root=repo_root)
    expected = {
        "eval_split": runtime.eval_split,
        "qualification_scenario": f"pilot_prepare:{identity_digest(preparation)}",
    }
    if context.manifest["config"]["evaluation"] != expected:
        raise ValueError("preparation configuration differs from reservation")
    variant = "benchmark_compile" if preparation["mode"] == "compile" else "source_inventory"
    if context.manifest["variant"] != variant:
        raise ValueError("preparation mode/variant mismatch")
    snapshot = preparation["mode"] == "snapshot"
    required = (
        {"source_confirmation", "allowlist_csv", "source_annotation", "video_directory"} if snapshot else {"catalog"}
    )
    required.add("source_confirmation")
    if preparation["mode"] == "compile":
        required.add("annotation")
    for key in PREPARATION_KEYS - {"mode", "protocol_id", "relevance_space", "allowed_root", "candidate_orders"}:
        if (preparation[key] is not None) != (key in required):
            raise ValueError("unused/missing preparation input must be explicitly null")
    for key in required - {"video_directory"}:
        _bound(context, preparation[key])
    if preparation["candidate_orders"] is not None:
        _bound(context, preparation["candidate_orders"])
    validate_source_confirmation(context, preparation)
    start_run(context)
    try:
        if snapshot:
            result = export_snapshot(context, preparation)
        else:
            catalog_meta = json.loads(Path(preparation["catalog"]["path"]).read_text())
            for key in ("allowlist", "source_report"):
                _bound(context, catalog_meta[key])
            report_meta = json.loads(Path(catalog_meta["source_report"]["path"]).read_text())
            for key in ("training_records", "allowlist_source"):
                _bound(context, report_meta[key])
            catalog = load_pilot_catalog(
                Path(preparation["catalog"]["path"]),
                expected_sha256=preparation["catalog"]["sha256"],
                allowed_root=Path(preparation["allowed_root"]),
                source_confirmation=preparation["source_confirmation"],
            )
            refs = context_input_references(context)
            if any(refs.get(str(e.path)) != e.sha256 for e in catalog):
                raise ValueError("catalog assets must all be reserved")
            result = dict(
                official_video_count=sum(e.kind == "video" for e in catalog),
                query_count=sum(e.kind == "query" for e in catalog),
            )
            if preparation["mode"] == "compile":
                label = load_benchmark_labels(
                    Path(preparation["annotation"]["path"]),
                    catalog=catalog,
                    expected_sha256=preparation["annotation"]["sha256"],
                    protocol_id=preparation["protocol_id"],
                )
                with ThreadPoolExecutor(max_workers=8) as executor:
                    list(
                        executor.map(
                            lambda entry: inspect_training_video(
                                entry.path, expected_sha256=entry.sha256, expected_frame_count=entry.frame_count
                            ),
                            (entry for entry in catalog if entry.kind == "video"),
                        )
                    )
                assignment, summary = prepare_assignment(
                    catalog, label, report_path=context.paths.provenance / "input-summary.json"
                )
                label_refs = {}
                for name, table in [
                    ("relations", label.official_positives_by_query),
                    ("id_map", label.official_to_canonical),
                ]:
                    label_refs[name] = write_input(context.paths.provenance / f"{name}.json", table)
                orders = (
                    None
                    if preparation["candidate_orders"] is None
                    else json.loads(Path(preparation["candidate_orders"]["path"]).read_text())
                )
                compiled = compile_pilot_manifests(
                    catalog=catalog,
                    assignment=assignment,
                    labels=label,
                    input_summary=summary,
                    preprocessing=EpisodePreprocessing(32, 8, 224),
                    label_references=label_refs,
                    candidate_orders=orders,
                )
                manifests = {}
                all_episodes = []
                for split, group in compiled.items():
                    manifests[split] = {}
                    for kind, value in group.items():
                        manifests[split][kind] = write_input(context.paths.provenance / f"{split}-{kind}.json", value)
                    ep = group["episodes"]
                    all_episodes.extend(
                        read_episodes(
                            Path(manifests[split]["episodes"]["path"]),
                            gallery_version=ep["gallery_version"],
                            split=split,
                        )
                    )
                audit_episode_splits(all_episodes)
                result.update(
                    summary=summary,
                    manifests=manifests,
                    assignment_sha256=identity_digest(asdict(assignment)),
                    label_references=label_refs,
                )
                write_input(context.paths.provenance / "split-assignment.json", asdict(assignment))
        write_input(context.paths.provenance / "preparation-result.json", result)
        publish_metrics(
            context,
            dict(
                split=runtime.eval_split,
                summary_metrics={k: float(v) for k, v in result.items() if type(v) is int},
                array_references=[],
            ),
        )
        complete_run(context)
        return result
    except BaseException:
        fail_run(context, exit_code=1)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    for field in ("run-id", "manifest-sha256", "source-sha"):
        parser.add_argument(f"--{field}")
    args = parser.parse_args(argv)
    payload = json.loads(args.config.read_text())
    if set(payload) != {"runtime", "preparation"}:
        raise ValueError("preparation config fields mismatch")
    rt = dict(payload["runtime"])
    for field in ("study_path", "registry_path", "run_dir"):
        rt[field] = Path(rt[field])
    runtime = RuntimeConfig(**rt)
    for field in ("run_id", "manifest_sha256", "source_sha"):
        if getattr(args, field) is not None and getattr(args, field) != getattr(runtime, field):
            raise ValueError("CLI identity mismatch")
    execute_preparation(runtime, payload["preparation"], repo_root=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
