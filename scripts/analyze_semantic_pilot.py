#!/usr/bin/env python3
"""Reserved CPU C0 freeze or V0 benchmark pilot analysis worker."""

import argparse
import json
from pathlib import Path

from experiment_artifacts import manifest_sha256
from semantic_retrieval.config import RuntimeConfig
from semantic_retrieval.evaluation.pilot_analysis import _read_hashed, file_sha256, run_pilot_analysis
from semantic_retrieval.runtime.artifacts import complete_run, fail_run, prepare_run, publish_metrics, start_run
from semantic_retrieval.runtime.distributed import configure_fp32


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    configure_fp32()
    path = args.config.resolve()
    raw = json.loads(path.read_text())
    if set(raw) != {"bundle", "mode", "bootstrap_samples"} or raw["mode"] not in ("freeze", "locked"):
        raise ValueError("exact immutable analysis configuration required")
    repository = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    expected_split = "val_development" if raw["mode"] == "freeze" else "val_locked"
    registry = repository / manifest["registry_path"]
    config = RuntimeConfig(
        registry.parent / "study.yaml",
        registry,
        run_dir,
        manifest["run_id"],
        manifest_sha256(manifest),
        manifest["git_sha"],
        expected_split,
    )
    context = prepare_run(config, repo_root=repository)
    if context.manifest["config"]["evaluation"] != {
        "eval_split": expected_split,
        "qualification_scenario": "pilot_analysis",
    }:
        raise ValueError("analysis evaluation scenario differs from reservation")
    refs = {str(r["path"]): r["sha256"] for r in context.manifest["dataset_splits"]}
    if refs.get(str(path)) != file_sha256(path):
        raise ValueError("analysis configuration must be a reserved immutable input")
    bundle_path = _read_hashed(raw["bundle"])
    if refs.get(str(bundle_path)) != raw["bundle"]["sha256"]:
        raise ValueError("analysis bundle must be reserved")
    bundle = json.loads(bundle_path.read_text())
    required = [g["arrays"] for g in bundle["groups"]]
    required += [bundle[k] for k in ("frozen", "qualification") if k in bundle]
    for reference in required:
        if refs.get(reference["path"]) != reference["sha256"]:
            raise ValueError("every score/frozen/gate input must be reserved")
        _read_hashed(reference)
    start_run(context)
    try:
        result = run_pilot_analysis(context, bundle, mode=raw["mode"], bootstrap_samples=raw["bootstrap_samples"])
        # Full per-group detail stays in hashed run-local provenance documents.
        summary = dict(
            mode=raw["mode"],
            verdict=result.get("verdict", "frozen"),
            strongest_scalar=result["strongest_scalar"],
            array_references=result["array_references"][:400],
        )
        provenance = context.paths.provenance / (
            "pilot_frozen.json" if raw["mode"] == "freeze" else "pilot_analysis.json"
        )
        summary["analysis_reference"] = dict(
            path=provenance.relative_to(context.paths.root).as_posix(), sha256=file_sha256(provenance)
        )
        publish_metrics(context, summary)
        for offset in range(400, len(result["array_references"]), 400):
            publish_metrics(
                context,
                dict(
                    array_references=result["array_references"][offset : offset + 400],
                    analysis_reference=summary["analysis_reference"],
                ),
            )
        complete_run(context)
    except BaseException:
        fail_run(context, exit_code=1)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
