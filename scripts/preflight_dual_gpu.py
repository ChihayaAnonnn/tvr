#!/usr/bin/env python3
"""Run the separately archived two-GPU gradient/checkpoint software preflight."""

import argparse
import datetime
import json
import os
import signal
import subprocess
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from experiment_artifacts import manifest_sha256
from scripts.qualify_semantic_retrieval import local_device_uuid
from semantic_retrieval.config import RuntimeConfig, config_payload, load_system_config
from semantic_retrieval.runtime.artifacts import complete_run, fail_run, prepare_run, publish_metrics, start_run
from semantic_retrieval.runtime.distributed import configure_fp32
from semantic_retrieval.runtime.dual_gpu_preflight import PROTOCOL, run_preflight, validate_report, validate_scope
from semantic_retrieval.runtime.pilot_candidates import _bound, file_reference, write_json
from semantic_retrieval.runtime.pilot_protocol import verify_phase_evidence


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    control = repo / "experiments" / PROTOCOL
    study = yaml.safe_load((control / "study.yaml").read_text())
    validate_scope(manifest, study)
    configure_fp32()
    rank = int(os.environ.get("RANK", "-1"))
    if (
        int(os.environ.get("WORLD_SIZE", "0")) != 2
        or rank not in (0, 1)
        or rank != int(os.environ.get("LOCAL_RANK", "-1"))
    ):
        raise ValueError("preflight needs exactly two local CUDA ranks")
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"], text=True
    )
    rows = {int(r.split(",")[0]): [s.strip() for s in r.split(",")[1:]] for r in hardware.splitlines()}
    uuids = [rows[i][0] for i in (3, 4)]
    if (
        uuids != study["gpu_uuids"]
        or os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") != uuids
        or any("A800" not in rows[i][1] for i in (3, 4))
        or local_device_uuid(rank) != uuids[rank]
    ):
        raise ValueError("preflight CUDA/physical UUID identity differs")
    runtime = RuntimeConfig(
        control / "study.yaml",
        control / "registry.jsonl",
        args.run_dir,
        manifest["run_id"],
        manifest_sha256(manifest),
        manifest["git_sha"],
        "train",
    )
    cfg = replace(load_system_config(args.template), runtime=runtime)
    if cfg.model.clip.fixture or cfg.train.pilot is not None or cfg.model.pilot_arm != "P1":
        raise ValueError("preflight requires standalone formal-capacity P1 config")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=8))
    context = None
    try:
        context = prepare_run(runtime, repo_root=repo, system_config=cfg)
        _bound(context, file_reference(args.template))
        # Authenticate the completed producer and exactly its full input inventory.
        parent = verify_phase_evidence(repo, study["source_parent"], "Q0P")["runs"][0]
        if parent["manifest"]["run_id"] not in manifest["parent_run_ids"]:
            raise ValueError("synthetic producer is not a declared parent")
        inventory = json.loads((parent["paths"].provenance / "pilot-input-inventory.json").read_text())
        expected = {r["path"]: r["sha256"] for r in inventory["inputs"]}
        expected[str(args.template)] = file_reference(args.template)["sha256"]
        expected[str(repo / study["source_parent"]["path"])] = study["source_parent"]["sha256"]
        actual = {r["path"]: r["sha256"] for f in ("input_checkpoints", "dataset_splits") for r in manifest[f]}
        if (
            actual != expected
            or inventory.get("random_weights") is not True
            or inventory.get("software_only") is not True
        ):
            raise ValueError("preflight must bind its complete software-only producer inventory")
        original = load_system_config(parent["paths"].provenance / "pilot-system-template.json")
        expected_cfg = replace(original, runtime=runtime, train=replace(original.train, pilot=None))
        if cfg != expected_cfg:
            raise ValueError("preflight configuration differs from approved producer projection")

        def interrupted(signum, frame):
            raise InterruptedError(f"preflight signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        dist.barrier()
        if rank == 0:
            write_json(context.paths.provenance / "system-config.json", config_payload(cfg))
            write_json(
                context.paths.provenance / "hardware.json",
                dict(physical_indices=[3, 4], gpu_uuids=uuids, nvidia_smi=hardware),
            )
            start_run(context)
        dist.barrier()
        report = run_preflight(context, cfg, device=torch.device("cuda", rank))
        validate_report(report, require_cuda=True)
        dist.barrier()
        if rank == 0:
            publish_metrics(
                context,
                dict(
                    software_only=True,
                    summary_metrics=dict(
                        resume_parameter_max_abs_delta=report["resume"]["parameter_max_abs_delta"],
                        resume_optimizer_max_abs_delta=report["resume"]["optimizer_max_abs_delta"],
                    ),
                ),
            )
            complete_run(context)
        dist.barrier()
    except BaseException:
        if rank == 0 and context is not None:
            fail_run(context, exit_code=1)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
