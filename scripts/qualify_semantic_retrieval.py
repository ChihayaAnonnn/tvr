#!/usr/bin/env python3
"""Archived synthetic qualification; never a benchmark/scientific launcher."""

import argparse
import datetime
import json
import os
import signal
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from experiment_artifacts import manifest_sha256, parse_registry_text, validate_metrics_document
from semantic_retrieval.config import RuntimeConfig, load_system_config
from semantic_retrieval.runtime.artifacts import complete_run, fail_run, prepare_run, publish_metrics, start_run
from semantic_retrieval.runtime.distributed import configure_fp32
from semantic_retrieval.runtime.gpu_qualification import GALLERY, build_fixture_inputs, qualify_chain, require_four_gpu, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare-production", "four-gpu"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    configure_fp32()
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    registry = repo / manifest["registry_path"]
    rt = RuntimeConfig(registry.parent / "study.yaml", registry, args.run_dir,
                       manifest["run_id"], manifest_sha256(manifest), manifest["git_sha"],
                       "val" if args.mode == "prepare-production" else "train")
    policy = yaml.safe_load(rt.study_path.read_text())["rebuilt_code_qualification"]
    if policy["scope"] != "synthetic_software_only" or policy["test_authorized"] is not False or manifest["test_authorized"]:
        raise ValueError("qualification must be synthetic and test sealed")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    context = None
    try:
        if args.mode == "prepare-production":
            if args.config is not None or world != 1 or torch.cuda.device_count() != 0:
                raise ValueError("input preparation must run once with CUDA visibility disabled")
            context = prepare_run(rt, repo_root=repo)
            if context.manifest["config"]["evaluation"].get("qualification_scenario") != "production_fixture":
                raise ValueError("input preparation scenario differs from reservation")
        else:
            if args.config is None:
                raise ValueError("qualification requires bound system config")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=8))
            require_four_gpu()
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if visible != [str(i) for i in policy["physical_indices"]]:
                raise ValueError("physical GPU mapping differs from frozen qualification policy")
            info = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
                                  check=True, capture_output=True, text=True).stdout
            rows = {int(row.split(",")[0]): tuple(x.strip() for x in row.split(",")[1:]) for row in info.splitlines()}
            if [rows[i][0] for i in policy["physical_indices"]] != policy["gpu_uuids"]:
                raise ValueError("GPU UUID mapping differs from frozen qualification policy")
            cfg = load_system_config(args.config)
            if (cfg.runtime != rt or cfg.model.clip.fixture or cfg.data is None
                or cfg.data.split != "train" or cfg.data.gallery_version != GALLERY):
                raise ValueError("qualification requires bound production capacity and synthetic inputs")
            context = prepare_run(rt, repo_root=repo, system_config=cfg)
            # Every input must come from this completed, archived synthetic preparation.
            if len(manifest["parent_run_ids"]) != 1:
                raise ValueError("qualification requires one archived input-preparation parent")
            from experiment_artifacts import load_validated_run_lifecycle, make_run_paths
            parent = make_run_paths(args.run_dir.parents[3], manifest["study"], manifest["parent_run_ids"][0])
            pm, ps, _ = load_validated_run_lifecycle(parent)
            if ps["state"] != "completed" or pm["config"]["evaluation"].get("qualification_scenario") != "production_fixture":
                raise ValueError("qualification parent is not completed synthetic preparation")
            committed = subprocess.run(["git", "show", f"HEAD:{manifest['registry_path']}"], cwd=repo,
                                       check=True, capture_output=True, text=True).stdout
            if not any(r["run_id"] == pm["run_id"] and r["state"] == "completed"
                       and r["manifest_sha256"] == manifest_sha256(pm) for r in parse_registry_text(committed)):
                raise ValueError("qualification parent terminal record must be committed")
            for ref in manifest["input_checkpoints"] + manifest["dataset_splits"]:
                path = Path(ref["path"])
                if not path.is_relative_to(parent.root) or ps["output_hashes"].get(path.relative_to(parent.root).as_posix()) != ref["sha256"]:
                    raise ValueError("qualification input differs from archived parent inventory")

        def interrupted(signum, frame):
            raise InterruptedError(f"qualification signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        if world > 1:
            dist.barrier()
        if rank == 0:
            start_run(context)
        if world > 1:
            dist.barrier()
        if args.mode == "prepare-production":
            _, refs = build_fixture_inputs(context)
            publish_metrics(context, dict(software_only=True, input_count=len(refs)))
            selected = dict(split="validation", summary_metrics=dict(software_input_count=len(refs)), array_references=[])
        else:
            if rank == 0:
                write_json(context.paths.provenance / "hardware.json", dict(
                    physical_indices=policy["physical_indices"], gpu_uuids=policy["gpu_uuids"],
                    cuda_visible_devices=visible, nvidia_smi=info, precision="fp32", tf32=False,
                    rank_mapping="local rank i maps to physical_indices[i]",
                ))
            result = qualify_chain(context, cfg, device=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
            peaks = [None] * world
            dist.all_gather_object(peaks, torch.cuda.max_memory_allocated())
            if rank == 0:
                # A rank table is metadata; numerical vectors go into external arrays.
                ref = context.array_store.write("rank_peak_memory_bytes", peaks).as_dict()
                publish_metrics(context, dict(software_only=True, summary_metrics={
                    k: float(result[k]) for k in ("checks_passed", "resume_parameter_max_abs_delta", "elapsed_seconds")
                }, array_references=[ref]))
                selected = dict(split="validation", summary_metrics={
                    k: float(result[k]) for k in ("checks_passed", "resume_parameter_max_abs_delta", "elapsed_seconds")
                }, array_references=[ref])
                print(json.dumps(result), flush=True)
        if rank == 0:
            write_json(context.paths.metrics / "selected_validation.json", validate_metrics_document(selected))
            complete_run(context)
        if world > 1:
            dist.barrier()
    except BaseException:
        if context is not None and rank == 0:
            fail_run(context, exit_code=1)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
