import json

import pytest
import torch
from test_pilot_qualification import qualification_context


def _rank(rank, init, context, cfg):
    import torch.distributed as dist

    from semantic_retrieval.runtime.dual_gpu_preflight import run_preflight

    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=2)
    try:
        run_preflight(context, cfg, device=torch.device("cpu"))
    finally:
        dist.destroy_process_group()


def test_actual_two_rank_gradient_and_restore(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace

    from test_runtime_artifacts import commit_reservation, reservation

    from semantic_retrieval.config import effective_manifest_config
    from semantic_retrieval.runtime.artifacts import prepare_run, start_run
    from semantic_retrieval.runtime.dual_gpu_preflight import validate_report

    prep_dir = tmp_path / "prep"
    prep_dir.mkdir()
    prep, cfg = qualification_context(prep_dir)
    cfg = replace(cfg, train=replace(cfg.train, pilot=None))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    repo, runtime = reservation.__wrapped__(
        run_dir,
        SimpleNamespace(
            param=dict(
                variant="dual-software",
                config=effective_manifest_config(cfg, eval_split="train"),
                input_checkpoints=prep.manifest["input_checkpoints"],
                dataset_splits=prep.manifest["dataset_splits"],
            )
        ),
    )
    runtime = replace(runtime, eval_split="train")
    cfg = replace(cfg, runtime=runtime)
    commit_reservation(repo, runtime)
    context = prepare_run(runtime, repo_root=repo, system_config=cfg)
    start_run(context)
    torch.multiprocessing.spawn(_rank, args=(f"file://{tmp_path / 'init'}", context, cfg), nprocs=2, join=True)
    report = json.loads((context.paths.provenance / "dual-gpu-preflight.json").read_text())
    assert report["world_size"] == 2
    assert report["optimizer"] == "AdamW"
    assert report["resume"]["parameter_max_abs_delta"] == 0
    assert report["resume"]["optimizer_max_abs_delta"] == 0
    assert report["resume"]["sampler_restored"] and report["resume"]["rng_restored"]
    assert report["gradient_probes"]["A2"]["presence_gradient_norm"] == 0
    assert report["gradient_probes"]["A3"]["presence_gradient_norm"] > 0
    assert report["gradient_probes"]["A4"]["presence_gradient_norm"] == 0
    assert len(report["rank_records"]) == 2
    with pytest.raises(ValueError, match="CUDA"):
        validate_report(report, require_cuda=True)


@pytest.mark.parametrize("defect", ["gradients", "restore", "ranks"])
def test_report_rejects_false_success(defect):
    from semantic_retrieval.runtime.dual_gpu_preflight import validate_report

    report = dict(
        software_only=True,
        protocol="dual-gpu-preflight-v1",
        world_size=2,
        optimizer="AdamW",
        device="cpu",
        passed=True,
        rank_records=[{"rank": 0}, {"rank": 1}],
        resume=dict(
            parameter_max_abs_delta=0.0,
            optimizer_max_abs_delta=0.0,
            sampler_restored=True,
            rng_restored=True,
            scheduler_restored=True,
            rank_consistency=True,
        ),
        gradient_probes={
            a: dict(matching_gradient_norm=1.0, presence_gradient_norm=float(a == "A3"), update_performed=True)
            for a in ("A2", "A3", "A4")
        },
    )
    if defect == "gradients":
        report["gradient_probes"]["A2"]["presence_gradient_norm"] = 1.0
    if defect == "restore":
        report["resume"]["sampler_restored"] = False
    if defect == "ranks":
        report["rank_records"][1]["rank"] = 0
    with pytest.raises(ValueError):
        validate_report(report, require_cuda=False)


@pytest.mark.parametrize("defect", ["science", "four", "test", "variant"])
def test_worker_refuses_unregistered_scope(defect):
    from semantic_retrieval.runtime.dual_gpu_preflight import validate_scope

    manifest = dict(
        study="dual-gpu-preflight-v1",
        registry_path="experiments/dual-gpu-preflight-v1/registry.jsonl",
        dataset="synthetic",
        stage="baseline",
        variant="gradient-resume",
        test_authorized=False,
        environment=dict(gpu_count=2, precision="fp32", distributed_topology="torchrun-2"),
    )
    study = dict(software_only=True, scientific_authorized=False, test_authorized=False, physical_indices=[3, 4])
    if defect == "science":
        study["scientific_authorized"] = True
    if defect == "four":
        manifest["environment"]["gpu_count"] = 4
    if defect == "test":
        manifest["test_authorized"] = True
    if defect == "variant":
        manifest["variant"] = "train-science"
    with pytest.raises(ValueError):
        validate_scope(manifest, study)


def test_reservation_policy_cannot_ignore_new_software_namespace(tmp_path):
    from semantic_retrieval.runtime.pilot_policy import assert_pilot_authorized

    with pytest.raises(ValueError, match="study"):
        assert_pilot_authorized(
            repo_root=tmp_path,
            registry_path=tmp_path / "experiments/dual-gpu-preflight-v1/registry.jsonl",
            manifest=dict(
                study="dual-gpu-preflight-v1",
                artifact_path=str(tmp_path / "artifacts/dual-gpu-preflight-v1/synthetic/baseline/run"),
            ),
            operation="reserve",
        )


def test_foreign_study_cannot_use_dual_registry(tmp_path):
    from semantic_retrieval.runtime.pilot_policy import assert_pilot_authorized

    with pytest.raises(ValueError):
        assert_pilot_authorized(
            repo_root=tmp_path,
            registry_path=tmp_path / "experiments/dual-gpu-preflight-v1/registry.jsonl",
            manifest=dict(study="foreign", artifact_path=str(tmp_path / "artifacts/foreign/synthetic/baseline/run")),
            operation="reserve",
        )
