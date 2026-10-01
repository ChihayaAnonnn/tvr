from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_runtime_artifacts import commit_reservation, reservation


def test_explicit_fp32_initialization():
    from semantic_retrieval.runtime.distributed import configure_fp32

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    configure_fp32()
    assert torch.get_float32_matmul_precision() == "highest"
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def test_generated_raw_fixture_and_full_chain(tmp_path):
    from semantic_retrieval.config import effective_manifest_config
    from semantic_retrieval.runtime.artifacts import complete_run, prepare_run, start_run
    from semantic_retrieval.runtime.gpu_qualification import build_fixture_inputs, qualify_chain

    repo, rt = reservation.__wrapped__(tmp_path, SimpleNamespace(param="success"))
    commit_reservation(repo, rt)
    prep = prepare_run(rt, repo_root=repo)
    start_run(prep)
    cfg, refs = build_fixture_inputs(prep, tiny=True)
    assert len(refs) == 7  # weights, three raw clips, three episode manifests
    assert cfg.model.clip_weights.suffix == ".npz"
    assert all(str(ref["path"]).startswith(str(rt.run_dir)) for ref in refs)
    complete_run(prep)

    # Separate, genuinely bound Scheme A training identity, not a patched context.
    case = tmp_path / "qualification"
    case.mkdir()
    repo2, rt2 = reservation.__wrapped__(case, SimpleNamespace(param=dict(
        config=effective_manifest_config(cfg, eval_split="train"),
        input_checkpoints=[refs[0]], dataset_splits=refs[1:], variant="chain",
    )))
    rt2 = replace(rt2, eval_split="train")
    cfg = replace(cfg, runtime=rt2)
    commit_reservation(repo2, rt2)
    context = prepare_run(rt2, repo_root=repo2, system_config=cfg)
    start_run(context)
    result = qualify_chain(context, cfg, device=torch.device("cpu"))
    assert result["checks_passed"] >= 10
    assert result["resume_parameter_max_abs_delta"] == 0
    assert result["unknown_stepped"] == 0
    assert result["expansion_budget_used"] == 3
    assert result["tamper_rejected"] == 1
    assert (rt2.run_dir / "provenance" / "qualification-report.json").is_file()


def test_four_gpu_guard_rejects_cpu():
    from semantic_retrieval.runtime.gpu_qualification import require_four_gpu

    with pytest.raises(ValueError, match="four CUDA"):
        require_four_gpu()


def _cpu_rank(rank, init, context, cfg):
    import torch.distributed as dist
    from semantic_retrieval.runtime.gpu_qualification import qualify_chain

    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=4)
    try:
        qualify_chain(context, cfg, device=torch.device("cpu"))
    finally:
        dist.destroy_process_group()


def test_four_rank_unknown_then_resume(tmp_path):
    from semantic_retrieval.config import effective_manifest_config
    from semantic_retrieval.runtime.artifacts import prepare_run, start_run
    from semantic_retrieval.runtime.gpu_qualification import build_fixture_inputs

    repo, rt = reservation.__wrapped__(tmp_path, SimpleNamespace(param="success"))
    commit_reservation(repo, rt)
    prep = prepare_run(rt, repo_root=repo)
    start_run(prep)
    cfg, refs = build_fixture_inputs(prep, tiny=True)
    case = tmp_path / "distributed"
    case.mkdir()
    repo2, rt2 = reservation.__wrapped__(case, SimpleNamespace(param=dict(
        config=effective_manifest_config(cfg, eval_split="train"),
        input_checkpoints=[refs[0]], dataset_splits=refs[1:], variant="four-rank",
    )))
    rt2 = replace(rt2, eval_split="train")
    cfg = replace(cfg, runtime=rt2)
    commit_reservation(repo2, rt2)
    context = prepare_run(rt2, repo_root=repo2, system_config=cfg)
    start_run(context)
    torch.multiprocessing.spawn(_cpu_rank, args=(f"file://{tmp_path / 'init'}", context, cfg), nprocs=4, join=True)


def test_local_clip_npz_loading_and_hash(model_config, tmp_path):
    import numpy as np
    from semantic_retrieval.data.episodes import file_sha256
    from semantic_retrieval.models.clip import ClipBackbone
    from semantic_retrieval.models.model import SemanticRetrievalModel

    state = torch.load(model_config.clip_weights, weights_only=True)
    path = tmp_path / "clip.npz"
    np.savez(path, **{k: v.numpy() for k, v in state.items()})
    cfg = replace(model_config, clip_weights=path, clip_sha256=file_sha256(path))
    model = SemanticRetrievalModel(cfg)
    assert all(torch.equal(model.clip.state_dict()[k], v) for k, v in state.items())
    with pytest.raises(ValueError, match="SHA-256"):
        SemanticRetrievalModel(replace(cfg, clip_sha256="0" * 64))


@pytest.mark.parametrize("defect", ["omitted", "tampered"])
def test_qualification_split_must_be_reserved_and_unchanged(tmp_path, defect):
    from semantic_retrieval.runtime.gpu_qualification import qualification_episodes

    path = tmp_path / "val_locked.json"
    path.write_text("{}")
    from semantic_retrieval.data.episodes import file_sha256
    refs = [] if defect == "omitted" else [dict(path=str(path), sha256=file_sha256(path))]
    if defect == "tampered":
        path.write_text('{"tampered":true}')
    cfg = SimpleNamespace(data=SimpleNamespace(episode_manifest=tmp_path / "train.json"))
    context = SimpleNamespace(manifest=dict(dataset_splits=refs))
    with pytest.raises(ValueError, match="reserved|hash"):
        qualification_episodes(context, cfg, "val_locked")


@pytest.mark.parametrize("defect", ["prefix", "loss", "hash"])
def test_qualification_template_enforces_core_chain(training_context, tmp_path, defect):
    from semantic_retrieval.config import config_payload
    from semantic_retrieval.data.episodes import file_sha256
    from semantic_retrieval.runtime.gpu_qualification import validate_qualification_template, write_json

    _, cfg = training_context
    path = tmp_path / "template.json"
    write_json(path, config_payload(cfg))
    digest = file_sha256(path)
    validate_qualification_template(cfg, path, digest)
    if defect == "prefix":
        cfg = replace(cfg, train=replace(cfg.train, trainable_prefixes=("matcher.",)))
    elif defect == "loss":
        cfg = replace(cfg, train=replace(cfg.train, loss_weights={**cfg.train.loss_weights, "exist": 0.0}))
    else:
        digest = "0" * 64
    with pytest.raises(ValueError, match="template"):
        validate_qualification_template(cfg, path, digest)


def test_gpu_visibility_requires_selected_uuid_order(monkeypatch):
    from scripts.qualify_semantic_retrieval import verify_visible_uuids

    policy = dict(gpu_uuids=["GPU-a", "GPU-b", "GPU-c", "GPU-d"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,3,4")
    with pytest.raises(ValueError, match="UUID"):
        verify_visible_uuids(policy)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    assert verify_visible_uuids(policy) == policy["gpu_uuids"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b,GPU-a,GPU-c,GPU-d")
    with pytest.raises(ValueError, match="UUID"):
        verify_visible_uuids(policy)
