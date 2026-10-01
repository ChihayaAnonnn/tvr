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
