from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import experiment_tracking
import main_task_retrieval as retrieval
import dataloaders.msrvtt_protocol as msrvtt_protocol
from prob_models.uncertainty_protocol import file_sha256


def _rng_draws():
    return (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )


def _tiny_optimizer(total_steps=10):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = retrieval._build_warmup_cosine_scheduler(
        optimizer, warmup_proportion=0.1, total_steps=total_steps
    )
    return model, optimizer, scheduler


def _take_steps(model, optimizer, scheduler, count):
    lrs = []
    for _ in range(count):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr())
    return lrs


def test_registered_scheduler_is_one_five_epoch_trajectory_across_resume():
    torch.manual_seed(11)
    uninterrupted_model, uninterrupted_opt, uninterrupted_sched = _tiny_optimizer()
    uninterrupted_lrs = _take_steps(
        uninterrupted_model, uninterrupted_opt, uninterrupted_sched, 10
    )

    torch.manual_seed(11)
    resumed_model, resumed_opt, resumed_sched = _tiny_optimizer()
    first_lrs = _take_steps(resumed_model, resumed_opt, resumed_sched, 6)
    payload = retrieval._build_training_checkpoint_payload(
        epoch=2,
        optimizer=resumed_opt,
        scheduler=resumed_sched,
        tr_loss=0.5,
        global_step=6,
        scheduler_horizon_epochs=5,
        scheduler_total_steps=10,
        optimizer_steps_per_epoch=2,
        rng_state_by_rank={0: retrieval._capture_local_rng_state()},
        contextual_risk_diagnostic_window={
            "term_sums": {}, "term_count": 0, "gradient_total": 0.0,
            "optimizer_steps": 0, "zero_gradient_windows": 0,
        },
    )

    continued_model, continued_opt, continued_sched = _tiny_optimizer()
    continued_model.load_state_dict(resumed_model.state_dict())
    epoch, step = retrieval._restore_training_checkpoint(
        payload,
        continued_opt,
        continued_sched,
        require_complete=True,
        expected_scheduler_horizon_epochs=5,
        expected_scheduler_total_steps=10,
        expected_optimizer_steps_per_epoch=2,
        restore_rng=False,
        model=torch.nn.Module(),
    )
    second_lrs = _take_steps(continued_model, continued_opt, continued_sched, 4)

    assert (epoch, step) == (3, 6)
    assert first_lrs + second_lrs == uninterrupted_lrs
    assert payload["scheduler_horizon_epochs"] == 5
    assert payload["scheduler_total_steps"] == 10


def test_rng_round_trip_restores_python_numpy_and_torch_cpu():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = retrieval._capture_local_rng_state()
    expected = _rng_draws()
    _rng_draws()

    retrieval._restore_rng_state_for_current_rank({0: state}, rank=0)
    actual = _rng_draws()

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    assert set(state) == {"python", "numpy", "torch_cpu", "torch_cuda"}


def test_rng_round_trip_reproduces_the_next_dropout_forward_exactly():
    torch.manual_seed(23)
    dropout = torch.nn.Dropout(p=0.5).train()
    inputs = torch.ones(32)
    state = retrieval._capture_local_rng_state()
    expected = dropout(inputs)
    dropout(inputs)

    retrieval._restore_rng_state_for_current_rank({0: state}, rank=0)

    torch.testing.assert_close(dropout(inputs), expected, rtol=0, atol=0)


def test_a3_checkpoint_pair_is_published_only_through_a_hashed_marker(
    tmp_path, monkeypatch
):
    model, optimizer, scheduler = _tiny_optimizer()
    monkeypatch.setattr(
        retrieval,
        "logger",
        SimpleNamespace(info=lambda *_args, **_kwargs: None),
        raising=False,
    )
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        csd_logvar_sidecar_only=False,
        contextual_risk_sidecar_only=True,
    )

    model_path = Path(
        retrieval.save_model(
            0,
            args,
            model,
            optimizer,
            tr_loss=0.5,
            scheduler=scheduler,
            global_step=2,
            scheduler_horizon_epochs=5,
            scheduler_total_steps=10,
            optimizer_steps_per_epoch=2,
            rng_state_by_rank={0: retrieval._capture_local_rng_state()},
        )
    )
    optimizer_path = Path(retrieval._optimizer_checkpoint_path(0, args))
    marker_path = retrieval._checkpoint_publication_marker_path(0, args)
    marker = retrieval._validate_checkpoint_publication(
        marker_path,
        epoch=0,
        model_path=model_path,
        optimizer_path=optimizer_path,
    )

    assert marker["model_sha256"] == file_sha256(model_path)
    assert marker["optimizer_sha256"] == file_sha256(optimizer_path)
    assert not list(tmp_path.glob("*.staging"))


def test_a3_checkpoint_save_rejects_retained_incomplete_staging(tmp_path):
    stale = tmp_path / ".pytorch_model.bin.0.crash.staging"
    stale.write_bytes(b"crash evidence")
    model, optimizer, scheduler = _tiny_optimizer()
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        csd_logvar_sidecar_only=False,
        contextual_risk_sidecar_only=True,
    )

    with pytest.raises(FileExistsError, match="incomplete staged checkpoint"):
        retrieval.save_model(
            0,
            args,
            model,
            optimizer,
            tr_loss=0.5,
            scheduler=scheduler,
            global_step=2,
            scheduler_horizon_epochs=5,
            scheduler_total_steps=10,
            optimizer_steps_per_epoch=2,
            rng_state_by_rank={0: retrieval._capture_local_rng_state()},
        )
    assert stale.read_bytes() == b"crash evidence"


def test_a3_protocol_startup_rejects_retained_incomplete_staging(tmp_path):
    stale = tmp_path / ".pytorch_opt.bin.3.crash.staging"
    stale.write_bytes(b"crash evidence")

    with pytest.raises(ValueError, match="incomplete staged checkpoint"):
        retrieval._initialize_contextual_risk_protocol(
            SimpleNamespace(output_dir=str(tmp_path), resume_model=""),
            object(), object(), torch.device("cpu"), 1,
        )
    assert stale.read_bytes() == b"crash evidence"


def test_partial_checkpoint_replace_never_creates_publication_marker(
    tmp_path, monkeypatch
):
    model, optimizer, scheduler = _tiny_optimizer()
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        csd_logvar_sidecar_only=False,
        contextual_risk_sidecar_only=True,
    )
    monkeypatch.setattr(
        retrieval,
        "logger",
        SimpleNamespace(info=lambda *_args, **_kwargs: None),
        raising=False,
    )
    real_replace = retrieval.os.replace
    replacements = []

    def fail_second_checkpoint_replace(source, destination):
        destination = Path(destination)
        if destination.name.startswith("pytorch_"):
            replacements.append(destination)
            if len(replacements) == 2:
                raise OSError("simulated optimizer publication crash")
        return real_replace(source, destination)

    monkeypatch.setattr(retrieval.os, "replace", fail_second_checkpoint_replace)
    with pytest.raises(OSError, match="simulated optimizer publication crash"):
        retrieval.save_model(
            0, args, model, optimizer, tr_loss=0.5, scheduler=scheduler,
            global_step=2, scheduler_horizon_epochs=5,
            scheduler_total_steps=10, optimizer_steps_per_epoch=2,
            rng_state_by_rank={0: retrieval._capture_local_rng_state()},
        )

    assert (tmp_path / "pytorch_model.bin.0").is_file()
    assert not retrieval._checkpoint_publication_marker_path(0, args).exists()
    with pytest.raises(ValueError, match="publication marker"):
        retrieval._validate_checkpoint_publication(
            retrieval._checkpoint_publication_marker_path(0, args),
            epoch=0,
            model_path=tmp_path / "pytorch_model.bin.0",
            optimizer_path=tmp_path / "pytorch_opt.bin.0",
        )


def test_a3_candidate_recording_requires_a_valid_publication_marker(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "pytorch_model.bin.0"
    optimizer_path = tmp_path / "pytorch_opt.bin.0"
    torch.save({"contextual_risk_sidecar.weight": torch.ones(1)}, model_path)
    torch.save({"epoch": 0}, optimizer_path)
    monkeypatch.setattr(
        retrieval,
        "_build_contextual_risk_candidate",
        lambda **_kwargs: pytest.fail("unpublished pair reached candidate audit"),
    )

    with pytest.raises(ValueError, match="publication marker"):
        retrieval._record_contextual_risk_selection(
            epoch=0,
            checkpoint_path=model_path,
            optimizer_checkpoint_path=optimizer_path,
            validation_metrics={},
            paired_baseline={},
            reference_checkpoint=tmp_path / "a1.pt",
            candidate_path=tmp_path / "contextual_risk_candidates.json",
            selection_path=tmp_path / "best_contextual_risk_checkpoint.json",
            bootstrap_samples=0,
            expected_retrieval_config={},
            expected_head_configs={},
        )


def test_collect_git_state_ignores_untracked_but_reports_tracked_changes(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("original\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.py"], check=True)
    subprocess.run(
        [
            "git", "-C", str(tmp_path),
            "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "initial",
        ],
        check=True,
    )
    (tmp_path / "untracked.log").write_text("allowed\n")

    clean = experiment_tracking.collect_git_state(tmp_path)
    tracked.write_text("changed\n")
    dirty = experiment_tracking.collect_git_state(tmp_path)

    assert clean["dirty"] is False
    assert clean["modified_paths"] == []
    assert dirty["dirty"] is True
    assert dirty["modified_paths"] == ["tracked.py"]


def test_a3_resume_manifest_requires_same_commit_and_clean_tracked_state():
    base = {
        "protocol_version": "trusted-v1",
        "git": {"commit": "abc", "dirty": False, "modified_paths": []},
        "split": {}, "seed": 0, "profile": "hygiene",
        "run_final_test": True, "backbone": {}, "data": {}, "runtime": {},
        "batch": {}, "retrieval_risk": {}, "contextual_risk": {},
    }
    changed_commit = json.loads(json.dumps(base))
    changed_commit["git"]["commit"] = "def"
    dirty = json.loads(json.dumps(base))
    dirty["git"] = {"commit": "abc", "dirty": True, "modified_paths": ["x.py"]}

    with pytest.raises(ValueError, match="git commit"):
        experiment_tracking.validate_contextual_risk_resume_manifest(
            base, changed_commit
        )
    with pytest.raises(ValueError, match="clean tracked"):
        experiment_tracking.validate_contextual_risk_resume_manifest(base, dirty)


def _exact_fresh_a3_launcher_args(output_dir):
    repository = Path(__file__).resolve().parents[1]
    common_repository = Path("/data2/hxj/project/UATVR")
    data_root = Path("/data2/hxj/data/MSRVTT")
    generated = repository / "data/generated/msrvtt_trusted_v1"
    return [
        "--do_train",
        "--num_thread_reader", "12", "--prefetch_factor", "4", "--epochs=5",
        "--batch_size", "256", "--gradient_accumulation_steps", "1",
        "--n_display=20",
        "--train_csv", str(generated / "train.csv"),
        "--val_csv", str(generated / "val.csv"),
        "--source_train_csv", str(data_root / "csv/MSRVTT_train.9k.csv"),
        "--test_csv", str(data_root / "csv/MSRVTT_JSFUSION_test.csv"),
        "--split_manifest", str(
            repository / "dataloaders/splits/msrvtt_trusted_v1_seed0.json"
        ),
        "--eval_split", "val",
        "--data_path", str(data_root / "annotation/MSRVTT_v2.json"),
        "--features_path",
        f"{data_root}/videos/compressed_videos/msrvtt_224_12fps/",
        "--output_dir", str(output_dir),
        "--lr", "1e-4", "--max_words", "32", "--max_frames", "8",
        "--batch_size_val", "16", "--datatype", "msrvtt",
        "--expand_msrvtt_sentences", "--feature_framerate", "1",
        "--coef_lr", "1e-3", "--freeze_layer_num", "-1",
        "--slice_framepos", "3", "--linear_patch", "2d",
        "--sim_header", "seqTransf", "--pretrained_clip_name", "ViT-B/16",
        "--clip_layer_norm_precision", "fp32", "--clip_gradient_checkpointing",
        "--clip_visual_checkpoint_layers", "4", "--extra_video_cls_num", "2",
        "--extra_text_cls_num", "2", "--experiment_profile", "hygiene",
        "--experiment_desc", "A3 contextual retrieval-risk sidecar",
        "--use_uatvr_full", "--use_ada_norm", "--init_model", str(
            common_repository
            / "ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4"
        ),
        "--use_retrieval_risk_sidecar", "--w_retrieval_risk", "0.0",
        "--use_contextual_risk_sidecar", "--contextual_risk_sidecar_only",
        "--registered_contextual_risk_a3", "--registered_clip_asset_path", str(
            common_repository / ".cache/ViT-B-16.pt"
        ),
        "--registered_clip_asset_sha256",
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
        "--contextual_risk_topk", "16", "--contextual_risk_projection_dim", "128",
        "--contextual_risk_hidden_dim", "128", "--contextual_risk_temperature", "1.0",
        "--w_contextual_risk_fit", "0.25", "--w_contextual_risk_rank", "0.25",
        "--seed", "0", "--epochs", "3", "--run_final_test",
    ]


def _parse_exact_fresh_a3_args(monkeypatch, output_dir):
    monkeypatch.setattr(
        sys, "argv", ["main_task_retrieval.py", *_exact_fresh_a3_launcher_args(output_dir)]
    )
    args = retrieval.get_args()
    # These are established after parsing and before the production contract gate.
    args.world_size = 4
    args.n_gpu = 4
    args.pin_memory = True
    return args


def _clean_registered_launcher_environment():
    env = os.environ.copy()
    rejected = {
        "OUTPUT_DIR", "RUN_ID", "RUN_DATE", "RUN_TIME", "RUN_TAG",
        "NPROC", "TRAIN_BATCH_SIZE", "TRAIN_GRADIENT_ACCUMULATION_STEPS",
        "CLIP_LAYER_NORM_PRECISION", "EXPERIMENT_PROFILE", "TRAIN_PID_FILE",
        "RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER", "TVR_PYTHON", "TVR_TORCHRUN",
        "A800_THROUGHPUT_COMPARISON", "CLIP_GRADIENT_CHECKPOINTING",
        "CLIP_VISUAL_CHECKPOINT_LAYERS", "FREEZE_LAYER_NUM", "DATA_PATH",
        "CLIP_CACHE_DIR", "TRAIN_NUM_WORKERS", "TRAIN_PREFETCH_FACTOR",
        "MASTER_PORT", "MASTER_ADDR", "WORLD_SIZE", "RANK", "LOCAL_RANK",
        "GROUP_RANK", "ROLE_RANK", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NCCL_P2P_LEVEL",
        "NCCL_IB_DISABLE", "NCCL_DEBUG", "TORCH_WARN_ONCE", "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG", "SMOKE_STEPS", "EXPERIMENT_DESC",
        "REGISTERED_CLIP_ASSET_PATH", "REGISTERED_CLIP_ASSET_SHA256",
    }
    for name in list(env):
        if name in rejected or name.startswith(
            ("NCCL_", "TORCH_NCCL_", "CUDA_", "CUBLAS_")
        ) or name in {"NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"}:
            env.pop(name)
    return env


def test_exact_a3_args_are_composed_by_the_real_shell_launchers(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    contextual_script = repository / "run_train_msrvtt_contextual_risk_bg.sh"
    base_script = repository / "run_train_msrvtt_bg.sh"
    tail_capture = tmp_path / "contextual-tail.args"
    contextual_output = tmp_path / "contextual-capture-output"
    controller = tmp_path / "capture-controller.sh"
    controller.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\0' \"$@\" > \"${A3_ARGV_CAPTURE}\"\n"
        "printf '%s\\n' \"$$\" > \"${TRAIN_PID_FILE}\"\n"
        "mkdir -p \"${OUTPUT_DIR}\"\n"
        "sleep 1\n"
    )
    controller.chmod(0o755)
    env = _clean_registered_launcher_environment()
    env.update(
        {
                "A3_LAUNCHER_TEST_MODE": "1",
                "LEGACY_A3_LAYOUT": "1",
                "A3_BG_CONTROLLER": str(controller),
            "A3_TEST_OUTPUT_DIR": str(contextual_output),
            "A3_ARGV_CAPTURE": str(tail_capture),
        }
    )
    contextual = subprocess.run(
        ["bash", str(contextual_script)],
        cwd=repository,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert contextual.returncode == 0, contextual.stderr
    tail_args = [
        item.decode() for item in tail_capture.read_bytes().split(b"\0") if item
    ]
    contextual_output.rmdir()

    fake_python = tmp_path / "no-op-python.sh"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_python.chmod(0o755)
    full_capture = tmp_path / "full-worker.args"
    fake_torchrun = tmp_path / "capture-torchrun.sh"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\0' \"$@\" > \"${A3_ARGV_CAPTURE}\"\n"
    )
    fake_torchrun.chmod(0o755)
    worker_output = tmp_path / "worker-must-not-create-output"
    worker_env = os.environ.copy()
    worker_env.update(
        {
            "RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER": "1",
            "TVR_PYTHON": str(fake_python),
            "TVR_TORCHRUN": str(fake_torchrun),
            "A3_ARGV_CAPTURE": str(full_capture),
            "OUTPUT_DIR": str(worker_output),
            "RUN_ID": "contract_only_capture",
            "CUDA_VISIBLE_DEVICES": "0,2,3,4",
            "NPROC": "4",
            "TRAIN_BATCH_SIZE": "256",
            "TRAIN_GRADIENT_ACCUMULATION_STEPS": "1",
            "TRAIN_NUM_WORKERS": "12",
            "TRAIN_PREFETCH_FACTOR": "4",
            "CLIP_LAYER_NORM_PRECISION": "fp32",
            "CLIP_GRADIENT_CHECKPOINTING": "1",
            "CLIP_VISUAL_CHECKPOINT_LAYERS": "4",
            "FREEZE_LAYER_NUM": "-1",
            "A800_THROUGHPUT_COMPARISON": "0",
            "DATA_PATH": "/data2/hxj/data/MSRVTT",
            "EXPERIMENT_PROFILE": "hygiene",
            "EXPERIMENT_DESC": "A3 contextual retrieval-risk sidecar",
            "CLIP_CACHE_DIR": "/data2/hxj/project/UATVR/.cache",
            "MASTER_PORT": "29547",
        }
    )
    worker = subprocess.run(
        ["bash", str(base_script), *tail_args],
        cwd=repository,
        env=worker_env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert worker.returncode == 0, worker.stderr
    full_args = [
        item.decode() for item in full_capture.read_bytes().split(b"\0") if item
    ]
    main_index = full_args.index(str(repository / "main_task_retrieval.py"))
    assert full_args[main_index + 1 :] == _exact_fresh_a3_launcher_args(
        worker_output
    )
    assert not worker_output.exists()


def test_registered_a3_exact_real_parser_contract_passes(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,3,4")
    args = _parse_exact_fresh_a3_args(monkeypatch, tmp_path / "unused-output")

    retrieval._assert_registered_a3_contract(args)


def test_registered_a3_python_contract_rejects_manifest_material_mismatch(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,3,4")
    args = _parse_exact_fresh_a3_args(monkeypatch, tmp_path / "unused-output")
    batch = {
        "world_size": 4,
        "requested_effective_batch": 256,
        "optimizer_effective_batch": 256,
        "gradient_accumulation_steps": 1,
        "optimizer_steps_per_epoch": 10,
        "total_optimizer_steps": 50,
    }
    retrieval._assert_registered_a3_contract(args, batch)

    args.batch_size_val = 32
    with pytest.raises(ValueError, match="batch_size_val"):
        retrieval._assert_registered_a3_contract(args, batch)


def test_registered_a3_contract_rejects_missing_required_parser_field(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,3,4")
    args = _parse_exact_fresh_a3_args(monkeypatch, tmp_path / "unused-output")
    del args.contextual_risk_topk

    with pytest.raises(ValueError, match="contextual_risk_topk.*required"):
        retrieval._assert_registered_a3_contract(args)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("do_eval", 1, "do_eval.*boolean"),
        ("contextual_risk_topk", "16", "contextual_risk_topk.*integer"),
        ("contextual_risk_temperature", "1.0", "contextual_risk_temperature.*float"),
        ("contextual_risk_temperature", 1, "contextual_risk_temperature.*float"),
    ],
)
def test_registered_a3_contract_rejects_wrong_semantic_types(
    monkeypatch, tmp_path, name, value, message,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,3,4")
    args = _parse_exact_fresh_a3_args(monkeypatch, tmp_path / "unused-output")
    setattr(args, name, value)

    with pytest.raises(ValueError, match=message):
        retrieval._assert_registered_a3_contract(args)


def test_registered_a3_contract_normalizes_paths_semantically(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,3,4")
    args = _parse_exact_fresh_a3_args(monkeypatch, tmp_path / "unused-output")
    args.train_csv = str(Path(args.train_csv).parent / ".." / "msrvtt_trusted_v1" / "train.csv")
    retrieval._assert_registered_a3_contract(args)

    args.train_csv = str(tmp_path / "other.csv")
    with pytest.raises(ValueError, match="train_csv"):
        retrieval._assert_registered_a3_contract(args)


def test_registered_a3_contract_normalizes_only_declared_missing_future_flag(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,3,4")
    args = _parse_exact_fresh_a3_args(monkeypatch, tmp_path / "unused-output")
    assert not hasattr(args, "intrinsic_concentration_only")
    retrieval._assert_registered_a3_contract(args)

    args.intrinsic_concentration_only = True
    with pytest.raises(ValueError, match="intrinsic_concentration_only"):
        retrieval._assert_registered_a3_contract(args)


def test_registered_a3_contract_schema_is_complete_for_real_parser(monkeypatch, tmp_path):
    args = _parse_exact_fresh_a3_args(monkeypatch, tmp_path / "unused-output")
    schema = retrieval._registered_a3_contract_schema()
    names = [field.name for field in schema]

    assert len(names) == len(set(names))
    assert {field.source for field in schema} == {"namespace", "derived", "environment"}
    assert all(field.expected is not retrieval._REGISTERED_A3_MISSING for field in schema)
    assert {
        field.name for field in schema if field.source == "namespace" and field.required
    } <= vars(args).keys()
    assert {
        field.name for field in schema if field.source == "derived"
    } <= vars(args).keys()

    required_schema_namespace_names = {
        field.name
        for field in schema
        if field.source in {"namespace", "derived"} and field.required
    }
    assert vars(args).keys() == (
        required_schema_namespace_names
        | retrieval._REGISTERED_A3_DYNAMIC_NAMESPACE_FIELDS
    )


def test_registered_a3_four_rank_contract_only_exact_launcher_probe(tmp_path):
    output_dir = tmp_path / "must-not-be-created"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,2,3,4"
    command = [
        "/home/xujie/.conda/envs/tvr/bin/torchrun",
        "--nproc_per_node=4",
        "--master_addr=127.0.0.9",
        "--master_port=29547",
        "--",
        "-c",
        (
            "import os; import main_task_retrieval as r; "
            "a=r.get_args(); a.world_size=int(os.environ['WORLD_SIZE']); "
            "a.n_gpu=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')); "
            "a.pin_memory=True; r._assert_registered_a3_contract(a); "
            "print('A3_CONTRACT_ONLY_OK', flush=True)"
        ),
        *_exact_fresh_a3_launcher_args(output_dir),
    ]

    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("A3_CONTRACT_ONLY_OK") == 4
    assert not output_dir.exists()


def _set_registered_a3_initial_environment(monkeypatch):
    expected_environment = retrieval._registered_a3_expected_environment()
    for name in list(os.environ):
        if name.startswith(("NCCL_", "TORCH_NCCL_", "CUDA_", "CUBLAS_")) or name in {
            "NVIDIA_TF32_OVERRIDE",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        }:
            monkeypatch.delenv(name)
    for name, value in expected_environment.items():
        monkeypatch.setenv(name, value)
    for name, value in {
        "ROLE_NAME": "default",
        "TORCHELASTIC_RESTART_COUNT": "0",
        "TORCHELASTIC_MAX_RESTARTS": "0",
        "TORCHELASTIC_RUN_ID": "none",
        "TORCHELASTIC_USE_AGENT_STORE": "True",
    }.items():
        monkeypatch.setenv(name, value)
    for name in ("RANK", "LOCAL_RANK", "ROLE_RANK"):
        monkeypatch.setenv(name, "0")
    return expected_environment


def _remove_torchrun_worker_topology(monkeypatch):
    for name in (
        "ROLE_NAME",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_registered_a3_initial_environment_is_checked_before_seed_normalization(
    monkeypatch,
):
    args = SimpleNamespace(contextual_risk_sidecar_only=True)
    _set_registered_a3_initial_environment(monkeypatch)
    retrieval._assert_registered_a3_initial_environment(args)

    monkeypatch.setenv("PYTHONHASHSEED", "9")
    with pytest.raises(ValueError, match="initial environment"):
        retrieval._assert_registered_a3_initial_environment(args)


@pytest.mark.parametrize(
    "name",
    [
        "CLIP_CACHE_DIR",
        "REGISTERED_CLIP_ASSET_SHA256",
        "CUBLAS_WORKSPACE_CONFIG",
        "NCCL_IB_DISABLE",
        "OMP_NUM_THREADS",
        "TVR_PYTHON",
    ],
)
def test_registered_a3_initial_environment_rejects_missing_material_value(
    monkeypatch, name
):
    args = SimpleNamespace(contextual_risk_sidecar_only=True)
    _set_registered_a3_initial_environment(monkeypatch)
    monkeypatch.delenv(name)

    with pytest.raises(ValueError, match="initial environment"):
        retrieval._assert_registered_a3_initial_environment(args)


@pytest.mark.parametrize(
    "name",
    [
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NCCL_P2P_DISABLE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "NVIDIA_TF32_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "CUBLAS_LOGINFO_DBG",
        "CUDA_LAUNCH_BLOCKING",
    ],
)
def test_registered_a3_initial_environment_rejects_extra_accelerator_override(
    monkeypatch, name
):
    args = SimpleNamespace(contextual_risk_sidecar_only=True)
    _set_registered_a3_initial_environment(monkeypatch)
    monkeypatch.setenv(name, "1")

    with pytest.raises(ValueError, match=rf"initial environment.*{name}"):
        retrieval._assert_registered_a3_initial_environment(args)


def test_registered_a3_initial_environment_rejects_direct_async_env_spoof(
    monkeypatch,
):
    args = SimpleNamespace(contextual_risk_sidecar_only=True)
    _set_registered_a3_initial_environment(monkeypatch)
    _remove_torchrun_worker_topology(monkeypatch)
    monkeypatch.setenv("NCCL_ASYNC_ERROR_HANDLING", "1")

    with pytest.raises(ValueError, match="torchrun worker topology"):
        retrieval._assert_registered_a3_initial_environment(args)


def test_registered_a3_initial_environment_rejects_wrong_torchrun_async_value(
    monkeypatch,
):
    args = SimpleNamespace(contextual_risk_sidecar_only=True)
    _set_registered_a3_initial_environment(monkeypatch)
    monkeypatch.setenv("NCCL_ASYNC_ERROR_HANDLING", "0")

    with pytest.raises(
        ValueError,
        match=r"NCCL_ASYNC_ERROR_HANDLING='0'.*registered '1'",
    ):
        retrieval._assert_registered_a3_initial_environment(args)


def test_registered_a3_canonical_torchrun_worker_environment_passes():
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("NCCL_", "TORCH_NCCL_", "CUDA_", "CUBLAS_")) or name in {
            "NVIDIA_TF32_OVERRIDE",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        }:
            env.pop(name)
    worker_owned = {
        "RANK",
        "LOCAL_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "GROUP_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
        "NCCL_ASYNC_ERROR_HANDLING",
    }
    env.update(
        {
            name: value
            for name, value in retrieval._registered_a3_expected_environment().items()
            if name not in worker_owned
        }
    )
    command = [
        "/home/xujie/.conda/envs/tvr/bin/torchrun",
        "--nproc_per_node=4",
        "--master_addr=127.0.0.9",
        "--master_port=29547",
        "--",
        "-c",
        (
            "from types import SimpleNamespace; "
            "import main_task_retrieval as retrieval; "
            "retrieval._assert_registered_a3_initial_environment("
            "SimpleNamespace(contextual_risk_sidecar_only=True)); "
            "print('A3_ENV_OK', flush=True)"
        ),
    ]

    assert "NCCL_ASYNC_ERROR_HANDLING" not in env
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in env
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("A3_ENV_OK") == 4


def test_registered_a3_accelerator_environment_exact_whitelist_passes(monkeypatch):
    args = SimpleNamespace(contextual_risk_sidecar_only=True)
    expected = _set_registered_a3_initial_environment(monkeypatch)
    material_names = {
        name
        for name in expected
        if name.startswith(("NCCL_", "CUDA_", "CUBLAS_"))
    }

    assert material_names == {
        "NCCL_ASYNC_ERROR_HANDLING",
        "NCCL_P2P_LEVEL",
        "NCCL_IB_DISABLE",
        "NCCL_DEBUG",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_MODULE_LOADING",
        "CUBLAS_WORKSPACE_CONFIG",
    }
    retrieval._assert_registered_a3_initial_environment(args)


def test_registered_clip_asset_preflight_rejects_path_hash_and_cache_drift(
    tmp_path, monkeypatch
):
    asset = tmp_path / "ViT-B-16.pt"
    asset.write_bytes(b"clip-fixture")
    digest = msrvtt_protocol.sha256_file(asset)
    monkeypatch.setattr(retrieval, "REGISTERED_A3_CLIP_ASSET_PATH", asset)
    monkeypatch.setattr(retrieval, "REGISTERED_A3_CLIP_ASSET_SHA256", digest)
    monkeypatch.setattr(retrieval, "REGISTERED_A3_CLIP_CACHE_DIR", tmp_path)
    monkeypatch.setenv("CLIP_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("REGISTERED_CLIP_ASSET_PATH", str(asset))
    monkeypatch.setenv("REGISTERED_CLIP_ASSET_SHA256", digest)
    args = SimpleNamespace(
        registered_contextual_risk_a3=True,
        contextual_risk_sidecar_only=True,
        registered_clip_asset_path=str(asset),
        registered_clip_asset_sha256=digest,
    )
    assert retrieval._preflight_registered_a3_clip_asset(
        args, verify_load=False
    )["sha256"] == digest

    args.registered_clip_asset_sha256 = "0" * 64
    with pytest.raises(ValueError, match="CLI SHA-256"):
        retrieval._preflight_registered_a3_clip_asset(args, verify_load=False)
    args.registered_clip_asset_sha256 = digest
    monkeypatch.setenv("CLIP_CACHE_DIR", str(tmp_path / "other"))
    with pytest.raises(ValueError, match="CLIP_CACHE_DIR"):
        retrieval._preflight_registered_a3_clip_asset(args, verify_load=False)
    monkeypatch.setenv("CLIP_CACHE_DIR", str(tmp_path))
    asset.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="asset SHA-256"):
        retrieval._preflight_registered_a3_clip_asset(args, verify_load=False)


def _tiny_split_manifest():
    return {
        "protocol_version": "trusted-v1",
        "seed": 0,
        "algorithm": "fixture deterministic split",
        "val_size": 1,
        "expected_captions_per_video": 2,
        "source_sha256": {
            "train_csv": "a" * 64,
            "annotation_json": "b" * 64,
            "test_csv": "c" * 64,
        },
        "counts": {
            "source_train_videos": 2, "train_videos": 1,
            "val_videos": 1, "val_sentences": 2,
            "test_csv_rows": 0, "test_videos": 0,
        },
        "overlap_counts": {"train_val": 0, "train_test": 0, "val_test": 0},
        "test_video_ids_sha256": "d" * 64,
        "train_video_ids": ["video1"],
        "val_video_ids": ["video2"],
    }


def test_generated_split_validation_binds_exact_csv_bytes(tmp_path):
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({
        "videos": [{"video_id": "video1"}, {"video_id": "video2"}],
        "sentences": [
            {"video_id": "video2", "caption": "first"},
            {"video_id": "video2", "caption": "second"},
        ],
    }))
    generated = tmp_path / "generated"
    paths = msrvtt_protocol.write_generated_split_files(
        _tiny_split_manifest(), annotation, generated
    )
    summary = msrvtt_protocol.validate_generated_split_files(
        _tiny_split_manifest(),
        annotation,
        paths["train_csv"],
        paths["val_csv"],
    )

    assert summary["train_csv"]["path"] == str(
        Path(paths["train_csv"]).resolve()
    )
    assert summary["train_csv"]["sha256"] == msrvtt_protocol.sha256_file(
        paths["train_csv"]
    )
    assert summary["val_csv"]["sha256"] == msrvtt_protocol.sha256_file(
        paths["val_csv"]
    )
    assert summary["derivation"]["protocol_version"] == "trusted-v1"

    Path(paths["val_csv"]).write_text(
        Path(paths["val_csv"]).read_text() + "video2,tampered\n"
    )
    with pytest.raises(ValueError, match="generated val CSV"):
        msrvtt_protocol.validate_generated_split_files(
            _tiny_split_manifest(),
            annotation,
            paths["train_csv"],
            paths["val_csv"],
        )


def _advance_diagnostic_steps(model, start, stop, *, gradient):
    emitted = []
    for step in range(start, stop):
        retrieval._accumulate_contextual_risk_window(
            model, {"contextual_risk_loss": float(step)}
        )
        emitted.append(
            retrieval._finish_contextual_risk_window_step(model, gradient)
        )
    return emitted


def test_a3_diagnostic_window_checkpoint_resume_matches_uninterrupted():
    uninterrupted = torch.nn.Module()
    # Six optimizer steps per epoch makes this an exact 3-epoch + 2-epoch
    # continuation boundary while also crossing the 20-step diagnostic window.
    _advance_diagnostic_steps(uninterrupted, 0, 18, gradient=1.0)
    snapshot = retrieval._snapshot_contextual_risk_diagnostic_window(
        uninterrupted
    )
    model, optimizer, scheduler = _tiny_optimizer(total_steps=30)
    payload = retrieval._build_training_checkpoint_payload(
        epoch=2, optimizer=optimizer, scheduler=scheduler, tr_loss=0.5,
        global_step=18, scheduler_horizon_epochs=5,
        scheduler_total_steps=30, optimizer_steps_per_epoch=6,
        rng_state_by_rank={0: retrieval._capture_local_rng_state()},
        contextual_risk_diagnostic_window=snapshot,
    )

    expected = _advance_diagnostic_steps(uninterrupted, 18, 30, gradient=1.0)
    resumed = torch.nn.Module()
    _, restored_opt, restored_sched = _tiny_optimizer(total_steps=30)
    retrieval._restore_training_checkpoint(
        payload, restored_opt, restored_sched, require_complete=True,
        expected_scheduler_horizon_epochs=5,
        expected_scheduler_total_steps=30,
        expected_optimizer_steps_per_epoch=6,
        restore_rng=False,
        model=resumed,
    )
    actual = _advance_diagnostic_steps(resumed, 18, 30, gradient=1.0)

    assert actual == expected
    assert retrieval._snapshot_contextual_risk_diagnostic_window(resumed) == (
        retrieval._snapshot_contextual_risk_diagnostic_window(uninterrupted)
    )

    zero_source = torch.nn.Module()
    _advance_diagnostic_steps(zero_source, 0, 20, gradient=0.0)
    zero_snapshot = retrieval._snapshot_contextual_risk_diagnostic_window(
        zero_source
    )
    zero_restored = torch.nn.Module()
    retrieval._restore_contextual_risk_diagnostic_window(
        zero_restored, zero_snapshot
    )
    assert retrieval._snapshot_contextual_risk_diagnostic_window(
        zero_restored
    )["zero_gradient_windows"] == 1
    assert retrieval._snapshot_contextual_risk_diagnostic_window(
        zero_restored
    ) == zero_snapshot


@pytest.mark.parametrize(
    "bad_state",
    [
        {"term_sums": {}, "term_count": -1, "gradient_total": 0.0,
         "optimizer_steps": 0, "zero_gradient_windows": 0},
        {"term_sums": {"contextual_risk_loss": float("nan")},
         "term_count": 1, "gradient_total": 1.0,
         "optimizer_steps": 1, "zero_gradient_windows": 0},
        {"term_sums": {}, "term_count": 0, "gradient_total": 0.0,
         "optimizer_steps": 20, "zero_gradient_windows": 0},
        {"term_sums": {"contextual_risk_loss": "invalid"},
         "term_count": 1, "gradient_total": 1.0,
         "optimizer_steps": 1, "zero_gradient_windows": 0},
    ],
)
def test_a3_diagnostic_window_restore_rejects_invalid_schema(bad_state):
    with pytest.raises(ValueError, match="diagnostic window"):
        retrieval._restore_contextual_risk_diagnostic_window(
            torch.nn.Module(), bad_state
        )
