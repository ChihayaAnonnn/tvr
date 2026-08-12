from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import experiment_tracking
import main_task_retrieval as retrieval
from prob_models.uncertainty_protocol import file_sha256


def _rng_draws():
    return (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )


def _tiny_optimizer():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = retrieval._build_warmup_cosine_scheduler(
        optimizer, warmup_proportion=0.1, total_steps=10
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


def test_registered_a3_python_contract_rejects_manifest_material_mismatch(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,3,4")
    args = SimpleNamespace(**retrieval._registered_a3_expected_config())
    args.output_dir = "/tmp/a3-contract"
    args.init_model = str(experiment_tracking.CONTEXTUAL_RISK_CANONICAL_A1_CHECKPOINT)
    args.resume_model = None
    args.epochs = 3
    args.world_size = 4
    args.n_gpu = 4
    args.requested_effective_batch_size = 256
    args.batch_size = 256
    args.registered_contextual_risk_a3 = True
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


def test_registered_a3_initial_environment_is_checked_before_seed_normalization(
    monkeypatch,
):
    args = SimpleNamespace(contextual_risk_sidecar_only=True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,3,4")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    retrieval._assert_registered_a3_initial_environment(args)

    monkeypatch.setenv("PYTHONHASHSEED", "9")
    with pytest.raises(ValueError, match="initial environment"):
        retrieval._assert_registered_a3_initial_environment(args)
