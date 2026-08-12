import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import main_task_retrieval
from main_task_retrieval import (
    _contextual_risk_should_continue,
    _initialize_contextual_risk_protocol,
    _maybe_run_contextual_risk_final_test,
    _record_contextual_risk_selection,
    _restore_training_checkpoint,
    save_model,
)
from prob_models.uncertainty_protocol import file_sha256


def _paired_digest(raw="raw"):
    return {
        "raw_wti_sha256": raw,
        "a1_failure_risk_sha256": {"t2v": "tf", "v2t": "vf"},
        "a1_vacuity_sha256": {"t2v": "tv", "v2t": "vv"},
        "retrieval": {"t2v": {"R1": 1.0}, "v2t": {"R1": 2.0}},
    }


def _direction(rho=0.30, auroc=0.70, aurc=0.25):
    return {
        "spearman_risk_vs_rank": rho,
        "failure_auroc": {"value": auroc, "ci_low": 0.51, "ci_high": 0.80},
        "risk_coverage": {
            "accuracy_at_0.25": 0.90,
            "accuracy_at_0.50": 0.80,
            "accuracy_at_0.75": 0.70,
            "accuracy_at_1.00": 0.60,
        },
        "risk_stats": {"mean": 0.40, "std": 0.02},
        "aurc": aurc,
    }


def _validation(rho=0.30, auroc=0.70, aurc=0.25, *, raw="raw"):
    return {
        "t2v": {"R1": 1.0},
        "v2t": {"R1": 2.0},
        "paired_digest": _paired_digest(raw),
        "contextual_risk": {
            "t2v": _direction(rho, auroc, aurc),
            "v2t": _direction(rho, auroc, aurc),
        },
        "contextual_risk_evidence": {
            "t2v": {"risk": [0.1, 0.2], "rank": [1, 2]},
            "v2t": {"risk": [0.2, 0.1], "rank": [2, 1]},
        },
    }


def _write_pair(run, reference, epoch):
    model_path = run / f"pytorch_model.bin.{epoch}"
    optimizer_path = run / f"pytorch_opt.bin.{epoch}"
    torch.save(
        {
            "backbone": torch.tensor([1.0]),
            "contextual_risk_sidecar.weight": torch.tensor([float(epoch)]),
        },
        model_path,
    )
    payload = {
        "epoch": epoch,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"last_epoch": epoch + 1},
        "global_step": (epoch + 1) * 5,
    }
    torch.save(payload, optimizer_path)
    return model_path, optimizer_path, payload


def _record(run, reference, candidates, epoch, metrics):
    model, optimizer, payload = _write_pair(run, reference, epoch)
    winner = _record_contextual_risk_selection(
        epoch=epoch,
        checkpoint_path=model,
        optimizer_checkpoint_path=optimizer,
        validation_metrics=metrics,
        paired_baseline=_paired_digest(),
        reference_checkpoint=reference,
        candidate_path=run / "contextual_risk_candidates.json",
        selection_path=run / "best_contextual_risk_checkpoint.json",
    )
    candidates[:] = json.loads(
        (run / "contextual_risk_candidates.json").read_text()
    )
    return winner, model, optimizer, payload


def test_initial_three_candidates_are_append_only_and_drive_strict_continuation(
    tmp_path,
):
    run = tmp_path / "run"
    run.mkdir()
    reference = tmp_path / "a1.pt"
    torch.save({"backbone": torch.tensor([1.0])}, reference)
    candidates = []

    for epoch, rho in enumerate((0.20, 0.21, 0.22)):
        _record(run, reference, candidates, epoch, _validation(rho=rho))

    assert [item["epoch"] for item in candidates] == [0, 1, 2]
    for candidate in candidates:
        assert candidate["model_sha256"] == file_sha256(candidate["path"])
        assert candidate["optimizer_sha256"] == file_sha256(
            candidate["optimizer_path"]
        )
        assert candidate["non_a3_mismatches"] == []
        assert candidate["paired_invariance"]["passed"] is True
        assert candidate["finite"] is True
        assert candidate["evidence"]
        assert "metrics" in candidate and "acceptance" in candidate
    assert _contextual_risk_should_continue(candidates) is True

    not_improved = copy.deepcopy(candidates)
    not_improved[2]["rho"] = list(not_improved[1]["rho"])
    not_improved[2]["auroc"] = list(not_improved[1]["auroc"])
    not_improved[2]["aurc"] = list(not_improved[1]["aurc"])
    assert _contextual_risk_should_continue(not_improved) is False

    with pytest.raises(ValueError, match="append-only"):
        _record_contextual_risk_selection(
            epoch=2,
            checkpoint_path=candidates[2]["path"],
            optimizer_checkpoint_path=candidates[2]["optimizer_path"],
            validation_metrics=_validation(),
            paired_baseline=_paired_digest(),
            reference_checkpoint=reference,
            candidate_path=run / "contextual_risk_candidates.json",
            selection_path=run / "best_contextual_risk_checkpoint.json",
        )


def test_continuation_requires_exact_epoch_two_pair_and_restores_human_epoch_four(
    tmp_path, monkeypatch
):
    run = tmp_path / "run"
    run.mkdir()
    reference = tmp_path / "a1.pt"
    torch.save({"backbone": torch.tensor([1.0])}, reference)
    reference_sha = file_sha256(reference)
    monkeypatch.setattr(
        main_task_retrieval, "CONTEXTUAL_RISK_CANONICAL_A1_SHA256", reference_sha
    )
    candidates = []
    pairs = []
    for epoch, rho in enumerate((0.20, 0.21, 0.22)):
        _, model, optimizer, payload = _record(
            run, reference, candidates, epoch, _validation(rho=rho)
        )
        pairs.append((model, optimizer, payload))
    main_task_retrieval.atomic_write_json(
        run / "a3_paired_baseline.json",
        {
            "reference_checkpoint": str(reference.resolve()),
            "reference_checkpoint_sha256": reference_sha,
            "paired_digest": _paired_digest(),
        },
    )
    model, optimizer_path, optimizer_payload = pairs[2]
    protocol = _initialize_contextual_risk_protocol(
        SimpleNamespace(
            output_dir=str(run),
            init_model=str(model),
            resume_model=str(optimizer_path),
            epochs=5,
        ),
        object(),
        object(),
        torch.device("cpu"),
        1,
        resume_payload=optimizer_payload,
    )

    assert protocol["resume_candidate"]["epoch"] == 2
    opt = SimpleNamespace(load_state_dict=lambda state: None)
    scheduler = SimpleNamespace(load_state_dict=lambda state: None)
    resumed_epoch, global_step = _restore_training_checkpoint(
        optimizer_payload, opt, scheduler, require_complete=True
    )
    assert resumed_epoch + 1 == 4
    assert global_step == 15

    with pytest.raises(ValueError, match="exactly one recorded candidate"):
        _initialize_contextual_risk_protocol(
            SimpleNamespace(
                output_dir=str(run),
                init_model=str(tmp_path / "copy.pt"),
                resume_model=str(optimizer_path),
                epochs=5,
            ),
            object(), object(), torch.device("cpu"), 1,
            resume_payload=optimizer_payload,
        )

    wrong_optimizer = tmp_path / "wrong-optimizer.pt"
    torch.save(optimizer_payload, wrong_optimizer)
    with pytest.raises(ValueError, match="optimizer checkpoint path mismatch"):
        _initialize_contextual_risk_protocol(
            SimpleNamespace(
                output_dir=str(run),
                init_model=str(model),
                resume_model=str(wrong_optimizer),
                epochs=5,
            ),
            object(), object(), torch.device("cpu"), 1,
            resume_payload=optimizer_payload,
        )

    candidate_path = run / "contextual_risk_candidates.json"
    selection_path = run / "best_contextual_risk_checkpoint.json"
    original_candidates = json.loads(candidate_path.read_text())
    original_selection = json.loads(selection_path.read_text())
    tampered = copy.deepcopy(original_candidates)
    tampered[2]["paired_digest"] = _paired_digest("forged")
    tampered[2]["paired_invariance"] = {
        "passed": True,
        "checks": original_candidates[2]["paired_invariance"]["checks"],
    }
    forged_selection = copy.deepcopy(original_selection)
    forged_selection["candidates"] = tampered
    forged_selection["winner"] = tampered[2]
    main_task_retrieval.atomic_write_json(candidate_path, tampered)
    main_task_retrieval.atomic_write_json(selection_path, forged_selection)
    with pytest.raises(ValueError, match="immutable candidate audit mismatch"):
        _initialize_contextual_risk_protocol(
            SimpleNamespace(
                output_dir=str(run), init_model=str(model),
                resume_model=str(optimizer_path), epochs=5,
            ),
            object(), object(), torch.device("cpu"), 1,
            resume_payload=optimizer_payload,
        )

    nonfinite = copy.deepcopy(original_candidates)
    nonfinite[2]["evidence"]["t2v"]["risk"][0] = float("nan")
    nonfinite_selection = copy.deepcopy(original_selection)
    nonfinite_selection["candidates"] = nonfinite
    nonfinite_selection["winner"] = nonfinite[2]
    main_task_retrieval.atomic_write_json(candidate_path, nonfinite)
    main_task_retrieval.atomic_write_json(selection_path, nonfinite_selection)
    with pytest.raises(ValueError, match="immutable candidate audit mismatch"):
        _initialize_contextual_risk_protocol(
            SimpleNamespace(
                output_dir=str(run), init_model=str(model),
                resume_model=str(optimizer_path), epochs=5,
            ),
            object(), object(), torch.device("cpu"), 1,
            resume_payload=optimizer_payload,
        )


def test_later_valid_candidates_can_win_but_invalid_candidate_cannot(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    reference = tmp_path / "a1.pt"
    torch.save({"backbone": torch.tensor([1.0])}, reference)
    candidates = []
    for epoch, rho in enumerate((0.20, 0.21, 0.22, 0.40)):
        winner, *_ = _record(
            run, reference, candidates, epoch, _validation(rho=rho)
        )
    assert winner["epoch"] == 3

    invalid = _validation(rho=0.99, raw="changed")
    winner, *_ = _record(run, reference, candidates, 4, invalid)
    assert winner["epoch"] == 3
    assert candidates[4]["valid"] is False


def test_selection_receives_validation_candidate_only(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    reference = tmp_path / "a1.pt"
    torch.save({"backbone": torch.tensor([1.0])}, reference)
    model, optimizer, _ = _write_pair(run, reference, 0)
    observed = []

    def select(candidates, *, output_dir):
        observed.extend(copy.deepcopy(candidates))
        return candidates[0]

    monkeypatch.setattr(main_task_retrieval, "choose_contextual_risk_checkpoint", select)
    metrics = _validation()
    metrics["test_metrics"] = {"t2v": {"R1": 99.0}}
    _record_contextual_risk_selection(
        epoch=0,
        checkpoint_path=model,
        optimizer_checkpoint_path=optimizer,
        validation_metrics=metrics,
        paired_baseline=_paired_digest(),
        reference_checkpoint=reference,
        candidate_path=run / "contextual_risk_candidates.json",
        selection_path=run / "best_contextual_risk_checkpoint.json",
    )

    assert len(observed) == 1
    assert "test_metrics" not in observed[0]
    assert set(observed[0]["metrics"]) == {
        "t2v", "v2t", "paired_invariance", "non_a3_mismatches"
    }


def test_final_test_gate_calls_jsfusion_once_only_on_validation_pass(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "pytorch_model.bin.0"
    checkpoint.write_bytes(b"model")
    winner = {
        "epoch": 0,
        "path": str(checkpoint.resolve()),
        "model_sha256": file_sha256(checkpoint),
        "metrics": {
            **_validation()["contextual_risk"],
            "paired_invariance": {"passed": True},
            "non_a3_mismatches": [],
        },
    }
    selection = {"winner": winner, "candidates": [winner]}
    calls = []
    args = SimpleNamespace(run_final_test=True, output_dir=str(tmp_path))

    result = _maybe_run_contextual_risk_final_test(
        args, selection, lambda selected: calls.append(selected) or {"test": 1}
    )

    assert result == {"test": 1}
    assert calls == [winner]
    claim = json.loads((tmp_path / "a3_final_test_claim.json").read_text())
    assert claim["status"] == "completed"

    with pytest.raises(RuntimeError, match="already been claimed"):
        _maybe_run_contextual_risk_final_test(
            args,
            selection,
            lambda selected: calls.append(selected),
        )
    assert calls == [winner]

    failed = copy.deepcopy(selection)
    failed["winner"]["metrics"]["v2t"]["risk_stats"]["std"] = 0.0
    calls.clear()
    monkeypatch.setattr(
        main_task_retrieval,
        "logger",
        SimpleNamespace(warning=lambda *args: None),
        raising=False,
    )
    assert _maybe_run_contextual_risk_final_test(
        args, failed, lambda selected: calls.append(selected)
    ) is None
    assert calls == []

    calls.clear()
    assert _maybe_run_contextual_risk_final_test(
        args, {"winner": None, "candidates": []},
        lambda selected: calls.append(selected),
    ) is None
    assert calls == []


def test_failed_final_test_claim_cannot_be_retried(tmp_path):
    checkpoint = tmp_path / "pytorch_model.bin.0"
    checkpoint.write_bytes(b"model")
    winner = {
        "epoch": 0,
        "path": str(checkpoint.resolve()),
        "model_sha256": file_sha256(checkpoint),
        "metrics": {
            **_validation()["contextual_risk"],
            "paired_invariance": {"passed": True},
            "non_a3_mismatches": [],
        },
    }
    selection = {"winner": winner, "candidates": [winner]}
    args = SimpleNamespace(run_final_test=True, output_dir=str(tmp_path))

    with pytest.raises(OSError, match="test failed"):
        _maybe_run_contextual_risk_final_test(
            args, selection, lambda selected: (_ for _ in ()).throw(
                OSError("test failed")
            )
        )
    claim = json.loads((tmp_path / "a3_final_test_claim.json").read_text())
    assert claim["status"] == "failed"
    assert claim["error_type"] == "OSError"

    with pytest.raises(RuntimeError, match="already been claimed"):
        _maybe_run_contextual_risk_final_test(args, selection, lambda selected: None)


def test_final_test_is_deferred_when_three_epoch_continuation_is_eligible(
    tmp_path, monkeypatch
):
    candidates = []
    for epoch, rho in enumerate((0.20, 0.21, 0.22)):
        checkpoint = tmp_path / f"pytorch_model.bin.{epoch}"
        checkpoint.write_bytes(f"model-{epoch}".encode())
        candidates.append({
            "epoch": epoch,
            "path": str(checkpoint.resolve()),
            "model_sha256": file_sha256(checkpoint),
            "rho": [rho, rho],
            "auroc": [0.70, 0.70],
            "aurc": [0.25, 0.25],
            "metrics": {
                **_validation(rho=rho)["contextual_risk"],
                "paired_invariance": {"passed": True},
                "non_a3_mismatches": [],
            },
        })
    calls = []
    monkeypatch.setattr(
        main_task_retrieval,
        "logger",
        SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
        raising=False,
    )

    result = _maybe_run_contextual_risk_final_test(
        SimpleNamespace(run_final_test=True, output_dir=str(tmp_path)),
        {"winner": candidates[2], "candidates": candidates},
        lambda selected: calls.append(selected),
    )

    assert result is None
    assert calls == []


def test_a3_save_refuses_checkpoint_overwrite(tmp_path, monkeypatch):
    existing = tmp_path / "pytorch_model.bin.0"
    existing.write_bytes(b"immutable")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    monkeypatch.setattr(
        main_task_retrieval,
        "logger",
        SimpleNamespace(info=lambda *args: None),
        raising=False,
    )

    with pytest.raises(FileExistsError, match="append-only"):
        save_model(
            0,
            SimpleNamespace(
                output_dir=str(tmp_path),
                csd_logvar_sidecar_only=False,
                contextual_risk_sidecar_only=True,
            ),
            model,
            optimizer,
            tr_loss=0.5,
        )
    assert existing.read_bytes() == b"immutable"
