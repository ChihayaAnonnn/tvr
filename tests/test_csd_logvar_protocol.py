import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import main_task_retrieval
from main_task_retrieval import (
    _build_csd_logvar_candidate,
    _build_training_checkpoint_payload,
    _evaluate_selected_test_checkpoints,
    _initialize_csd_logvar_protocol,
    _record_csd_logvar_selection,
    _restore_training_checkpoint,
)
from prob_models.csd_logvar_protocol import (
    array_sha256,
    choose_csd_logvar_checkpoint,
    compare_paired_invariance,
    csd_logvar_acceptance,
    file_sha256,
    non_csd_state_dict_mismatches,
    numeric_metrics_are_finite,
    validate_csd_logvar_resume_lineage,
)


def test_array_digest_changes_on_one_value():
    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    second = first.copy()
    second[1, 2] += 1

    assert array_sha256(first) != array_sha256(second)


def test_array_digest_includes_dtype_and_shape():
    flat = np.arange(6, dtype=np.float32)

    assert array_sha256(flat) != array_sha256(flat.reshape(2, 3))
    assert array_sha256(flat) != array_sha256(flat.astype(np.float64))


def test_file_digest_changes_on_one_byte(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"checkpoint-a")
    second.write_bytes(b"checkpoint-b")

    assert file_sha256(first) != file_sha256(second)


def _lineage_fixture(tmp_path):
    model_path = tmp_path / "pytorch_model.bin.2"
    optimizer_path = tmp_path / "pytorch_opt.bin.2"
    torch.save({"backbone": torch.tensor([1.0])}, model_path)
    optimizer_payload = {
        "epoch": 2,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"last_epoch": 3},
        "global_step": 12,
    }
    torch.save(optimizer_payload, optimizer_path)
    candidate = {
        "epoch": 2,
        "path": str(model_path.resolve()),
        "model_sha256": file_sha256(model_path),
        "optimizer_path": str(optimizer_path.resolve()),
        "optimizer_sha256": file_sha256(optimizer_path),
        "non_a2_mismatches": [],
    }
    return model_path, optimizer_path, optimizer_payload, candidate


def test_resume_lineage_accepts_exact_recorded_model_optimizer_and_epoch(
    tmp_path,
):
    model_path, optimizer_path, payload, candidate = _lineage_fixture(tmp_path)

    selected = validate_csd_logvar_resume_lineage(
        [candidate],
        output_dir=tmp_path,
        model_checkpoint=model_path,
        optimizer_checkpoint=optimizer_path,
        optimizer_payload=payload,
    )

    assert selected == candidate


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unrecorded_model", "exactly one recorded candidate"),
        ("model_hash", "model checkpoint SHA-256"),
        ("optimizer_path", "optimizer checkpoint path"),
        ("optimizer_hash", "optimizer checkpoint SHA-256"),
        ("optimizer_epoch", "optimizer checkpoint epoch"),
        ("optimizer_epoch_missing", "optimizer checkpoint epoch"),
        ("non_a2_mismatch", "non-A2"),
    ],
)
def test_resume_lineage_rejects_any_mismatch(tmp_path, mutation, message):
    model_path, optimizer_path, payload, candidate = _lineage_fixture(tmp_path)
    requested_model = model_path
    requested_optimizer = optimizer_path
    if mutation == "unrecorded_model":
        requested_model = tmp_path / "pytorch_model.bin.copy"
        requested_model.write_bytes(model_path.read_bytes())
    elif mutation == "model_hash":
        model_path.write_bytes(b"mutated-model")
    elif mutation == "optimizer_path":
        requested_optimizer = tmp_path / "pytorch_opt.bin.copy"
        requested_optimizer.write_bytes(optimizer_path.read_bytes())
    elif mutation == "optimizer_hash":
        optimizer_path.write_bytes(b"mutated-optimizer")
    elif mutation == "optimizer_epoch":
        payload = {**payload, "epoch": 3}
    elif mutation == "optimizer_epoch_missing":
        payload = {key: value for key, value in payload.items() if key != "epoch"}
    elif mutation == "non_a2_mismatch":
        candidate = {**candidate, "non_a2_mismatches": ["backbone"]}

    with pytest.raises(ValueError, match=message):
        validate_csd_logvar_resume_lineage(
            [candidate],
            output_dir=tmp_path,
            model_checkpoint=requested_model,
            optimizer_checkpoint=requested_optimizer,
            optimizer_payload=payload,
        )


def test_state_audit_ignores_only_a2_keys(tmp_path):
    base = {
        "backbone": torch.tensor([1.0]),
        "retrieval_risk_sidecar.w": torch.tensor([2.0]),
    }
    candidate = {
        **base,
        "csd_logvar_sidecar.raw_scale": torch.tensor(3.0),
    }
    base_path = tmp_path / "base.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(base, base_path)
    torch.save(candidate, candidate_path)

    assert non_csd_state_dict_mismatches(base_path, candidate_path) == []

    candidate["backbone"] = torch.tensor([1.5])
    torch.save(candidate, candidate_path)
    assert non_csd_state_dict_mismatches(base_path, candidate_path) == [
        "backbone"
    ]


def test_state_audit_casts_reference_to_candidate_dtype(tmp_path):
    base_path = tmp_path / "base.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save({"backbone": torch.tensor([1.5], dtype=torch.float16)}, base_path)
    torch.save({"backbone": torch.tensor([1.5], dtype=torch.float32)}, candidate_path)

    assert non_csd_state_dict_mismatches(base_path, candidate_path) == []


def test_state_audit_reports_missing_and_extra_non_a2_keys(tmp_path):
    base_path = tmp_path / "base.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save({"backbone": torch.tensor([1.0])}, base_path)
    torch.save({"head": torch.tensor([1.0])}, candidate_path)

    assert non_csd_state_dict_mismatches(base_path, candidate_path) == [
        "backbone",
        "head",
    ]


def test_paired_invariance_reports_each_exact_check():
    baseline = {
        "raw_wti_sha256": "wti",
        "a1_failure_risk_sha256": {"t2v": "tf", "v2t": "vf"},
        "a1_vacuity_sha256": {"t2v": "tv", "v2t": "vv"},
        "retrieval": {"t2v": {"R1": 1.0}, "v2t": {"R1": 2.0}},
    }
    candidate = {
        **baseline,
        "a1_vacuity_sha256": {"t2v": "changed", "v2t": "vv"},
    }

    comparison = compare_paired_invariance(baseline, candidate)

    assert comparison == {
        "passed": False,
        "checks": {
            "raw_wti": True,
            "a1_failure_risk": True,
            "a1_vacuity": False,
            "retrieval": True,
        },
    }


def test_selector_uses_min_rho_then_mean_auroc_then_earlier_epoch():
    candidates = [
        {
            "epoch": 0,
            "path": "a",
            "valid": True,
            "rho": [0.30, 0.21],
            "auroc": [0.70, 0.70],
        },
        {
            "epoch": 1,
            "path": "b",
            "valid": True,
            "rho": [0.22, 0.22],
            "auroc": [0.65, 0.65],
        },
        {
            "epoch": 2,
            "path": "c",
            "valid": True,
            "rho": [0.22, 0.22],
            "auroc": [0.80, 0.80],
        },
        {
            "epoch": 3,
            "path": "d",
            "valid": True,
            "rho": [0.22, 0.22],
            "auroc": [0.80, 0.80],
        },
    ]

    assert choose_csd_logvar_checkpoint(candidates)["path"] == "c"


def test_selector_rejects_invalid_candidates():
    with pytest.raises(ValueError, match="valid"):
        choose_csd_logvar_checkpoint(
            [
                {
                    "epoch": 0,
                    "path": "a",
                    "valid": False,
                    "rho": [0.9, 0.9],
                    "auroc": [0.9, 0.9],
                }
            ]
        )


def _acceptance_metrics():
    direction = {
        "spearman_uncertainty_vs_rank": 0.3,
        "failure_auroc": {"value": 0.7},
        "risk_coverage": {
            "accuracy_at_0.25": 1.0,
            "accuracy_at_0.50": 0.9,
            "accuracy_at_0.75": 0.8,
            "accuracy_at_1.00": 0.7,
        },
    }
    distribution = {
        "variance": {"std": 0.1},
        "clamp_low_fraction": 0.01,
        "clamp_high_fraction": 0.02,
    }
    return {
        "t2v": direction,
        "v2t": direction,
        "text_distribution": distribution,
        "video_distribution": distribution,
    }


def _paired_digest(raw_wti="wti"):
    return {
        "raw_wti_sha256": raw_wti,
        "a1_failure_risk_sha256": {"t2v": "tf", "v2t": "vf"},
        "a1_vacuity_sha256": {"t2v": "tv", "v2t": "vv"},
        "retrieval": {"t2v": {"R1": 1.0}, "v2t": {"R1": 2.0}},
    }


def test_acceptance_requires_all_bidirectional_quality_checks():
    accepted = csd_logvar_acceptance(_acceptance_metrics())
    rejected_metrics = _acceptance_metrics()
    rejected_metrics["v2t"] = {
        **rejected_metrics["v2t"],
        "spearman_uncertainty_vs_rank": 0.19,
    }
    rejected = csd_logvar_acceptance(rejected_metrics)

    assert accepted["passed"] is True
    assert all(accepted["checks"].values())
    assert rejected["passed"] is False
    assert rejected["checks"]["rho"] is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"metric": [1, 2.0, np.float32(3)]}, True),
        ({"metric": float("nan")}, False),
        ({"metric": torch.tensor(float("inf"))}, False),
    ],
)
def test_numeric_metric_finiteness_is_recursive(value, expected):
    assert numeric_metrics_are_finite(value) is expected


def test_candidate_requires_finite_paired_audited_low_clamp_metrics(tmp_path):
    reference_path = tmp_path / "a1.pt"
    candidate_path = tmp_path / "a2.pt"
    optimizer_path = tmp_path / "a2-opt.pt"
    torch.save({"backbone": torch.tensor([1.0])}, reference_path)
    torch.save(
        {
            "backbone": torch.tensor([1.0]),
            "csd_logvar_sidecar.weight": torch.tensor([2.0]),
        },
        candidate_path,
    )
    optimizer_path.write_bytes(b"optimizer")
    validation_metrics = {
        "t2v": {"R1": 1.0},
        "v2t": {"R1": 2.0},
        "paired_digest": _paired_digest(),
        "csd_logvar": _acceptance_metrics(),
    }

    candidate = _build_csd_logvar_candidate(
        epoch=2,
        checkpoint_path=candidate_path,
        optimizer_checkpoint_path=optimizer_path,
        validation_metrics=validation_metrics,
        paired_baseline=_paired_digest(),
        reference_checkpoint=reference_path,
    )

    assert candidate["valid"] is True
    assert candidate["path"] == str(candidate_path.resolve())
    assert candidate["model_sha256"] == file_sha256(candidate_path)
    assert candidate["optimizer_path"] == str(optimizer_path.resolve())
    assert candidate["optimizer_sha256"] == file_sha256(optimizer_path)
    assert candidate["rho"] == [0.3, 0.3]
    assert candidate["auroc"] == [0.7, 0.7]
    assert candidate["non_a2_mismatches"] == []
    assert candidate["acceptance"]["passed"] is True

    validation_metrics["paired_digest"] = _paired_digest("changed")
    invalid = _build_csd_logvar_candidate(
        epoch=3,
        checkpoint_path=candidate_path,
        optimizer_checkpoint_path=optimizer_path,
        validation_metrics=validation_metrics,
        paired_baseline=_paired_digest(),
        reference_checkpoint=reference_path,
    )
    assert invalid["valid"] is False
    assert invalid["paired_invariance"]["checks"]["raw_wti"] is False

    nonfinite_metrics = copy.deepcopy(validation_metrics)
    nonfinite_metrics["paired_digest"] = _paired_digest()
    nonfinite_metrics["csd_logvar"]["t2v"][
        "spearman_uncertainty_vs_rank"
    ] = float("nan")
    nonfinite = _build_csd_logvar_candidate(
        epoch=4,
        checkpoint_path=candidate_path,
        optimizer_checkpoint_path=optimizer_path,
        validation_metrics=nonfinite_metrics,
        paired_baseline=_paired_digest(),
        reference_checkpoint=reference_path,
    )
    assert nonfinite["finite"] is False
    assert nonfinite["valid"] is False

    clamped_metrics = copy.deepcopy(validation_metrics)
    clamped_metrics["paired_digest"] = _paired_digest()
    clamped_metrics["csd_logvar"]["text_distribution"][
        "clamp_low_fraction"
    ] = 0.03
    clamped_metrics["csd_logvar"]["text_distribution"][
        "clamp_high_fraction"
    ] = 0.02
    clamped = _build_csd_logvar_candidate(
        epoch=5,
        checkpoint_path=candidate_path,
        optimizer_checkpoint_path=optimizer_path,
        validation_metrics=clamped_metrics,
        paired_baseline=_paired_digest(),
        reference_checkpoint=reference_path,
    )
    assert clamped["clamp_ok"] is False
    assert clamped["valid"] is False

    torch.save(
        {
            "backbone": torch.tensor([9.0]),
            "csd_logvar_sidecar.weight": torch.tensor([2.0]),
        },
        candidate_path,
    )
    mismatched = _build_csd_logvar_candidate(
        epoch=6,
        checkpoint_path=candidate_path,
        optimizer_checkpoint_path=optimizer_path,
        validation_metrics=validation_metrics,
        paired_baseline=_paired_digest("changed"),
        reference_checkpoint=reference_path,
    )
    assert mismatched["non_a2_mismatches"] == ["backbone"]
    assert mismatched["valid"] is False


def test_fresh_protocol_registers_a1_baseline_without_intrinsic_metrics(
    tmp_path, monkeypatch
):
    reference_path = tmp_path / "a1.pt"
    torch.save({"backbone": torch.tensor([1.0])}, reference_path)
    args = SimpleNamespace(
        output_dir=str(tmp_path / "run"),
        init_model=str(reference_path),
        resume_model=None,
    )
    calls = []

    def fake_eval(*call_args, **kwargs):
        calls.append((call_args, kwargs))
        return {"paired_digest": _paired_digest()}

    monkeypatch.setattr(main_task_retrieval, "eval_epoch", fake_eval)

    protocol = _initialize_csd_logvar_protocol(
        args, object(), object(), torch.device("cpu"), 1
    )

    assert protocol["paired_baseline"] == _paired_digest()
    assert protocol["reference_checkpoint"] == str(reference_path)
    assert protocol["candidates"] == []
    assert calls[0][1]["report_csd_logvar"] is False
    registration = json.loads(
        (tmp_path / "run" / "a1_paired_baseline.json").read_text()
    )
    assert registration == {
        "reference_checkpoint": str(reference_path),
        "paired_digest": _paired_digest(),
    }


def test_fresh_protocol_rejects_existing_lineage_artifacts(tmp_path):
    run_path = tmp_path / "run"
    run_path.mkdir()
    main_task_retrieval.atomic_write_json(
        run_path / "a1_paired_baseline.json",
        {
            "reference_checkpoint": str(tmp_path / "old-a1.pt"),
            "paired_digest": _paired_digest("stale"),
        },
    )
    args = SimpleNamespace(
        output_dir=str(run_path),
        init_model=str(tmp_path / "canonical-a1.pt"),
        resume_model=None,
    )

    with pytest.raises(FileExistsError, match="fresh A2 run"):
        _initialize_csd_logvar_protocol(
            args, object(), object(), torch.device("cpu"), 1
        )


def test_continuation_reuses_original_reference_and_reaudits_candidates(
    tmp_path, monkeypatch
):
    run_path = tmp_path / "run"
    run_path.mkdir()
    reference_path = tmp_path / "original-a1.pt"
    candidate_path = run_path / "pytorch_model.bin.0"
    optimizer_path = run_path / "pytorch_opt.bin.0"
    torch.save({"backbone": torch.tensor([1.0])}, reference_path)
    torch.save({"backbone": torch.tensor([1.0])}, candidate_path)
    optimizer_payload = {
        "epoch": 0,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"last_epoch": 1},
        "global_step": 5,
    }
    torch.save(optimizer_payload, optimizer_path)
    main_task_retrieval.atomic_write_json(
        run_path / "a1_paired_baseline.json",
        {
            "reference_checkpoint": str(reference_path),
            "paired_digest": _paired_digest(),
        },
    )
    stored_candidate = {
        "epoch": 0,
        "path": str(candidate_path.resolve()),
        "model_sha256": file_sha256(candidate_path),
        "optimizer_path": str(optimizer_path.resolve()),
        "optimizer_sha256": file_sha256(optimizer_path),
        "valid": False,
        "finite": True,
        "clamp_ok": True,
        "paired_invariance": {"passed": True, "checks": {}},
        "non_a2_mismatches": ["stale-record"],
        "rho": [0.3, 0.3],
        "auroc": [0.7, 0.7],
    }
    main_task_retrieval.atomic_write_json(
        run_path / "best_csd_logvar_checkpoint.json",
        {"candidates": [stored_candidate], "winner": stored_candidate},
    )
    args = SimpleNamespace(
        output_dir=str(run_path),
        init_model=str(candidate_path),
        resume_model=str(optimizer_path),
    )
    monkeypatch.setattr(
        main_task_retrieval,
        "eval_epoch",
        lambda *args, **kwargs: pytest.fail("continuation must reuse baseline"),
    )

    protocol = _initialize_csd_logvar_protocol(
        args,
        object(),
        object(),
        torch.device("cpu"),
        1,
        resume_payload=optimizer_payload,
    )

    assert protocol["reference_checkpoint"] == str(reference_path)
    assert protocol["candidates"][0]["non_a2_mismatches"] == []
    assert protocol["candidates"][0]["valid"] is True
    assert protocol["resume_candidate"] == protocol["candidates"][0]
    persisted = json.loads(
        (run_path / "best_csd_logvar_checkpoint.json").read_text()
    )
    assert persisted["candidates"] == protocol["candidates"]
    assert persisted["winner"] == protocol["candidates"][0]


def test_continuation_requires_original_candidate_list(tmp_path):
    run_path = tmp_path / "run"
    run_path.mkdir()
    main_task_retrieval.atomic_write_json(
        run_path / "a1_paired_baseline.json",
        {
            "reference_checkpoint": str(tmp_path / "a1.pt"),
            "paired_digest": _paired_digest(),
        },
    )
    args = SimpleNamespace(
        output_dir=str(run_path),
        init_model=str(tmp_path / "a2.pt"),
        resume_model=str(tmp_path / "optimizer.pt"),
    )

    with pytest.raises(FileNotFoundError, match="candidate selection"):
        _initialize_csd_logvar_protocol(
            args, object(), object(), torch.device("cpu"), 1
        )


def test_validation_record_writes_bidirectional_winner_after_each_epoch(
    tmp_path,
):
    reference_path = tmp_path / "a1.pt"
    candidate_path = tmp_path / "a2.pt"
    optimizer_path = tmp_path / "a2-opt.pt"
    selection_path = tmp_path / "best_csd_logvar_checkpoint.json"
    torch.save({"backbone": torch.tensor([1.0])}, reference_path)
    torch.save(
        {
            "backbone": torch.tensor([1.0]),
            "csd_logvar_sidecar.weight": torch.tensor([2.0]),
        },
        candidate_path,
    )
    optimizer_path.write_bytes(b"optimizer")
    metrics = {
        "t2v": {"R1": 1.0},
        "v2t": {"R1": 2.0},
        "paired_digest": _paired_digest(),
        "csd_logvar": _acceptance_metrics(),
    }
    candidates = []

    winner = _record_csd_logvar_selection(
        epoch=1,
        checkpoint_path=candidate_path,
        optimizer_checkpoint_path=optimizer_path,
        validation_metrics=metrics,
        paired_baseline=_paired_digest(),
        reference_checkpoint=reference_path,
        candidates=candidates,
        selection_path=selection_path,
    )

    payload = json.loads(selection_path.read_text())
    assert winner["path"] == str(candidate_path)
    assert payload["winner"] == winner
    assert payload["candidates"] == candidates
    assert payload["selection_metric"] == (
        "max(min(t2v_rho,v2t_rho),mean(t2v_auroc,v2t_auroc))"
    )


def test_invalid_first_epoch_is_recorded_and_later_epoch_can_win(tmp_path):
    reference_path = tmp_path / "a1.pt"
    first_path = tmp_path / "a2-0.pt"
    second_path = tmp_path / "a2-1.pt"
    first_optimizer_path = tmp_path / "a2-opt-0.pt"
    second_optimizer_path = tmp_path / "a2-opt-1.pt"
    selection_path = tmp_path / "best_csd_logvar_checkpoint.json"
    base = {"backbone": torch.tensor([1.0])}
    torch.save(base, reference_path)
    torch.save({**base, "csd_logvar_sidecar.weight": torch.tensor([1.0])}, first_path)
    torch.save({**base, "csd_logvar_sidecar.weight": torch.tensor([2.0])}, second_path)
    first_optimizer_path.write_bytes(b"optimizer-0")
    second_optimizer_path.write_bytes(b"optimizer-1")
    invalid_metrics = {
        "t2v": {"R1": 1.0},
        "v2t": {"R1": 2.0},
        "paired_digest": _paired_digest("changed"),
        "csd_logvar": _acceptance_metrics(),
    }
    valid_metrics = {
        **invalid_metrics,
        "paired_digest": _paired_digest(),
    }
    candidates = []

    first_winner = _record_csd_logvar_selection(
        epoch=0,
        checkpoint_path=first_path,
        optimizer_checkpoint_path=first_optimizer_path,
        validation_metrics=invalid_metrics,
        paired_baseline=_paired_digest(),
        reference_checkpoint=reference_path,
        candidates=candidates,
        selection_path=selection_path,
    )
    first_payload = json.loads(selection_path.read_text())
    second_winner = _record_csd_logvar_selection(
        epoch=1,
        checkpoint_path=second_path,
        optimizer_checkpoint_path=second_optimizer_path,
        validation_metrics=valid_metrics,
        paired_baseline=_paired_digest(),
        reference_checkpoint=reference_path,
        candidates=candidates,
        selection_path=selection_path,
    )

    assert first_winner is None
    assert first_payload["winner"] is None
    assert first_payload["candidates"][0]["valid"] is False
    assert second_winner["path"] == str(second_path)
    assert len(json.loads(selection_path.read_text())["candidates"]) == 2


def test_a2_final_test_loads_one_winner_and_evaluates_both_directions(
    tmp_path, monkeypatch
):
    checkpoint_path = tmp_path / "winner.pt"
    checkpoint_path.write_bytes(b"placeholder")
    loaded = []
    evaluations = []

    class _Model:
        def load_state_dict(self, state):
            loaded.append(state)

        def to(self, device):
            return self

    monkeypatch.setattr(
        main_task_retrieval.torch,
        "load",
        lambda path, map_location: {"loaded_from": str(path)},
    )
    monkeypatch.setattr(
        main_task_retrieval,
        "logger",
        SimpleNamespace(info=lambda *args: None),
        raising=False,
    )

    def fake_eval(*args, **kwargs):
        evaluations.append(kwargs["directions"])
        return {"t2v": {"R1": 9.0}, "v2t": {"R1": 8.0}}

    monkeypatch.setattr(main_task_retrieval, "eval_epoch", fake_eval)
    selection = {
        "selection_split": "val",
        "winner": {"path": str(checkpoint_path), "rho": [0.3, 0.4]},
    }

    payload = _evaluate_selected_test_checkpoints(
        SimpleNamespace(),
        _Model(),
        object(),
        torch.device("cpu"),
        1,
        selection,
        csd_logvar_sidecar_only=True,
    )

    assert loaded == [{"loaded_from": str(checkpoint_path)}]
    assert evaluations == [("t2v", "v2t")]
    assert payload == {
        "test_split": "test",
        "selection": selection,
        "test_metrics": {"t2v": {"R1": 9.0}, "v2t": {"R1": 8.0}},
    }


def _optimizer_and_scheduler(steps=1):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 0.5**step
    )
    for _ in range(steps):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
    return optimizer, scheduler


def test_training_checkpoint_round_trip_restores_scheduler_and_global_step():
    optimizer, scheduler = _optimizer_and_scheduler()
    payload = _build_training_checkpoint_payload(
        epoch=2,
        optimizer=optimizer,
        scheduler=scheduler,
        tr_loss=1.25,
        global_step=17,
    )
    restored_optimizer, restored_scheduler = _optimizer_and_scheduler(steps=3)

    resumed_epoch, global_step = _restore_training_checkpoint(
        payload,
        restored_optimizer,
        restored_scheduler,
        require_complete=True,
    )

    assert resumed_epoch == 3
    assert global_step == 17
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    restored_state = restored_optimizer.state_dict()
    original_state = optimizer.state_dict()
    assert restored_state["param_groups"] == original_state["param_groups"]
    assert torch.equal(
        restored_state["state"][0]["exp_avg"],
        original_state["state"][0]["exp_avg"],
    )


@pytest.mark.parametrize("missing", ["scheduler_state_dict", "global_step"])
def test_a2_resume_rejects_incomplete_training_checkpoint(missing):
    optimizer, scheduler = _optimizer_and_scheduler()
    payload = _build_training_checkpoint_payload(
        epoch=1,
        optimizer=optimizer,
        scheduler=scheduler,
        tr_loss=0.5,
        global_step=9,
    )
    del payload[missing]
    restored_optimizer, restored_scheduler = _optimizer_and_scheduler()

    with pytest.raises(ValueError, match=missing):
        _restore_training_checkpoint(
            payload,
            restored_optimizer,
            restored_scheduler,
            require_complete=True,
        )


def test_legacy_resume_allows_missing_scheduler_and_global_step():
    optimizer, scheduler = _optimizer_and_scheduler()
    legacy = {
        "epoch": 4,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": 0.25,
    }
    restored_optimizer, restored_scheduler = _optimizer_and_scheduler()
    initial_scheduler_state = restored_scheduler.state_dict()

    resumed_epoch, global_step = _restore_training_checkpoint(
        legacy,
        restored_optimizer,
        restored_scheduler,
        require_complete=False,
    )

    assert (resumed_epoch, global_step) == (5, 0)
    assert restored_scheduler.state_dict() == initial_scheduler_state
